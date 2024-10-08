import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from dataset.celebv_hq import CelebvHqDataModule
from marlin_pytorch.config import resolve_config
from marlin_pytorch.util import read_yaml
from model.classifier import Classifier
from util.earlystop_lr import EarlyStoppingLR
from util.lr_logger import LrLogger
from util.seed import Seed
from util.system_stats_logger import SystemStatsLogger


from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix, CohenKappa, Recall, Recall, AveragePrecision, Precision

def train_celebvhq(args, config):
    data_path = args.data_path
    resume_ckpt = args.resume
    n_gpus = args.n_gpus
    max_epochs = args.epochs

    finetune = config["finetune"]
    learning_rate = config["learning_rate"]
    task = config["task"]

    if task == "appearance":
        num_classes = 2
    elif task == "action":
        # num_classes = 35
        num_classes = 2
    else:
        raise ValueError(f"Unknown task {task}")

    if finetune:
        backbone_config = resolve_config(config["backbone"])

        model = Classifier(
            num_classes, config["backbone"], True, args.marlin_ckpt, "multiclass", config["learning_rate"],
            args.n_gpus > 1,
        )

        dm = CelebvHqDataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_frames=backbone_config.n_frames,
            temporal_sample_rate=48
        )

    else:
        model = Classifier(
            num_classes, config["backbone"], False,
            None, "multilabel", config["learning_rate"], args.n_gpus > 1,
        )

        dm = CelebvHqDataModule(
            data_path, finetune, task,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            feature_dir=config["backbone"],
            temporal_reduction=config["temporal_reduction"]
        )

    if args.skip_train:
        dm.setup()
        return resume_ckpt, dm

    strategy = None if n_gpus <= 1 else "ddp"
    accelerator = "cpu" if n_gpus == 0 else "gpu"

    ckpt_filename = config["model_name"] + "-{epoch}-{val_acc:.4f}-{val_auc:.4f}"
    ckpt_monitor = "val_acc"

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    ckpt_callback = ModelCheckpoint(dirpath=f"ckpt/{config['model_name']}", save_last=True,
        filename=ckpt_filename,
        monitor=ckpt_monitor,
        save_top_k=3,
        mode="max")

    trainer = Trainer(log_every_n_steps=1, devices=n_gpus, accelerator=accelerator, benchmark=True, detect_anomaly=True,
        logger=True, precision=precision, max_epochs=max_epochs,
        strategy=strategy, resume_from_checkpoint=resume_ckpt,
        callbacks=[ckpt_callback, LrLogger(), EarlyStoppingLR(1e-6), SystemStatsLogger()])
        # callbacks=[ckpt_callback, LrLogger(), SystemStatsLogger()])

    trainer.fit(model, dm)

    return ckpt_callback.best_model_path, dm


def evaluate_celebvhq(args, ckpt, dm):
    print("Load checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    accelerator = "cpu" if args.n_gpus == 0 else "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    # collect predictions
    preds = trainer.predict(model, dm.val_dataloader())
    preds = torch.cat(preds)

    # collect ground truth
    ys = torch.zeros_like(preds, dtype=torch.long)
    for i, (_,_,_, y) in enumerate(tqdm(dm.val_dataloader())):
        ys[i * args.batch_size: (i + 1) * args.batch_size] = y

    preds = preds.sigmoid()
    acc = ((preds > 0.5) == ys).float().mean()
    

    # Convert predicted probabilities to class indices by taking the argmax
    preds_bool = torch.argmax(preds, dim=1)
    ys = torch.argmax(ys, dim=1)

    # acc = (preds_bool == ys).float().mean()
    # print(preds_bool, ys)
    # acc = ((preds > 0.5) == ys).float().mean()
    # accuracy = Accuracy(task="multiclass", num_classes=5)
    # acc = accuracy(preds_bool, ys)
    # # auc = model.auc_fn(preds_bool, torch.argmax(ys, dim=1))
    # print('dadadasdasdasdas', acc)
    # f1 = F1Score(task="multiclass", num_classes=3)
    # f1score = f1(preds_bool, ys)
    # average_precision = AveragePrecision(task="multiclass", num_classes=5, average=None)
    # precision = average_precision(preds_bool, torch.argmax(ys, dim=1))
    # recall = Recall(task="multiclass", average='micro', num_classes=3)
    # recall(preds_bool, ys)
    # cohenkappa = CohenKappa(task="multiclass", num_classes=3)
    # kappa = cohenkappa(preds_bool, ys)
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve
    import numpy as np
    # Convert one-hot encoded labels to class indices
    y_true = ys.numpy()
    y_pred = preds_bool.numpy()
    # from torcheval.metrics.functional import multiclass_f1_score
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1_scores = f1_score(y_true, y_pred, average=None)
    avg_f1_scores = np.mean(f1_scores)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    accuracy = accuracy_score(y_true, y_pred)

    # fpr, tpr, thresholds = roc_curve(y_true, preds, pos_label=2)
    # auc = auc(fpr, tpr)

    print(accuracy, precision, avg_precision, recall, avg_recall, f1_scores, avg_f1_scores)
    # print('adasd', multiclass_f1_score(preds_bool, ys, num_classes=3))
    # results = {
    #     "acc": acc,
    #     # "auc": auc,
    #     # "f1": f1score,
    #     # "precision": precision,
    #     # "recall": recall,
    #     # "kappa": kappa


    # }
    # print(results)
    # torch.set_printoptions(threshold=10_000)
    # print(preds_bool)
def test_celebvhq(args, ckpt, dm):
    print("Load checkpoint", ckpt)
    model = Classifier.load_from_checkpoint(ckpt)
    accelerator = "gpu"
    trainer = Trainer(log_every_n_steps=1, devices=1 if args.n_gpus > 0 else 0, accelerator=accelerator, benchmark=True,
        logger=False, enable_checkpointing=False)
    Seed.set(42)
    model.eval()

    # collect predictions
    preds = trainer.predict(model, dm.test_dataloader())
    # print('preds before: ', len(preds), preds[0])
    preds = torch.cat(preds)

    # collect ground truth
    ys = torch.zeros_like(preds, dtype=torch.long)
    paths = []
    for i, (_, path) in enumerate(tqdm(dm.test_dataloader())):
        # ys[i * args.batch_size: (i + 1) * args.batch_size] = y
        paths= [*paths, *path]

    preds = preds.sigmoid()
    preds_bool = torch.zeros_like(preds)
    preds_bool[torch.arange(preds_bool.size(0)), preds.argmax(dim=1)] = 1.

    import csv
    import numpy as np
    submit = []
    for i in range(len(preds)):
        # print(str(preds_bool[i] ))
        if str(preds_bool[i]) == 'tensor([0., 1., 0.])':
            label = 'Low_Pain'
        elif str(preds_bool[i]) == 'tensor([1., 0., 0.])':
            label = 'No_Pain'
        elif str(preds_bool[i]) == 'tensor([0., 0., 1.])':
            label = 'High_Pain'
        print(paths[i].split('/')[-2]+', '+paths[i].split('/')[-1].replace('.mp4', '')+', '+label, str(preds[i]))
        submit.append(paths[i].split('/')[-2]+', '+paths[i].split('/')[-1].replace('.mp4', '')+', '+ label)
    submit.sort()
    with open('{}.csv'.format('celebvhq_marlin_large_convtrans_fz_fc_unfz-epoch=363-val_acc=0.889-val_auc=0.427.ckpt'),'w') as file:
        for i in submit:
            # print(i, str(preds[i]))
            file.write(i)
            file.write('\n')
    # acc = (preds_bool == ys).float().mean()
    # print(preds_bool, ys)
    # acc = ((preds > 0.5) == ys).float().mean()
    # accuracy = Accuracy(task="multiclass", num_classes=3)
    # acc = accuracy(preds_bool, ys)
    # auc = model.auc_fn(preds_bool, torch.argmax(ys, dim=1))
    # results = {
    #     "acc": acc,
    #     "auc": auc
    # }
    # print(results, ys)

def train(args):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "celebvhq":
        ckpt, dm = train_celebvhq(args, config)
        evaluate_celebvhq(args, ckpt, dm)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

def evaluate(args, ckpt, dm):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "celebvhq":
        evaluate_celebvhq(args, ckpt, dm)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

def test(args, ckpt, dm):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "celebvhq":
        test_celebvhq(args, ckpt, dm)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ evaluation")
    parser.add_argument("--config", type=str, default='config/celebv_hq/action/celebvhq_marlin_action_ft.yaml', help="Path to CelebV-HQ evaluation config file.")
    parser.add_argument("--data_path", type=str, default='../BioVid-A', help="Path to CelebV-HQ dataset.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    parser.add_argument("--skip_train", action="store_true", default=False,
        help="Skip training and evaluate only.")

    args = parser.parse_args()
    if args.skip_train:
        assert args.resume is not None
    train(args)
    
    dm = CelebvHqDataModule(
        args.data_path, True, 'action',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_frames=16,
        temporal_sample_rate=48
    )
    dm.setup()
    evaluate(args,'/home/duke/Workspace/MARLIN/ckpt/Biovid-B_getgo_dualdnn/Biovid-B_getgo_dualdnn-epoch=70-val_acc=0.5427-val_auc=0.5427.ckpt',dm)
    # # #test(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_fc_unfz/celebvhq_marlin_large_convtrans_fz_fc_unfz-epoch=363-val_acc=0.889-val_auc=0.427.ckpt',dm)

    # # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48-epoch=17-val_acc=0.891-val_auc=0.930.ckpt',dm)
    # test(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64-epoch=120-val_acc=0.889-val_auc=0.861.ckpt',dm)

    # # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64-epoch=94-val_acc=0.888-val_auc=0.877.ckpt',dm)
    # # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48-epoch=17-val_acc=0.891-val_auc=0.930.ckpt',dm)
    # # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_64-epoch=120-val_acc=0.889-val_auc=0.861.ckpt',dm)
    # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_weight_samplerate_48/celebvhq_marlin_large_convtrans_fz_weight_samplerate_48-epoch=25-val_acc=0.881-val_auc=0.875.ckpt',dm)

    # evaluate(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48/celebvhq_marlin_large_convtrans_fz_noweight_samplerate_48-epoch=25-val_acc=0.889-val_auc=0.923.ckpt',dm)