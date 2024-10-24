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



def test(args, ckpt, dm):
    config = read_yaml(args.config)
    dataset_name = config["dataset"]

    if dataset_name == "celebvhq":
        test_celebvhq(args, ckpt, dm)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ evaluation")
    parser.add_argument("--config", type=str, default='/home/hdd1/duke/AI4pain/MARLIN/config/celebv_hq/action/celebvhq_marlin_action_ft.yaml', help="Path to CelebV-HQ evaluation config file.")
    parser.add_argument("--data_path", type=str, default='/home/hdd1/duke/AI4pain/BioVid-A', help="Path to CelebV-HQ dataset.")
    parser.add_argument("--marlin_ckpt", type=str, default=None,
        help="Path to MARLIN checkpoint. Default: None, load from online.")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs to train.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")

    args = parser.parse_args()
    if args.skip_train:
        assert args.resume is not None
    train(args)
    
    dm = CelebvHqDataModule(
        args.data_path, True, 'action',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_frames=16,
        temporal_sample_rate=4
    )
    dm.setup()

    test(args,'/home/hdd1/duke/AI4pain/MARLIN/ckpt/celebvhq_marlin_large_convtrans_fz_fc_unfz/celebvhq_marlin_large_convtrans_fz_fc_unfz-epoch=363-val_acc=0.889-val_auc=0.427.ckpt',dm)

  
