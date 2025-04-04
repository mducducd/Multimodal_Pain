from typing import Optional, Union, Sequence, Dict, Literal, Any
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Identity, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, AUROC
from copy import deepcopy
from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.trans_conv import ConvTran, Transformer, ConvTran_1
from model.lstm_fcn import FCN_model
from model.ts_transformer import TSTransformerEncoder
from model.AttentionBottleneckFusion import AttentionBottleneck
from model.crossatten import DCNLayer
from model.dual_dnn import DualCNNFusion
from vit_pytorch.vit_3d import ViT
from timesformer_pytorch import TimeSformer
from model.dual_lstm_att import DualBranchBiLSTM
# from vit_pytorch.vivit import ViT
from model.baselines import VGG3D

from ptflops import get_model_complexity_info

class Classifier(LightningModule):

    def __init__(self, num_classes: int, backbone: str, finetune: bool,
        marlin_ckpt: Optional[str] = None,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        learning_rate: float = 1e-4, distributed: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        if finetune:
            # if marlin_ckpt is None:
            #     self.model = Marlin.from_online(backbone).encoder
            # else:
            self.model = Marlin.from_file(backbone, '/home/duke/Workspace/MARLIN/ckpt/marlin_vit_small/last-v1.ckpt').encoder
        else:
            self.model = None

        for param in self.model.parameters():
            param.requires_grad = False

        config = resolve_config(backbone)
        self.ConvTran1 = ConvTran_1(config.encoder_embed_dim, 1568, emb_size=16, num_classes=num_classes)
        self.ConvTran = ConvTran_1(5, 2800, emb_size=16, num_classes=num_classes)
    
        ### fNISR model
        self.TSTransformerEncoder = TSTransformerEncoder(feat_dim=5, max_len=2800, d_model=128, n_heads=8, num_layers=3, dim_feedforward=256, dropout=0.1)
      
        self.learning_rate = learning_rate
        self.distributed = distributed
        self.task = task
        if task in "binary":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task=task, num_classes=1)
            self.auc_fn = AUROC(task=task, num_classes=1)
        elif task == "multiclass":
            self.loss_fn = CrossEntropyLoss()
            self.acc_fn = Accuracy(task="multiclass", num_classes=num_classes)
            self.auc_fn = AUROC(task="multiclass", num_classes=num_classes)
        elif task == "multilabel":
            self.loss_fn = BCEWithLogitsLoss()
            self.acc_fn = Accuracy(task="binary", num_classes=1)
            self.auc_fn = AUROC(task="binary", num_classes=1)

     
        self.coattn = DCNLayer(16, 16, 1, 0, 1568, 2800, num_classes)


    @classmethod
    def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
        return cls(model, learning_rate, distributed)

    def forward(self, x, fnirs_ft, padding_masks):
      
        if self.model is not None:
            feat = self.model.forward(x, True)
        else:
            feat = self.gap
        feat = self.ConvTran1(feat)
     
        fnirs_ft = self.TSTransformerEncoder(fnirs_ft, padding_masks)
       
        logits = self.ConvTran(fnirs_ft)
       
        logits = self.coattn(feat, logits)
       
        return logits
 

    def step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]]) -> Dict[str, Tensor]:
        x, fnirs, padding_masks,y = batch

        y_hat = self(x, fnirs, padding_masks)
        # if self.task == "multilabel":
        #     y_hat = y_hat.flatten()
        #     y = y.flatten()
        loss = self.loss_fn(y_hat, y.float())
        prob = y_hat.sigmoid()

        preds_bool = torch.argmax(prob, dim=1)
        y = torch.argmax(y, dim=1)

        acc = self.acc_fn(preds_bool, y)
      
        # auc = self.auc_fn(prob, torch.argmax(y, dim=1))

        return {"loss": loss, "acc": acc, "auc": acc}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        loss_dict = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return self(batch[0], batch[1], batch[2])

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=7, verbose=True, min_lr=1e-8),
                "monitor": "train_loss"
            }
        }
