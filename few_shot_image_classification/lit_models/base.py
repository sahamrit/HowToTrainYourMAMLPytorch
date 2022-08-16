import argparse
import pytorch_lightning as pl
import torch

from .optimizers.adam_explicit import AdamExplicitGrad
from .util import run_episodes

META_OPTIMIZER = "Adam"
META_LR = 1e-3
INNER_LOOP_LR = 5e-3
LOSS = "binary_cross_entropy_with_logits"
ONE_CYCLE_TOTAL_STEPS = 100
INNER_LOOP_STEPS = 5


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        meta_optimizer = self.args.get("optimizer", META_OPTIMIZER)
        self.meta_optimizer_class = getattr(torch.optim, meta_optimizer)
        self.inner_loop_optimizer_class = AdamExplicitGrad

        self.meta_lr = self.args.get("meta_lr", META_LR)
        self.inner_loop_lr = self.args.get("inner_loop_lr", INNER_LOOP_LR)
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_total_steps = self.args.get(
            "one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS
        )

        self.inner_loop_steps = self.args.get("inner_loop_steps", INNER_LOOP_STEPS)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--meta_optimizer",
            type=str,
            default=META_OPTIMIZER,
            help="optimizer class from torch.optim for meta model",
        )
        parser.add_argument("--meta_lr", type=float, default=META_LR)
        parser.add_argument("--inner_loop_lr", type=float, default=INNER_LOOP_LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument(
            "--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS
        )
        parser.add_argument("--inner_loop_steps", type=int, default=INNER_LOOP_STEPS)
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser

    def configure_optimizers(self):
        optimizer = self.meta_optimizer_class(self.parameters(), lr=self.meta_lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.one_cycle_max_lr,
            total_steps=self.one_cycle_total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (xs, ys), (xq, yq) = batch
        loss, correct_preds, total_preds = run_episodes(
            True,
            self.loss_fn,
            self.inner_loop_lr,
            self.inner_loop_optimizer_class,
            self.model,
            xs,
            xq,
            ys,
            yq,
            self.model.num_classes,
            self.inner_loop_steps,
        )

        self.log("train_loss", loss)
        self.log("train_acc", correct_preds / total_preds)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (xs, ys), (xq, yq) = batch
        loss, correct_preds, total_preds = run_episodes(
            False,
            self.loss_fn,
            self.inner_loop_lr,
            self.inner_loop_optimizer_class,
            self.model,
            xs,
            xq,
            ys,
            yq,
            self.model.num_classes,
            self.inner_loop_steps,
        )

        self.log("val_loss", loss)
        self.log("val_acc", correct_preds / total_preds)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        (xs, ys), (xq, yq) = batch
        loss, correct_preds, total_preds = run_episodes(
            False,
            self.loss_fn,
            self.inner_loop_lr,
            self.inner_loop_optimizer_class,
            self.model,
            xs,
            xq,
            ys,
            yq,
            self.model.num_classes,
            self.inner_loop_steps,
        )

        self.log("test_loss", loss)
        self.log("test_acc", correct_preds / total_preds)
