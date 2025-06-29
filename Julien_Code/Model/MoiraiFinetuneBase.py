from lightning.pytorch import LightningModule
from torch.optim import AdamW

from uni2ts.model.moirai import MoiraiModule
from uni2ts.optim import SchedulerType, get_scheduler


class MoiraiFinetuneBase(LightningModule):
    """
    A corrected and properly typed base class for Moirai fine-tuning.
    This class correctly handles hyperparameters and optimizer setup for modern
    PyTorch Lightning versions.
    """

    def __init__(
        self,
        module: MoiraiModule,
        min_patches: int = 2,
        min_mask_ratio: float = 0.2,
        max_mask_ratio: float = 0.4,
        max_dim: int = 128,
        num_training_steps: int = 20000,
        num_warmup_steps: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        num_samples: int = 100,
    ):
        super().__init__()
        # Use save_hyperparameters() to correctly store all arguments
        self.save_hyperparameters(ignore=["module"])
        self.module = module

    def forward(self, *args, **kwargs):
        # The forward call is delegated to the specific Finetuner implementation
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        # The training_step is delegated to the specific Finetuner implementation
        raise NotImplementedError

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler.
        """
        # Correctly access hyperparameters via self.hparams
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        } 