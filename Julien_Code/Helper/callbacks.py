from typing import Any
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.trainer.trainer import Trainer

class LRFinder(LearningRateFinder):
    def __init__(self, min_lr: float = 1e-8, max_lr: float = 1, num_training_steps: int = 100, mode: str = 'exponential', early_stop_threshold: float = 4.0, update_attr: bool = False):
        super().__init__(min_lr, max_lr, num_training_steps, mode, early_stop_threshold, update_attr)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Run the learning rate finder
        self.lr_find(trainer, pl_module)
        # Prevent the trainer from starting the training
        # The new LR is set on the model, and you can see it in the logs.
        # You can then comment out the LRFinder callback and re-run to train.
        trainer.should_stop = True 