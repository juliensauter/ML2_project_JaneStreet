print("Importing Modules...")

# Import timing module first
from Helper.timing import timer, timeit

with timer("Module Imports"):
    from sys import exit
    import os
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader

    from Helper.data_processing import Data
    from Helper.dataset import TransformedDataset, AddFreq
    from Helper.callbacks import LRFinder

    from Model.finetune import Finetuner

    from uni2ts.model.moirai import MoiraiModule
    from lightning.pytorch import Trainer, callbacks, loggers
    from lightning.pytorch.callbacks import TQDMProgressBar
    from uni2ts.data.loader import PackCollate

print("Importing Complete.")
print()

@timeit("Data Loading and Preprocessing")
def load_and_preprocess_data(data_path, patch_size, min_patches):
    """Load and preprocess the Jane Street dataset."""
    return Data(data_path, patch_size=patch_size, min_patches=min_patches)

@timeit("Model Loading")
def load_model(model_path, min_patches):
    """Load the pre-trained Moirai model."""
    return Finetuner(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_path}"),
        min_patches=min_patches
    )

@timeit("Dataset Transformation")
def create_transformed_datasets(data: Data, finetuner: Finetuner, patch_size):
    """Create training and validation datasets with transformations."""
    train_transform = finetuner.train_transform_map[type(data.train_dataset)](patch_size=patch_size)
    add_freq_transform = AddFreq(freq="s")
    
    train_dataset = TransformedDataset(data.train_dataset, add_freq_transform + train_transform)
    val_dataset = TransformedDataset(data.val_dataset, add_freq_transform + train_transform)
    
    return train_dataset, val_dataset

@timeit("DataLoader Creation")
def create_dataloaders(train_dataset, val_dataset, finetuner, batch_size, num_workers, persistent_workers):
    """Create training and validation data loaders."""
    collate_fn = PackCollate(max_length=2048, seq_fields=finetuner.seq_fields)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    
    return train_loader, val_loader

@timeit("Logger Setup")
def setup_logger():
    """Setup TensorBoard logger."""
    return loggers.TensorBoardLogger(
        save_dir="logs",
        name="moirai_run",
    )

@timeit("Callbacks Setup")
def setup_callbacks():
    """Setup all training callbacks."""
    ckpt_cb = callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch:02d}-{val/PackedNLLLoss:.4f}",
        every_n_epochs=1,
        save_last=True,
        save_top_k=-1,
        monitor="val/PackedNLLLoss",
        mode="min",
    )
    
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val/PackedNLLLoss",
        patience=10,
        mode="min",
        verbose=True,
    )
    
    lr_cb = callbacks.LearningRateMonitor(logging_interval="epoch")
    
    bar_cb = TQDMProgressBar(refresh_rate=1, leave=True)
    
    # lr_finder_cb = LRFinder(min_lr=1e-5, max_lr=1e-2)
    
    return [ckpt_cb, lr_cb, bar_cb, early_stop_cb]

@timeit("Trainer Setup")
def setup_trainer(logger, callbacks_list):
    """Setup the PyTorch Lightning trainer."""
    return Trainer(
        max_epochs=200,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks_list,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
    )

@timeit("Training Execution")
def run_training(trainer, finetuner, train_loader, val_loader):
    """Execute the training process."""
    print("🚀 Starting model training...")
    trainer.fit(finetuner, train_dataloaders=train_loader, val_dataloaders=val_loader)

def print_system_info(num_workers):
    """Print system configuration information."""
    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Number of CPUs: {num_workers}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    print()

def main(
    data_path,
    model_path,
    prediction_length,
    context_length,
    patch_size,
    num_samples,
    target_dim,
    batch_size,
    num_workers,
    persistent_workers,
    min_patches,
    from_disk=False
):
    """Main training pipeline with comprehensive timing."""
    
    # Print system information
    print_system_info(num_workers)
    
    # Set PyTorch precision
    torch.set_float32_matmul_precision('medium')
    
    # Step 1: Data Loading and Preprocessing
    with timer("Complete Data Pipeline"):
        data = load_and_preprocess_data(data_path, patch_size, min_patches)
    
    # Step 2: Model Loading
    finetuner = load_model(model_path, min_patches)
    
    # Step 3: Dataset Creation
    train_dataset, val_dataset = create_transformed_datasets(data, finetuner, patch_size)
    
    # Step 4: DataLoader Creation
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, finetuner, 
        batch_size, num_workers, persistent_workers
    )
    
    # Step 5: Logger Setup
    logger = setup_logger()
    
    # Step 6: Callbacks Setup
    callbacks_list = setup_callbacks()
    
    # Step 7: Trainer Setup
    trainer = setup_trainer(logger, callbacks_list)
    
    # Step 8: Training Execution
    run_training(trainer, finetuner, train_loader, val_loader)

def iter_transformed(base_ds, transformation):
    for entry in base_ds:
        yield transformation(entry)

if __name__ == "__main__":
    
    # Configuration
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "../data/jane_street_train.parquet"
    
    # Model configuration
    model_path = "large"
    
    context_length = 1536
    prediction_length = 256
    
    patch_size = 64 # {"auto", 8, 16, 32, 64, 128}
    num_samples = 50
    target_dim = 1
    batch_size = 8
    # System configuration
    
    num_workers = (
        None
        # os.cpu_count()
        or 0
    )
    persistent_workers = True if num_workers else False
    
    min_patches = 2
    
    # Execute main pipeline
    main(
        data_path=data_path,
        model_path=model_path,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=target_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        min_patches=min_patches,
    )
