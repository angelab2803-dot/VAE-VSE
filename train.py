"""
Training script for Interpretable VAE with VSE

This script provides functions to train the model on image datasets
like CelebA or Stanford Cars.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import (
    InterpretableVAE,
    compute_losses,
    compute_orthogonal_regularization
)
from tqdm import tqdm
import os
from typing import Optional


def train_epoch(
    model: InterpretableVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
    gamma: float = 10.0,
    ortho_weight: float = 1.0
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: InterpretableVAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence (paper uses 1 or 10)
        gamma: Weight for semantic loss (paper uses 10)
        ortho_weight: Weight for orthogonal regularization
    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    
    total_losses = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'semantic_loss': 0.0,
        'ortho_reg': 0.0
    }
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        data = data.to(device)
        
        # Forward pass
        outputs = model(data)
        
        # Compute losses
        losses = compute_losses(
            outputs,
            data,
            beta=beta,
            gamma=gamma,
            use_semantic_loss=(model.vse_encoder is not None)
        )
        
        # Add orthogonal regularization
        ortho_reg = compute_orthogonal_regularization(model)
        total_loss = losses['total_loss'] + ortho_weight * ortho_reg
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_losses['total_loss'] += total_loss.item()
        total_losses['recon_loss'] += losses['recon_loss'].item()
        total_losses['kl_loss'] += losses['kl_loss'].item()
        total_losses['semantic_loss'] += losses['semantic_loss'].item()
        total_losses['ortho_reg'] += ortho_reg.item()
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def validate(
    model: InterpretableVAE,
    dataloader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
    gamma: float = 10.0
) -> dict:
    """
    Validate the model.
    
    Args:
        model: InterpretableVAE model
        dataloader: Validation data loader
        device: Device to validate on
        beta: Weight for KL divergence
        gamma: Weight for semantic loss
    Returns:
        Dictionary of average validation losses
    """
    model.eval()
    
    total_losses = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'semantic_loss': 0.0
    }
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Compute losses
            losses = compute_losses(
                outputs,
                data,
                beta=beta,
                gamma=gamma,
                use_semantic_loss=(model.vse_encoder is not None)
            )
            
            # Accumulate losses
            total_losses['total_loss'] += losses['total_loss'].item()
            total_losses['recon_loss'] += losses['recon_loss'].item()
            total_losses['kl_loss'] += losses['kl_loss'].item()
            total_losses['semantic_loss'] += losses['semantic_loss'].item()
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def train(
    model: InterpretableVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    beta: float = 1.0,
    gamma: float = 10.0,
    ortho_weight: float = 1.0,
    device: Optional[torch.device] = None,
    save_dir: str = './checkpoints'
):
    """
    Complete training loop.
    
    Args:
        model: InterpretableVAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        beta: Weight for KL divergence (1 for standard VAE, 10 for strong disentanglement)
        gamma: Weight for semantic loss (paper uses 10)
        ortho_weight: Weight for orthogonal regularization
        device: Device to train on (auto-detected if None)
        save_dir: Directory to save checkpoints
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            beta=beta, gamma=gamma, ortho_weight=ortho_weight
        )
        
        print(f"\nTrain Losses:")
        print(f"  Total:    {train_losses['total_loss']:.4f}")
        print(f"  Recon:    {train_losses['recon_loss']:.4f}")
        print(f"  KL:       {train_losses['kl_loss']:.4f}")
        print(f"  Semantic: {train_losses['semantic_loss']:.4f}")
        print(f"  Ortho:    {train_losses['ortho_reg']:.4f}")
        
        # Validation
        val_losses = validate(model, val_loader, device, beta=beta, gamma=gamma)
        
        print(f"\nValidation Losses:")
        print(f"  Total:    {val_losses['total_loss']:.4f}")
        print(f"  Recon:    {val_losses['recon_loss']:.4f}")
        print(f"  KL:       {val_losses['kl_loss']:.4f}")
        print(f"  Semantic: {val_losses['semantic_loss']:.4f}")
        
        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'hyperparameters': {
                    'beta': beta,
                    'gamma': gamma,
                    'ortho_weight': ortho_weight,
                    'learning_rate': learning_rate
                }
            }, checkpoint_path)
            print(f"\n✓ Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint to {checkpoint_path}")


def get_dataloaders(
    dataset_name: str = 'celeba',
    data_path: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 64
):
    """
    Create dataloaders for training.
    
    Args:
        dataset_name: 'celeba', 'mnist', or 'cifar10'
        data_path: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Image size (64 in the paper)
    Returns:
        train_loader, val_loader
    """
    
    # Standard transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    if dataset_name.lower() == 'mnist':
        # MNIST for quick testing (grayscale, so use image_channels=1 in model)
        train_dataset = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )
        val_dataset = datasets.MNIST(
            data_path, train=False, download=True, transform=transform
        )
    
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 for testing
        train_dataset = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )
        val_dataset = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform
        )
    
    elif dataset_name.lower() == 'celeba':
        # CelebA dataset (requires manual download)
        # Download from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
        try:
            train_dataset = datasets.CelebA(
                data_path, split='train', download=False, transform=transform
            )
            val_dataset = datasets.CelebA(
                data_path, split='valid', download=False, transform=transform
            )
        except:
            print("CelebA dataset not found. Please download manually.")
            print("Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            raise
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    
    # Hyperparameters from the paper
    LATENT_DIM = 32
    VSE_DIM = 1024  # Dimension of VSE space (depends on your VSE model)
    IMAGE_CHANNELS = 3  # 3 for RGB, 1 for grayscale
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    BETA = 1.0  # Try 10 for stronger disentanglement
    GAMMA = 10.0  # Weight for semantic loss
    
    # Create model (without VSE encoder for now)
    model = InterpretableVAE(
        latent_dim=LATENT_DIM,
        vse_dim=VSE_DIM,
        image_channels=IMAGE_CHANNELS,
        vse_encoder=None  # TODO: Add your pre-trained VSE encoder here
    )
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_name='cifar10',  # Start with CIFAR-10 for testing
        batch_size=BATCH_SIZE
    )
    
    # Train
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        gamma=GAMMA,
        save_dir='./checkpoints'
    )
