"""
Example script demonstrating how to use the Interpretable VAE.

This script shows:
1. How to create and train the model
2. How to perform latent traversals
3. How to get semantic interpretations
4. How to visualize results

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import InterpretableVAE
from train import train, get_dataloaders
from utils import (
    latent_traversal,
    visualize_traversal,
    visualize_reconstructions,
    find_semantic_interpretation
)


def simple_example_without_vse():
    """
    Example 1: Train the model without a VSE encoder.
    
    This still learns disentangled representations, but without
    the semantic interpretation capability. Useful for testing.
    """
    print("="*80)
    print("EXAMPLE 1: Training without VSE encoder")
    print("="*80)
    
    # Model hyperparameters
    latent_dim = 32
    image_channels = 3  # RGB images
    batch_size = 128
    
    # Create model (no VSE encoder)
    model = InterpretableVAE(
        latent_dim=latent_dim,
        vse_dim=1024,  # Not used without VSE encoder
        image_channels=image_channels,
        vse_encoder=None
    )
    
    print(f"Model created with {latent_dim} latent dimensions")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader = get_dataloaders(
        dataset_name='cifar10',
        batch_size=batch_size,
        image_size=64
    )
    
    # Train for a few epochs (for demo purposes)
    print("\nTraining for 5 epochs...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-3,
        beta=10.0,  # Strong disentanglement
        gamma=0.0,  # No semantic loss (no VSE)
        ortho_weight=0.0,  # No orthogonal regularization (no VSE)
        save_dir='./checkpoints_example1'
    )
    
    return model, val_loader


def example_with_visualizations(model, val_loader):
    """
    Example 2: Visualize what the model has learned.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Visualizing learned representations")
    print("="*80)
    
    device = next(model.parameters()).device
    
    # Get a batch of validation images
    images, _ = next(iter(val_loader))
    images = images[:8]
    
    # 1. Visualize reconstructions
    print("\n1. Visualizing reconstructions...")
    fig = visualize_reconstructions(model, images, n_images=8)
    plt.savefig('reconstructions.png', dpi=150, bbox_inches='tight')
    print("   Saved to: reconstructions.png")
    plt.close()
    
    # 2. Perform latent traversals for first 5 dimensions
    print("\n2. Performing latent traversals...")
    test_image = images[0]
    
    for dim_idx in range(5):
        print(f"   Traversing dimension {dim_idx}...")
        traversal_images = latent_traversal(
            model, test_image, dim_idx, 
            values=np.linspace(-3, 3, 11)
        )
        
        fig = visualize_traversal(
            traversal_images,
            values=np.linspace(-3, 3, 11),
            title=f"Latent Dimension {dim_idx}",
            figsize=(18, 2)
        )
        plt.savefig(f'traversal_dim_{dim_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print("   Saved traversals to: traversal_dim_*.png")


def example_with_vse():
    """
    Example 3: Using the model with a VSE encoder for semantic interpretations.
    
    Note: This is a placeholder. You'll need to:
    1. Download or train a VSE encoder (e.g., from Kiros et al. 2014)
    2. Load the word embeddings
    3. Integrate it with the model
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: With VSE encoder (placeholder)")
    print("="*80)
    
    print("\nTo use semantic interpretations, you need:")
    print("1. A pre-trained VSE encoder (e.g., from the Unifying VSE paper)")
    print("2. Word embeddings in the same VSE space")
    print("\nExample integration:")
    print("""
    # Load your pre-trained VSE encoder
    vse_encoder = load_pretrained_vse_encoder()
    
    # Create model with VSE
    model = InterpretableVAE(
        latent_dim=32,
        vse_dim=1024,  # Match your VSE dimension
        vse_encoder=vse_encoder
    )
    
    # Train with semantic loss
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        beta=1.0,
        gamma=10.0,  # Enable semantic loss
        ortho_weight=1.0  # Enable orthogonal regularization
    )
    
    # Load word embeddings
    word_embeddings, words = load_vse_vocabulary()
    
    # Get interpretations
    interpretations = model.get_interpretations(
        word_embeddings, words, top_k=5
    )
    
    # Print what dimension 0 represents
    print(f"Dimension 0: {interpretations[0]}")
    # Example output: 
    # [('smiling', 0.85, '+'), ('happy', 0.78, '+'), ...]
    """)


def create_dummy_vse_embeddings(vocab_size=1000, vse_dim=1024):
    """
    Create dummy word embeddings for testing the interpretation function.
    
    In practice, you would load real word embeddings from a trained VSE model.
    """
    words = [
        # Facial attributes (for CelebA)
        'smiling', 'young', 'male', 'female', 'eyeglasses', 'bald', 'beard',
        'mustache', 'attractive', 'pale', 'oval', 'chubby', 'wavy', 'receding',
        'bangs', 'sideburns', 'black', 'blond', 'brown', 'gray',
        # General attributes
        'dark', 'light', 'bright', 'dim', 'high', 'low', 'wide', 'narrow',
        'large', 'small', 'thick', 'thin', 'heavy', 'light', 'strong', 'weak',
        # Colors
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'white',
        # Add random words
    ] + [f'word_{i}' for i in range(vocab_size - 50)]
    
    # Random embeddings (not meaningful, just for demonstration)
    embeddings = torch.randn(vocab_size, vse_dim)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize
    
    return embeddings, words


def example_with_dummy_interpretations(model):
    """
    Example showing how interpretations work with dummy embeddings.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Dummy semantic interpretations")
    print("="*80)
    
    print("\nCreating dummy word embeddings...")
    word_embeddings, words = create_dummy_vse_embeddings()
    
    print("\nFinding interpretations for first 5 dimensions:")
    for dim_idx in range(5):
        interp = find_semantic_interpretation(
            model, word_embeddings, words, dim_idx, top_k=3
        )
        
        pos_words = [f"{w}({s:.3f})" for w, s, sign in interp if sign == '+']
        neg_words = [f"{w}({s:.3f})" for w, s, sign in interp if sign == '-']
        
        print(f"\nDimension {dim_idx}:")
        print(f"  Positive: {', '.join(pos_words)}")
        print(f"  Negative: {', '.join(neg_words)}")
    
    print("\nNote: These are random embeddings! Real VSE embeddings would")
    print("give meaningful semantic interpretations.")


if __name__ == "__main__":
    import torch.nn.functional as F
    
    print("Interpretable VAE - Example Usage")
    print("="*80)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run examples
    try:
        # Example 1: Train without VSE
        model, val_loader = simple_example_without_vse()
        
        # Example 2: Visualizations
        example_with_visualizations(model, val_loader)
        
        # Example 3: VSE integration (instructions only)
        example_with_vse()
        
        # Example 4: Dummy interpretations
        example_with_dummy_interpretations(model)
        
        print("\n" + "="*80)
        print("Examples completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
