"""
Utilities for visualizing and interpreting the learned representations.

Includes functions for:
- Latent traversals (visualize what each latent dimension controls)
- Finding semantic interpretations (what words explain each dimension)
- Visualization of reconstructions
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
from model import InterpretableVAE


def latent_traversal(
    model: InterpretableVAE,
    image: torch.Tensor,
    latent_idx: int,
    values: np.ndarray = None,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Perform latent traversal: fix all latent dimensions except one,
    and vary that dimension across a range of values.
    
    Args:
        model: Trained InterpretableVAE model
        image: Input image [1, C, H, W] or [C, H, W]
        latent_idx: Index of latent dimension to traverse (0 to latent_dim-1)
        values: Array of values to traverse (default: -3 to 3 in 11 steps)
        device: Device to run on
    Returns:
        List of generated images
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Ensure image has batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    # Default traversal range: -3 to 3 (covers ~99.7% of standard normal)
    if values is None:
        values = np.linspace(-3, 3, 11)
    
    with torch.no_grad():
        # Get latent code for the image
        z = model.encode(image)  # [1, latent_dim]
        
        # Generate images by varying one latent dimension
        generated_images = []
        for value in values:
            z_modified = z.clone()
            z_modified[0, latent_idx] = value
            recon = model.decode(z_modified)
            generated_images.append(recon[0].cpu())
    
    return generated_images


def visualize_traversal(
    images: List[torch.Tensor],
    values: np.ndarray = None,
    title: str = "",
    figsize: Tuple[int, int] = (15, 2)
):
    """
    Visualize a latent traversal.
    
    Args:
        images: List of images from latent_traversal
        values: Values used for traversal (for labeling)
        title: Title for the plot
        figsize: Figure size
    """
    n = len(images)
    
    if values is None:
        values = np.linspace(-3, 3, n)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    for i, (img, val) in enumerate(zip(images, values)):
        # Convert to numpy and transpose to HWC
        img_np = img.permute(1, 2, 0).numpy()
        
        # Handle grayscale
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
            axes[i].imshow(img_np, cmap='gray')
        else:
            axes[i].imshow(np.clip(img_np, 0, 1))
        
        axes[i].axis('off')
        axes[i].set_title(f"{val:.1f}", fontsize=10)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    plt.tight_layout()
    return fig


def find_semantic_interpretation(
    model: InterpretableVAE,
    word_embeddings: torch.Tensor,
    words: List[str],
    latent_idx: int,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """
    Find words that best explain a latent dimension.
    
    Args:
        model: Trained InterpretableVAE model
        word_embeddings: Word embeddings [vocab_size, vse_dim]
        words: List of words corresponding to embeddings
        latent_idx: Index of latent dimension to interpret
        top_k: Number of top words to return
    Returns:
        List of (word, similarity, sign) tuples
    """
    model.eval()
    
    # Get basis vector for this latent dimension
    basis_vectors = model.semantic_sub_decoder.get_basis_vectors()
    basis = basis_vectors[latent_idx].unsqueeze(0)  # [1, vse_dim]
    
    # Compute cosine similarities with all words
    with torch.no_grad():
        similarities = F.cosine_similarity(
            basis.unsqueeze(1),  # [1, 1, vse_dim]
            word_embeddings.unsqueeze(0),  # [1, vocab_size, vse_dim]
            dim=2
        ).squeeze(0)  # [vocab_size]
    
    # Get top-k positive and negative
    top_pos_indices = similarities.topk(top_k).indices
    top_neg_indices = similarities.topk(top_k, largest=False).indices
    
    results = []
    
    # Positive words (increase this latent -> these concepts)
    for idx in top_pos_indices:
        results.append((words[idx], similarities[idx].item(), '+'))
    
    # Negative words (decrease this latent -> these concepts)
    for idx in top_neg_indices:
        results.append((words[idx], similarities[idx].item(), '-'))
    
    return results


def visualize_reconstructions(
    model: InterpretableVAE,
    images: torch.Tensor,
    n_images: int = 8,
    device: torch.device = None
):
    """
    Visualize original images and their reconstructions.
    
    Args:
        model: Trained InterpretableVAE model
        images: Batch of images [batch_size, C, H, W]
        n_images: Number of images to show
        device: Device to run on
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    images = images[:n_images].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        reconstructions = outputs['x_recon']
    
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
    
    for i in range(n_images):
        # Original
        img_orig = images[i].cpu().permute(1, 2, 0).numpy()
        if img_orig.shape[2] == 1:
            img_orig = img_orig.squeeze(2)
            axes[0, i].imshow(img_orig, cmap='gray')
        else:
            axes[0, i].imshow(np.clip(img_orig, 0, 1))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12, loc='left')
        
        # Reconstruction
        img_recon = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        if img_recon.shape[2] == 1:
            img_recon = img_recon.squeeze(2)
            axes[1, i].imshow(img_recon, cmap='gray')
        else:
            axes[1, i].imshow(np.clip(img_recon, 0, 1))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstruction", fontsize=12, loc='left')
    
    plt.tight_layout()
    return fig


def generate_interpretation_report(
    model: InterpretableVAE,
    word_embeddings: torch.Tensor,
    words: List[str],
    save_path: Optional[str] = None
):
    """
    Generate a full interpretation report for all latent dimensions.
    
    Args:
        model: Trained InterpretableVAE model
        word_embeddings: Word embeddings [vocab_size, vse_dim]
        words: List of words
        save_path: Optional path to save the report
    """
    print("="*80)
    print("LATENT DIMENSION INTERPRETATIONS")
    print("="*80)
    
    interpretations = model.get_interpretations(word_embeddings, words, top_k=4)
    
    report_lines = []
    
    for dim_idx in range(model.latent_dim):
        line = f"\nDimension {dim_idx}:"
        print(line)
        report_lines.append(line)
        
        # Positive words
        pos_words = [item for item in interpretations[dim_idx] if item[2] == '+']
        line = f"  Positive: {', '.join([f'{w}({s:.3f})' for w, s, _ in pos_words])}"
        print(line)
        report_lines.append(line)
        
        # Negative words
        neg_words = [item for item in interpretations[dim_idx] if item[2] == '-']
        line = f"  Negative: {', '.join([f'{w}({s:.3f})' for w, s, _ in neg_words])}"
        print(line)
        report_lines.append(line)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\nReport saved to {save_path}")


def interpolate_between_images(
    model: InterpretableVAE,
    image1: torch.Tensor,
    image2: torch.Tensor,
    n_steps: int = 10,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Interpolate between two images in latent space.
    
    Args:
        model: Trained InterpretableVAE model
        image1: First image [C, H, W]
        image2: Second image [C, H, W]
        n_steps: Number of interpolation steps
        device: Device to run on
    Returns:
        List of interpolated images
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Add batch dimension
    if image1.dim() == 3:
        image1 = image1.unsqueeze(0)
    if image2.dim() == 3:
        image2 = image2.unsqueeze(0)
    
    image1 = image1.to(device)
    image2 = image2.to(device)
    
    with torch.no_grad():
        # Encode both images
        z1 = model.encode(image1)
        z2 = model.encode(image2)
        
        # Interpolate
        alphas = np.linspace(0, 1, n_steps)
        interpolated_images = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img_interp = model.decode(z_interp)
            interpolated_images.append(img_interp[0].cpu())
    
    return interpolated_images


if __name__ == "__main__":
    print("Visualization utilities loaded.")
    print("\nAvailable functions:")
    print("  - latent_traversal: Vary one latent dimension")
    print("  - visualize_traversal: Plot traversal results")
    print("  - find_semantic_interpretation: Find words explaining a dimension")
    print("  - visualize_reconstructions: Compare originals and reconstructions")
    print("  - generate_interpretation_report: Full report for all dimensions")
    print("  - interpolate_between_images: Smooth interpolation in latent space")
