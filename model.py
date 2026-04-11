"""
Interpretable Representation Learning via Visual-Semantic Embedding (VSE)
Based on: Nakagawa et al. "Interpretable Representation Learning on Natural Image 
Datasets via Reconstruction in Visual-Semantic Embedding Space" ICIP 2021

Implementation of the core model components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class Encoder(nn.Module):
    """
    Standard VAE encoder that outputs mean and log-variance.
    Maps images to latent distribution parameters.
    """
    
    def __init__(self, latent_dim: int = 32, image_channels: int = 3):
        super().__init__()
        
        # Convolutional layers (for 64x64 images as in the paper)
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size: 256 * 4 * 4 = 4096
        self.flatten_size = 256 * 4 * 4
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Standard VAE decoder that reconstructs images from latent codes.
    """
    
    def __init__(self, latent_dim: int = 32, image_channels: int = 3):
        super().__init__()
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Transposed convolutional layers
        self.deconv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes [batch_size, latent_dim]
        Returns:
            x_recon: Reconstructed images [batch_size, channels, height, width]
        """
        h = self.fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        x_recon = self.deconv_layers(h)
        return x_recon


class SemanticSubDecoder(nn.Module):
    """
    Linear semantic sub-decoder that maps latent codes to VSE space.
    
    Key innovation: Each column of the weight matrix A corresponds to a word vector 
    that explains one latent dimension. This allows semantic interpretation of what 
    each latent variable represents.
    """
    
    def __init__(self, latent_dim: int = 32, vse_dim: int = 1024):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.vse_dim = vse_dim
        
        # Linear transformation: A in R^{V x N}
        # Each column a_i is the word vector for latent dimension z_i
        self.A = nn.Linear(latent_dim, vse_dim, bias=True)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct VSE vector from latent code.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
        Returns:
            w_hat: Semantic reconstruction in VSE space [batch_size, vse_dim]
        """
        # w_hat = Az + b = sum_i(a_i * z_i) + b
        w_hat = self.A(z)
        return w_hat
    
    def get_basis_vectors(self) -> torch.Tensor:
        """
        Returns the basis vectors (columns of A) that represent word embeddings
        for each latent dimension.
        
        Returns:
            basis_vectors: [latent_dim, vse_dim] where each row is a word vector
        """
        # Transpose weight matrix to get [latent_dim, vse_dim]
        return self.A.weight.T.detach()


class InterpretableVAE(nn.Module):
    """
    Complete VAE model with semantic sub-decoder for interpretable representations.
    
    This model learns disentangled representations where each latent dimension
    can be explained by word vectors from a Visual-Semantic Embedding space.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        vse_dim: int = 1024,
        image_channels: int = 3,
        vse_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.vse_dim = vse_dim
        
        # Main VAE components
        self.encoder = Encoder(latent_dim, image_channels)
        self.decoder = Decoder(latent_dim, image_channels)
        
        # Semantic components (the key innovation)
        self.semantic_sub_decoder = SemanticSubDecoder(latent_dim, vse_dim)
        
        # Pre-trained VSE encoder (frozen during training)
        self.vse_encoder = vse_encoder
        if self.vse_encoder is not None:
            for param in self.vse_encoder.parameters():
                param.requires_grad = False
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log variance [batch_size, latent_dim]
        Returns:
            z: Sampled latent codes [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        Returns:
            Dictionary containing:
                - x_recon: Reconstructed images
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent codes
                - w: VSE vectors (if vse_encoder is provided)
                - w_hat: Semantic reconstruction
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample latent codes using reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Decode to image
        x_recon = self.decoder(z)
        
        # Semantic reconstruction (map latent to VSE space)
        w_hat = self.semantic_sub_decoder(z)
        
        outputs = {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'w_hat': w_hat
        }
        
        # Get VSE vectors if encoder is available
        if self.vse_encoder is not None:
            with torch.no_grad():
                w = self.vse_encoder(x)
            outputs['w'] = w
        
        return outputs
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent codes (using mean, no sampling)"""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to images"""
        return self.decoder(z)
    
    def get_interpretations(
        self,
        word_embeddings: torch.Tensor,
        words: list,
        top_k: int = 5
    ) -> Dict[int, list]:
        """
        Get word interpretations for each latent dimension by finding
        the words most similar to each basis vector.
        
        Args:
            word_embeddings: Pre-computed word embeddings [vocab_size, vse_dim]
            words: List of words corresponding to embeddings
            top_k: Number of top words to return for each dimension
        Returns:
            Dictionary mapping latent dimension index to list of (word, similarity, sign) tuples
        """
        basis_vectors = self.semantic_sub_decoder.get_basis_vectors()
        
        interpretations = {}
        
        for i in range(self.latent_dim):
            # Get basis vector for dimension i
            basis = basis_vectors[i].unsqueeze(0)  # [1, vse_dim]
            
            # Compute cosine similarities with all words
            similarities = F.cosine_similarity(
                basis.unsqueeze(1),  # [1, 1, vse_dim]
                word_embeddings.unsqueeze(0),  # [1, vocab_size, vse_dim]
                dim=2
            ).squeeze(0)  # [vocab_size]
            
            # Get top-k positive and negative
            top_pos_indices = similarities.topk(top_k).indices
            top_neg_indices = similarities.topk(top_k, largest=False).indices
            
            top_words = []
            for idx in top_pos_indices:
                top_words.append((words[idx], similarities[idx].item(), '+'))
            for idx in top_neg_indices:
                top_words.append((words[idx], similarities[idx].item(), '-'))
            
            interpretations[i] = top_words
        
        return interpretations


def compute_losses(
    model_outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 10.0,
    use_semantic_loss: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute all losses as defined in the paper.
    
    Total Loss = L_AE + beta * L_REG + gamma * L_VSE + orthogonal_reg
    
    Args:
        model_outputs: Outputs from forward pass
        x: Original images
        beta: Weight for KL divergence (> 1 encourages disentanglement)
        gamma: Weight for semantic loss (> 0)
        use_semantic_loss: Whether to use semantic reconstruction loss
    Returns:
        Dictionary of losses
    """
    x_recon = model_outputs['x_recon']
    mu = model_outputs['mu']
    logvar = model_outputs['logvar']
    w_hat = model_outputs['w_hat']
    
    # Reconstruction loss L_AE (negative log likelihood)
    # Using MSE (can also use BCE for binary images)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    
    # Regularization loss L_REG (KL divergence)
    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    losses = {
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
    }
    
    # Semantic loss L_VSE (if VSE vectors are available)
    if use_semantic_loss and 'w' in model_outputs:
        w = model_outputs['w']
        # MSE between VSE reconstruction and ground truth
        semantic_loss = F.mse_loss(w_hat, w, reduction='sum') / x.size(0)
        losses['semantic_loss'] = semantic_loss
    else:
        semantic_loss = torch.tensor(0.0, device=x.device)
        losses['semantic_loss'] = semantic_loss
    
    # Total loss (orthogonal regularization added separately)
    total_loss = recon_loss + beta * kl_loss + gamma * semantic_loss
    losses['total_loss'] = total_loss
    
    return losses


def compute_orthogonal_regularization(model: InterpretableVAE) -> torch.Tensor:
    """
    Compute orthogonal regularization for basis vectors.
    
    This encourages different latent dimensions to capture different semantic concepts
    by penalizing correlations between the basis vectors.
    
    Regularization = sum_{i != j} (a_i^T a_j)
    
    Args:
        model: The InterpretableVAE model
    Returns:
        Orthogonal regularization term
    """
    # Get weight matrix A [vse_dim, latent_dim]
    A = model.semantic_sub_decoder.A.weight
    
    # Compute A^T A [latent_dim, latent_dim]
    # This gives us the dot products between all pairs of basis vectors
    gram = torch.mm(A.T, A)
    
    # Sum of off-diagonal elements (correlations between different dimensions)
    mask = 1 - torch.eye(gram.size(0), device=gram.device)
    ortho_reg = torch.sum(torch.abs(gram * mask))
    
    return ortho_reg
