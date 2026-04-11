# Interpretable Representation Learning via Visual-Semantic Embedding

PyTorch implementation of the model from the paper:

**"Interpretable Representation Learning on Natural Image Datasets via Reconstruction in Visual-Semantic Embedding Space"**  
*Nao Nakagawa, Ren Togo, Takahiro Ogawa, and Miki Haseyama*  
ICIP 2021

## Overview

This implementation provides a VAE-based model that learns **disentangled and interpretable** representations. The key innovation is the **semantic sub-decoder** that maps each latent dimension to a word vector in a Visual-Semantic Embedding (VSE) space, allowing you to understand what each latent variable represents in natural language.

### Key Features

- ✅ Standard VAE architecture (encoder + decoder)
- ✅ Semantic sub-decoder for interpretable representations
- ✅ Latent traversal visualization
- ✅ Semantic interpretation via VSE space
- ✅ Support for CelebA, CIFAR-10, MNIST datasets
- ✅ Training scripts with proper hyperparameters from the paper

## Architecture

```
Input Image (64×64) 
    ↓
Encoder (Conv layers)
    ↓
μ, log σ² → z (latent code, dim=32)
    ↓
    ├─→ Decoder → Reconstructed Image
    │
    └─→ Semantic Sub-Decoder (Linear) → VSE Vector
                                          ↓
                                    Compare with E_VSE(x)
```

**Loss Function:**
```
L = L_AE + β·L_REG + γ·L_VSE + Σ(A^T A)_{i≠j}
```

Where:
- `L_AE`: Reconstruction loss (MSE)
- `L_REG`: KL divergence (regularization)
- `L_VSE`: Semantic reconstruction loss
- Last term: Orthogonal regularization for basis vectors

## Installation

### Requirements

```bash
Python >= 3.7
PyTorch >= 1.8.0
torchvision >= 0.9.0
numpy
matplotlib
tqdm
```

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## File Structure

```
.
├── model.py          # Core model implementation
├── train.py          # Training script
├── utils.py          # Visualization and interpretation utilities
├── example.py        # Example usage
└── README.md         # This file
```

## Quick Start

### 1. Basic Training (without VSE)

```python
from model import InterpretableVAE
from train import train, get_dataloaders

# Create model
model = InterpretableVAE(
    latent_dim=32,
    vse_dim=1024,
    image_channels=3,
    vse_encoder=None  # No VSE for now
)

# Load data
train_loader, val_loader = get_dataloaders(
    dataset_name='cifar10',
    batch_size=64,
    image_size=64
)

# Train
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    beta=10.0,  # Strong disentanglement
    gamma=0.0,  # No semantic loss (no VSE)
    save_dir='./checkpoints'
)
```

### 2. Visualize Learned Representations

```python
from utils import latent_traversal, visualize_traversal
import torch

# Load your trained model
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Get a test image
test_image = val_loader.dataset[0][0]  # First validation image

# Traverse latent dimension 0
images = latent_traversal(model, test_image, latent_idx=0)

# Visualize
fig = visualize_traversal(
    images,
    title="Latent Dimension 0",
    values=np.linspace(-3, 3, 11)
)
plt.show()
```

### 3. Semantic Interpretations (with VSE)

To use semantic interpretations, you need a pre-trained VSE encoder. Here's how it would work:

```python
# Load pre-trained VSE encoder (you need to provide this)
vse_encoder = load_your_vse_encoder()

# Create model with VSE
model = InterpretableVAE(
    latent_dim=32,
    vse_dim=1024,
    vse_encoder=vse_encoder  # Now with VSE
)

# Train with semantic loss
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    beta=1.0,
    gamma=10.0,  # Enable semantic loss
    ortho_weight=1.0,
    save_dir='./checkpoints'
)

# Get interpretations
word_embeddings, words = load_vse_vocabulary()
interpretations = model.get_interpretations(word_embeddings, words)

# Print what dimension 5 represents
print(f"Dimension 5: {interpretations[5]}")
# Example: [('smiling', 0.85, '+'), ('happy', 0.78, '+'), ...]
```

## Hyperparameters

As specified in the paper:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 32 | Number of latent dimensions |
| `image_size` | 64×64 | Input image size |
| `batch_size` | 64 | Batch size |
| `learning_rate` | 1e-4 | Learning rate for Adam |
| `β` | 1 or 10 | KL weight (10 for stronger disentanglement) |
| `γ` | 10 | Semantic loss weight |
| `ortho_weight` | 1 | Orthogonal regularization weight |

## Datasets

### Supported Datasets

1. **CIFAR-10** (automatic download)
2. **MNIST** (automatic download, use `image_channels=1`)
3. **CelebA** (manual download required)

### Using CelebA

1. Download CelebA from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Place in `./data/celeba/`
3. Load with:

```python
train_loader, val_loader = get_dataloaders(
    dataset_name='celeba',
    data_path='./data',
    batch_size=64
)
```

## Training Tips

1. **Start with CIFAR-10 or MNIST** for faster experimentation
2. **Without VSE**: Set `gamma=0` and `ortho_weight=0`
3. **For stronger disentanglement**: Increase `beta` to 10
4. **Monitor losses**: Check that KL loss doesn't collapse to 0
5. **Latent traversals**: Use to verify that dimensions are disentangled

## VSE Integration

The paper uses a pre-trained VSE encoder from:

> Kiros et al. "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models" (2014)

To fully replicate the paper:

1. Train or download a VSE model that embeds images and text in the same space
2. The VSE encoder should output 1024-dimensional vectors
3. Freeze the VSE encoder during training
4. Provide word embeddings from the same VSE space for interpretation

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{nakagawa2021interpretable,
  title={Interpretable Representation Learning on Natural Image Datasets via Reconstruction in Visual-Semantic Embedding Space},
  author={Nakagawa, Nao and Togo, Ren and Ogawa, Takahiro and Haseyama, Miki},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2473--2477},
  year={2021},
  organization={IEEE}
}
```

## Example Results

After training, you should observe:

- **Disentangled representations**: Each latent dimension controls a specific factor
- **Semantic interpretations**: Words like "smiling", "male", "glasses" explain dimensions
- **Smooth traversals**: Varying one dimension smoothly changes one attribute
- **Good reconstructions**: Images are accurately reconstructed

## Troubleshooting

### Common Issues

**Q: KL loss is 0**  
A: This is posterior collapse. Try:
- Reducing `beta` initially, then gradually increase
- Using a lower learning rate
- Implementing KL annealing

**Q: Reconstructions are blurry**  
A: Normal for MSE loss. You can:
- Use perceptual loss instead
- Reduce `beta` to prioritize reconstruction

**Q: Dimensions are not disentangled**  
A: Try:
- Increasing `beta` to 10 or higher
- Training for more epochs
- Using a simpler dataset first (MNIST, dSprites)

**Q: Where do I get VSE embeddings?**  
A: You need to:
- Train your own VSE model (see Kiros et al. 2014)
- Or find pre-trained VSE models online
- Or use sentence-transformers as an approximation

## License

This implementation is for research and educational purposes. Please refer to the original paper for the method.

## Acknowledgments

Based on the paper by Nakagawa et al. (ICIP 2021). This is an unofficial implementation created for educational purposes.
