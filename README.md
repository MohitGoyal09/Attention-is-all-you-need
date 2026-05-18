# Transformer: Attention Is All You Need

A PyTorch implementation of the Transformer architecture from the seminal paper "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" (Vaswani et al., 2017). This project provides a complete implementation of the Transformer model for neural machine translation, featuring all core components including multi-head attention, positional encoding, and feed-forward networks.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Model Components](#model-components)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Features

✨ **Complete Transformer Implementation**
- Full encoder-decoder architecture with self-attention mechanisms
- Multi-head attention for parallel processing of input features
- Positional encoding using sinusoidal position representations
- Feed-forward networks with ReLU activation
- Residual connections with layer normalization
- Support for custom model hyperparameters

📊 **Comprehensive Training Pipeline**
- Bilingual dataset support using Hugging Face datasets
- Automatic tokenization with BPE-style word-level tokenization
- Validation during training with real-time evaluation metrics
- Support for model checkpointing and resuming training
- TensorBoard integration for monitoring training progress

📈 **Evaluation & Metrics**
- Character Error Rate (CER) computation
- Word Error Rate (WER) computation
- BLEU score evaluation
- Greedy decoding for inference

🚀 **Production-Ready Features**
- Multi-GPU support (CUDA, MPS backend support)
- Model serialization and resumption
- Flexible configuration system
- Comprehensive error handling

## Project Structure

```
Attention-is-all-you-need/
├── model.py              # Core Transformer architecture and components
├── train.py              # Training pipeline and evaluation functions
├── dataset.py            # Dataset handling and preprocessing
├── config.py             # Configuration management
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Architecture Overview

### Transformer Components

The implementation includes all key components of the Transformer architecture:

#### 1. **Input Embedding**
Converts token indices to dense vectors with dimension scaling
```
Token Index → Embedding Vector × √d_model
```

#### 2. **Positional Encoding**
Adds positional information using sinusoidal functions
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 3. **Multi-Head Attention**
Parallel attention mechanisms allowing the model to attend to information from different representation subspaces
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### 4. **Feed-Forward Network**
Position-wise fully connected feed-forward network
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

#### 5. **Layer Normalization**
Normalizes input with learnable parameters
```
LayerNorm(x) = α × (x - μ) / √(σ² + ε) + β
```

#### 6. **Encoder**
Stacks multiple encoder blocks with self-attention and feed-forward sub-layers

#### 7. **Decoder**
Stacks multiple decoder blocks with self-attention, cross-attention, and feed-forward sub-layers

#### 8. **Projection Layer**
Projects decoder output to vocabulary size for token prediction

### Architecture Diagram

```
Input Sequence
      ↓
Input Embedding + Positional Encoding
      ↓
    Encoder Stack (N layers)
      ↓ (Context Vectors)
Output Sequence → Embedding + Positional Encoding
      ↓
    Decoder Stack (N layers)
      ↓
  Projection Layer
      ↓
Output Probability Distribution
```

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohitGoyal09/Attention-is-all-you-need.git
   cd Attention-is-all-you-need
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

The project requires the following key packages:
- **PyTorch 2.0.1** - Deep learning framework
- **TorchText 0.15.2** - Text processing utilities
- **Datasets 2.15.0** - Hugging Face datasets library
- **Tokenizers 0.13.3** - Fast tokenization library
- **TorchMetrics 1.0.3** - PyTorch metric library
- **TensorBoard 2.13.0** - Training visualization

## Quick Start

### 1. Training a Model

```bash
python train.py
```

This will:
- Load or create tokenizers for source and target languages
- Initialize the Transformer model
- Train for the specified number of epochs (default: 20)
- Save model checkpoints after each epoch
- Log metrics to TensorBoard

### 2. Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=runs/tmodel
```

Then navigate to `http://localhost:6006` in your browser.

### 3. Model Checkpoint Management

The training script automatically:
- Saves model weights to `opus_books_weights/` directory
- Resumes from the latest checkpoint if `preload: 'latest'` is set in config
- Maintains optimizer state for continued training

## Usage

### Training Configuration

Edit `config.py` to customize training parameters:

```python
{
    "batch_size": 8,              # Batch size for training
    "num_epochs": 20,             # Number of training epochs
    "lr": 1e-4,                   # Learning rate (Adam optimizer)
    "seq_len": 350,               # Maximum sequence length
    "d_model": 512,               # Model dimension
    "datasource": 'opus_books',   # Dataset source
    "lang_src": "en",             # Source language (English)
    "lang_tgt": "it",             # Target language (Italian)
    "model_folder": "weights",    # Weights directory
    "preload": "latest",          # Resume from latest checkpoint
    "tokenizer_file": "tokenizer_{0}.json",  # Tokenizer file format
    "experiment_name": "runs/tmodel"  # TensorBoard experiment name
}
```

### Custom Model Configuration

To use custom hyperparameters, modify the `build_transformer` function call in `train.py`:

```python
model = build_transformer(
    src_vocab_size=vocab_size_src,
    tgt_vocab_size=vocab_size_tgt,
    src_seq_len=config["seq_len"],
    tgt_seq_len=config["seq_len"],
    h=8,              # Number of attention heads (default: 8)
    d_model=512,      # Model dimension (default: 512)
    N=6,              # Number of encoder/decoder layers (default: 6)
    dropout=0.1,      # Dropout probability (default: 0.1)
    d_ff=2048         # Feed-forward hidden dimension (default: 2048)
)
```

### Training

```bash
# Train from scratch
python train.py

# Resume from latest checkpoint (set preload: 'latest' in config.py)
python train.py

# Resume from specific epoch
# Modify config.py: "preload": "00"  (for epoch 00)
```

### Inference

The `greedy_decode` function in `train.py` provides decoding during validation:

```python
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs greedy decoding to generate target sequence.
    
    Args:
        model: Transformer model
        source: Source sequence tensor
        source_mask: Source attention mask
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
        max_len: Maximum sequence length
        device: Computation device (cuda/cpu)
    
    Returns:
        Generated target sequence
    """
```

**Example Usage:**
```python
# Encode source sequence
encoder_output = model.encode(source, source_mask)

# Iteratively generate target tokens
decoder_input = torch.empty(1, 1).fill_(sos_idx).to(device)
for _ in range(max_len):
    decoder_mask = causal_mask(decoder_input.size(1)).to(device)
    output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
    next_token = torch.max(model.project(output[:, -1]), dim=1)[1]
    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
    if next_token == eos_idx:
        break
```

## Configuration

### Default Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 8 | Training batch size |
| num_epochs | 20 | Total training epochs |
| lr | 1e-4 | Adam optimizer learning rate |
| seq_len | 350 | Maximum sequence length |
| d_model | 512 | Transformer model dimension |
| h | 8 | Number of attention heads |
| N | 6 | Number of stacked layers |
| dropout | 0.1 | Dropout rate |
| d_ff | 2048 | Feed-forward hidden dimension |
| datasource | opus_books | Dataset source (Hugging Face) |
| lang_src | en | Source language |
| lang_tgt | it | Target language |

### Model Hyperparameters

Key architectural parameters can be adjusted in `build_transformer()`:

- **h**: Number of attention heads (impacts computational cost and representation capacity)
- **d_model**: Model dimension (must be divisible by h)
- **N**: Number of encoder/decoder layers (depth of the model)
- **dropout**: Regularization (typical range: 0.1-0.3)
- **d_ff**: Feed-forward inner dimension (typically 4× d_model)

## Model Components

### 1. **InputEmbedding**
Converts token IDs to embeddings and scales by √d_model

**Key Parameters:**
- `vocab_size`: Vocabulary size
- `d_model`: Embedding dimension

### 2. **PositionalEncoding**
Adds sinusoidal positional information to embeddings

**Key Parameters:**
- `d_model`: Model dimension
- `seq_len`: Maximum sequence length
- `dropout`: Regularization rate

### 3. **MultiHeadAttentionBlock**
Implements scaled dot-product attention with multiple heads

**Key Parameters:**
- `d_model`: Model dimension
- `n_heads`: Number of parallel attention heads
- `dropout`: Regularization rate

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
```

### 4. **FeedForwardBlock**
Two-layer fully connected network with ReLU activation

**Architecture:**
```
x → Linear(d_model→d_ff) → ReLU → Dropout → Linear(d_ff→d_model) → y
```

### 5. **EncoderBlock**
Self-attention + Feed-forward with residual connections and layer norm

**Processing Pipeline:**
```
x → LayerNorm → MultiHeadAttention → Residual Addition
  → LayerNorm → FeedForward → Residual Addition → y
```

### 6. **DecoderBlock**
Self-attention + Cross-attention + Feed-forward with residual connections

**Processing Pipeline:**
```
x → Self-Attention → Residual
  → Cross-Attention (with encoder output) → Residual
  → FeedForward → Residual → y
```

### 7. **Transformer**
Complete encoder-decoder architecture combining all components

## Dataset

### Supported Datasets

The project uses Hugging Face Datasets library and supports any bilingual dataset in the Hugging Face format.

**Default Configuration:** OPUS Books (English-Italian)
- **Source**: English
- **Target**: Italian
- **Train/Validation Split**: 90%/10%
- **Data Format**: Translation pairs

### Dataset Preparation

```python
# Automatic dataset loading in train.py
ds_raw = load_dataset('opus_books', 'en-it', split='train')

# Tokenization
tokenizer_src = get_or_build_tokenizer(config, ds_raw, 'en')
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, 'it')

# Dataset creation
train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, 'en', 'it', seq_len)
```

### Custom Datasets

To use a different language pair:

1. Update `config.py`:
   ```python
   "datasource": "opus_books",
   "lang_src": "en",      # Change source language
   "lang_tgt": "de",      # Change target language (e.g., German)
   ```

2. The tokenizers will be automatically created and saved

### Tokenization

- **Tokenizer Type**: Word-level (WordLevel)
- **Vocabulary**: Built from training data with minimum frequency threshold
- **Special Tokens**: `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`
- **Format**: JSON (stored as `tokenizer_en.json`, `tokenizer_it.json`)

## Evaluation Metrics

The model is evaluated using industry-standard machine translation metrics:

### 1. **Character Error Rate (CER)**
```
CER = (S + D + I) / N

where:
  S = Number of character substitutions
  D = Number of character deletions
  I = Number of character insertions
  N = Total characters in reference
```
Lower is better. Range: 0-1 (0 = perfect)

### 2. **Word Error Rate (WER)**
```
WER = (S + D + I) / N

where:
  S = Number of word substitutions
  D = Number of word deletions
  I = Number of word insertions
  N = Total words in reference
```
Lower is better. Range: 0-1 (0 = perfect)

### 3. **BLEU Score**
Bilingual Evaluation Understudy (BLEU) measures n-gram overlap between predicted and reference translations.

```
BLEU = BP × exp(∑ wn log pn)

where:
  BP = Brevity penalty
  pn = Precision for n-gram matches
  wn = Weight for each n-gram (typically uniform at 0.25)
```
Higher is better. Range: 0-1 (1 = perfect)

### Metrics Interpretation

| Metric Range | Quality Assessment |
|--------------|-------------------|
| CER/WER < 0.1 | Excellent (>90% accuracy) |
| CER/WER 0.1-0.3 | Good (70-90% accuracy) |
| CER/WER 0.3-0.5 | Moderate (50-70% accuracy) |
| CER/WER > 0.5 | Poor (<50% accuracy) |

| BLEU Score | Quality Assessment |
|-----------|-------------------|
| > 0.40 | Excellent |
| 0.30-0.40 | Good |
| 0.20-0.30 | Moderate |
| < 0.20 | Poor |

## Results

### Training Dynamics

The model implements:
- **Loss Function**: Cross-entropy with label smoothing (0.1)
- **Optimizer**: Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- **Warmup**: Custom learning rate scheduling
- **Regularization**: Dropout (0.1) and label smoothing

### Expected Performance

For English-Italian translation on OPUS Books:
- Training typically converges within 15-20 epochs
- Validation metrics improve steadily with training
- Model performance plateaus after reaching optimal configuration

### Sample Translations

Example outputs during validation:
```
SOURCE: The quick brown fox jumps over the lazy dog
TARGET: La veloce volpe marrone salta sopra il cane pigro
PREDICTED: Il veloce volpe marrone salti sopra il pigro cane
```

## Contributing

We welcome contributions! Please follow these guidelines:

### Getting Started

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit
   ```bash
   git commit -m "Add your descriptive commit message"
   ```
4. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all classes and functions
- Include type hints where appropriate

### Testing

Before submitting a PR:
- Verify the code runs without errors
- Test with different dataset configurations
- Ensure metrics are computed correctly
- Check for memory leaks with larger batches

### Areas for Contribution

- [ ] Beam search decoding implementation
- [ ] Additional evaluation metrics
- [ ] Data augmentation strategies
- [ ] Alternative attention mechanisms
- [ ] Distributed training support
- [ ] Inference optimization (quantization, pruning)
- [ ] Additional language pairs and datasets
- [ ] Documentation improvements

## References

### Primary Research

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Paper: https://arxiv.org/abs/1706.03762
   - Foundation for this implementation

2. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2014)
   - Introduces attention mechanisms
   - https://arxiv.org/abs/1409.0473

3. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
   - Foundation of sequence transduction models
   - https://arxiv.org/abs/1409.3215

### Implementation Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)
- [TorchMetrics Documentation](https://torchmetrics.readthedocs.io/)

### Related Implementations

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- [Fairseq](https://github.com/pytorch/fairseq)

## Acknowledgments

This implementation is based on the seminal work "Attention Is All You Need" by Vaswani et al. (2017). Thanks to the PyTorch and open-source community for excellent libraries and tools.

## License

This project is open source and available under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niklas and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Ilya},
  journal={arXiv preprint arXiv:1706.03762},
  year={2017}
}
```

## Contact & Support

For questions or issues:
- Open an [Issue](https://github.com/MohitGoyal09/Attention-is-all-you-need/issues) on GitHub
- Check existing [Discussions](https://github.com/MohitGoyal09/Attention-is-all-you-need/discussions)
- Review the [Wiki](https://github.com/MohitGoyal09/Attention-is-all-you-need/wiki) for additional documentation

---

**Last Updated**: 2024
**Python Version**: 3.9+
**PyTorch Version**: 2.0.1+

Made with ❤️ for the machine learning community
