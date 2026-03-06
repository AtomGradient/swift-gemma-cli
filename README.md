# Gemma-Prune

A multi-stage compression pipeline and Swift CLI for deploying **Gemma 3 4B IT VLM** on Apple Silicon edge devices with [MLX](https://github.com/ml-explore/mlx-swift).

We compress the original 2.8 GB QAT 4-bit model to **2.1 GB** while preserving both text generation and image understanding — achieving **22% faster generation**, **3.4x faster image processing**, and **23% lower memory**.

> **Project Page**: [https://atomgradient.github.io/swift-gemma-cli](https://atomgradient.github.io/swift-gemma-cli/)
>
> **Paper**: [Gemma-Prune: A Multi-Stage Compression Pipeline for Deploying Gemma 3 4B VLM on Mobile Devices](https://atomgradient.github.io/swift-gemma-cli/paper.pdf)

## Models

| Model | Size | Description | HuggingFace |
|-------|------|-------------|-------------|
| Original | 2.8 GB | Baseline QAT 4-bit | [mlx-community/gemma-3-4b-it-qat-4bit](https://huggingface.co/mlx-community/gemma-3-4b-it-qat-4bit) |
| **Lite** | 2.3 GB | Vocab pruned + vision fc2 quantized + 3 text layers removed + 672px | [AtomGradient/gemma-3-4b-it-qat-4bit-lite](https://huggingface.co/AtomGradient/gemma-3-4b-it-qat-4bit-lite) |
| **Mobile** | 2.1 GB | All above + neuron pruning + weight splitting | [AtomGradient/gemma-3-4b-it-qat-4bit-mobile](https://huggingface.co/AtomGradient/gemma-3-4b-it-qat-4bit-mobile) |

### Lite (`gemma-3-4b-it-qat-4bit-lite`)

Good balance of compression and quality. Single-file weights, easy to deploy.

- **Disk**: 2.3 GB (`model.safetensors`)
- **Text**: 31 layers, vocab 262K (token_map → 144K compact embeddings)
- **Vision**: 672px, 27-layer SigLIP, 144 image tokens, fc2 4-bit quantized

### Mobile (`gemma-3-4b-it-qat-4bit-mobile`)

Maximum compression for 8 GB mobile devices. Split weights allow text-only lazy loading.

- **Disk**: 2.1 GB (`language_model.safetensors` 1.9 GB + `vision_model.safetensors` 231 MB)
- **Text**: 31 layers, per-layer MLP sizes (layers 14-30 pruned 25% neurons)
- **Vision**: same as Lite
- **Text-only runtime**: ~2.2 GB (loads language model only)

## Performance

Benchmarked on Apple Silicon. Temperature = 0.0, greedy decoding.

### Text Generation

| Model | Disk | Generation (t/s) | Peak Memory |
|-------|------|-------------------|-------------|
| Original | 2.8 GB | 90 | 2910 MB |
| Lite | 2.3 GB | ~110 | ~2500 MB |
| **Mobile** | **2.1 GB** | **110** | **2231 MB** |

### Image Understanding (pizza photo, 200 tokens)

| Model | Prompt (t/s) | Generation (t/s) | Peak Memory | Quality |
|-------|-------------|-------------------|-------------|---------|
| Original (896px) | 54 | 27 | ~5500 MB | Excellent |
| **Mobile (672px)** | **184** | **104** | **4358 MB** | **Good** |
| Improvement | **3.4x** | **3.9x** | **-21%** | |

## Requirements

- macOS 18+
- Apple Silicon (M series or A Series), RAM Size >= 8GB
- Xcode 18+ / Swift 5.0+

## Build

This project depends on a **customized fork** of [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) with added support for `token_map` (vocabulary pruning) and `per_layer_intermediate_sizes` (neuron pruning) in the Gemma 3 model implementation.

### Step 1: Clone both repositories

```bash
# Clone this repo
git clone https://github.com/AtomGradient/swift-gemma-cli.git gemma-cli

# Clone our customized mlx-swift-lm (must be sibling directory)
git clone https://github.com/AtomGradient/mlx-swift-lm.git mlx-swift-lm
```

The directory layout should be:

```
parent-directory/
  gemma-cli/          # this repo
  mlx-swift-lm/       # customized fork (sibling, referenced by Package.swift)
```

### Step 2: Build

```bash
cd gemma-cli
swift build -c release
```

### What's customized in mlx-swift-lm?

The only modified file is `Libraries/MLXVLM/Models/Gemma3.swift`. Changes:

1. **Token map support** — Reads `vocab_pruning.compact_vocab_size` from config, initializes embedding with compact size, loads `token_map` array from weights, remaps token IDs before embedding lookup
2. **Per-layer MLP sizes** — Reads `per_layer_intermediate_sizes` from `text_config`, initializes each transformer block's MLP with its corresponding intermediate dimension
3. **MLXArrayBox wrapper** — Hides `token_map` from MLX Module reflection to avoid weight key mismatch during `model.update()`

These changes are **backward-compatible**: models without `vocab_pruning` or `per_layer_intermediate_sizes` work identically to the original code.

## Usage

```
gemma-cli <model-path> [--image <path>] [--prompt <text>] [--max-tokens N] [--temperature F]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `<model-path>` | (required) | Path to local model directory |
| `--image <path>` | (none) | Image file for multimodal inference |
| `--prompt`, `-p` | `"Hello, how are you?"` | Prompt text |
| `--max-tokens`, `-m` | `100` | Maximum tokens to generate |
| `--temperature`, `-t` | `0.6` | Sampling temperature (0 = greedy) |
| `--top-p` | `1.0` | Top-p nucleus sampling |
| `--repetition-penalty` | (none) | Repetition penalty factor |
| `--repetition-context-size` | `20` | Repetition penalty context window |

### Download models from HuggingFace

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download Lite model (2.3 GB, single file)
huggingface-cli download AtomGradient/gemma-3-4b-it-qat-4bit-lite --local-dir models/lite

# Download Mobile model (2.1 GB, split weights)
huggingface-cli download AtomGradient/gemma-3-4b-it-qat-4bit-mobile --local-dir models/mobile
```

### Examples

```bash
# Text generation (Mobile model)
swift run -c release gemma-cli models/mobile \
  --prompt "Explain quantum computing in simple terms." \
  --max-tokens 200 --temperature 0.0

# Image understanding (Mobile model)
swift run -c release gemma-cli models/mobile \
  --image photo.jpg \
  --prompt "Describe this image in detail." \
  --max-tokens 200 --temperature 0.0

# Original model (for comparison)
swift run -c release gemma-cli models/original \
  --prompt "Hello, how are you?" \
  --max-tokens 100 --temperature 0.0
```

### Output format

```
Loading model from: models/mobile...
Model loaded.

Prompt: Describe this image in detail.
------
[streaming generated text...]
------
Prompt:     159 tokens, 184.03 tokens/s, 0.86s
Generation: 200 tokens, 104.15 tokens/s, 1.92s
Peak memory: 4358 MB
```

## Compression Pipeline

| Step | Operation | Savings | Key Insight |
|------|-----------|---------|-------------|
| 1 | Vocab pruning (262K → 144K tokens) | 170 MB | ASCII vocab scan essential — dictionary-only (80K) breaks generation |
| 2 | Vision fc2 bf16 → 4-bit (pad 4304 → 4352) | 191 MB | 4304 = 16 x 269 (prime), zero-padding to 4352 is mathematically equivalent |
| 3 | Remove text layers 31, 32, 33 | 159 MB | Deepest layers most redundant |
| 4 | Resolution 896 → 672 | runtime | ~3x less vision attention compute; 448px causes token repetition |
| ~~5~~ | ~~Remove vision layers 12-15~~ | ~~35 MB~~ | ~~Destroys image understanding completely~~ |
| 6 | MLP neuron pruning (layers 14-30, -25%) | 188 MB | 60-100% dead neurons in deep layers |
| 7 | Weight splitting (language + vision) | runtime | Text-only: skip loading 231 MB vision weights |

**Total: 2.8 GB → 2.1 GB (25% reduction)**

### Failed Experiments

| Experiment | Result |
|-----------|--------|
| 448px resolution | Token repetition loops (insufficient visual tokens) |
| Remove 4 vision layers | Complete hallucination (pizza → "skin texture") |
| 80K vocabulary (v1) | Generation quality collapse (missing BPE merged tokens) |

## Project Structure

```
gemma-cli/
  Package.swift               # Swift package manifest
  Sources/
    GemmaCLI.swift            # CLI implementation
  docs/
    paper.tex                 # Technical paper (arxiv format)
  README.md
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (>= 0.30.3) — MLX framework for Apple Silicon
- [mlx-swift-lm](https://github.com/AtomGradient/mlx-swift-lm) (local, customized fork) — VLM/LLM model loading with pruning support
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) (>= 1.5.0) — CLI argument parsing

## Citation

```bibtex
@article{atomgradient2026gemmaprune,
  title={Gemma-Prune: A Multi-Stage Compression Pipeline for Deploying Gemma 3 4B Vision-Language Model on Mobile Devices},
  author={AtomGradient},
  year={2026},
  url={https://github.com/AtomGradient/swift-gemma-cli}
}
```

## License

This tool is for research and evaluation purposes. Model weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
