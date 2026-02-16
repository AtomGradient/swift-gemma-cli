# gemma-cli

A Swift CLI tool for running **Gemma 3 4B IT VLM** inference on Apple Silicon with [MLX](https://github.com/ml-explore/mlx-swift).

Supports both text-only and multimodal (text + image) inference with streaming output and performance statistics.

## Requirements

- macOS 14+
- Apple Silicon (M1/M2/M3/M4)
- Xcode 16+ / Swift 5.12+
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) cloned at `../mlx-swift-lm`

## Build

```bash
cd gemma-cli
swift build -c release
```

## Usage

```
gemma-cli <model-path> [--image <path>] [--prompt <text>] [--max-tokens N] [--temperature F]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `<model-path>` | (required) | Path to the local model directory |
| `--image <path>` | (none) | Path to an image file for multimodal inference |
| `--prompt`, `-p` | `"Hello, how are you?"` | Prompt text |
| `--max-tokens`, `-m` | `100` | Maximum tokens to generate |
| `--temperature`, `-t` | `0.6` | Sampling temperature (0 = greedy) |
| `--top-p` | `1.0` | Top-p nucleus sampling |
| `--repetition-penalty` | (none) | Penalty factor for repeating tokens |
| `--repetition-context-size` | `20` | Number of tokens for repetition penalty context |

## Models

### Original Model

[gemma-3-4b-it-qat-4bit](https://huggingface.co/mlx-community/gemma-3-4b-it-qat-4bit) — The original QAT 4-bit quantized Gemma 3 4B IT model.

- Disk: **2.8 GB**
- Vision: 896px, 27-layer SigLIP, 256 image tokens
- Text: 34 layers, vocab 262K

### Optimized Model (test-s4-res672)

Our pruned model after applying Steps 1-4 of the optimization pipeline:

1. **Vocab pruning**: 262K → 80K tokens (remove unused CJK tokens)
2. **Vision fc2 quantization**: bf16 → 4-bit (pad intermediate 4304 → 4352 for group_size alignment)
3. **Text layer pruning**: remove layers 31, 32, 33 (34 → 31 layers)
4. **Resolution reduction**: 896 → 672 (patches 4096 → 2304, image tokens 256 → 144)

- Disk: **2.5 GB**
- Vision: 672px, 27-layer SigLIP, 144 image tokens
- Text: 31 layers, vocab 262K (with token_map to 80K compact embeddings)

### Fully Optimized Model (step7-final-split)

After all 7 optimization steps (adds neuron pruning + weight splitting):

- Disk: **2.0 GB** (language 1.8 GB + vision 231 MB)
- Text: 31 layers, per-layer MLP pruning (back 17 layers -25% neurons)
- Text-only runtime: ~2.5 GB (load language_model only)
- Image runtime: ~2.8 GB (load both)

## Examples

### Text-only inference

```bash
# Original model
swift run -c release gemma-cli /path/to/gemma-3-4b-it-qat-4bit \
  --prompt "Hello, how are you?" --max-tokens 100 --temperature 0.0

# Optimized model
swift run -c release gemma-cli /path/to/gemma-pruned-models/test-s4-res672 \
  --prompt "Hello, how are you?" --max-tokens 100 --temperature 0.0
```

### Image understanding

```bash
# Original model (896px vision)
swift run -c release gemma-cli /path/to/gemma-3-4b-it-qat-4bit \
  --image photo.jpg \
  --prompt "Describe this image in detail." --max-tokens 200 --temperature 0.0

# Optimized model (672px vision)
swift run -c release gemma-cli /path/to/gemma-pruned-models/test-s4-res672 \
  --image photo.jpg \
  --prompt "Describe this image in detail." --max-tokens 200 --temperature 0.0
```

### Output format

```
Loading model from: /path/to/model...
Model loaded.

Prompt: Describe this image in detail.
------
[streaming generated text...]
------
Prompt:     271 tokens, 111.85 tokens/s, 2.42s
Generation: 200 tokens, 89.56 tokens/s, 2.23s
Peak memory: 5284 MB
```

## Performance Comparison

Tested on Apple Silicon with a pizza photo (IMG_3420_small.jpg).

### Text-only (prompt: "Hello, how are you?", max_tokens=50)

| Model | Prompt Speed | Generation Speed | Peak Memory |
|-------|-------------|-----------------|-------------|
| Original (gemma-3-4b-it-qat-4bit) | 7.3 tokens/s | 97.5 tokens/s | 2,910 MB |

### Image Understanding (prompt: "Describe this image in detail.", max_tokens=200)

| Model | Prompt Tokens | Prompt Speed | Generation Speed | Peak Memory |
|-------|--------------|-------------|-----------------|-------------|
| Original (896px) | 271 | 111.9 tokens/s | 89.6 tokens/s | 5,284 MB |
| Optimized (672px) | 159 | 89.5 tokens/s | 100.3 tokens/s | 4,699 MB |
| **Improvement** | **-41%** | — | **+12%** | **-585 MB (-11%)** |

### Key Observations

- **Prompt tokens reduced 41%**: 672px resolution yields 144 image tokens (vs 256 at 896px), directly reducing prompt processing work.
- **Generation speed improved 12%**: Fewer text layers (31 vs 34) and smaller KV cache accelerate token generation.
- **Peak memory reduced 585 MB**: Smaller vision activations + fewer parameters.
- **Image understanding quality preserved**: Both models correctly identify pizza, box, sauce, cheese, and toppings. The 672px model produces slightly less detailed descriptions but no hallucinations or quality collapse.
- **448px resolution was tested and rejected**: It causes token repetition loops due to insufficient visual information. 672px is the optimal balance point.

## Optimization Pipeline Summary

| Step | Operation | Disk Savings | Status |
|------|-----------|-------------|--------|
| 1 | Vocab pruning (262K → 80K) | 261 MB | Done |
| 2 | Vision fc2 bf16 → 4-bit (pad 4304 → 4352) | 191 MB | Done |
| 3 | Remove text layers 31, 32, 33 | 159 MB | Done |
| 4 | Resolution 896 → 672 | ~1 MB disk / runtime savings | Done |
| ~~5~~ | ~~Remove vision layers 12-15~~ | ~~35 MB~~ | Removed (destroys image quality) |
| 6 | MLP neuron pruning (back 17 layers -25%) | 188 MB | Done |
| 7 | Weight splitting (language + vision) | — | Done |

**Total disk reduction: 2.8 GB → 2.0 GB**

## Project Structure

```
gemma-cli/
  Package.swift           # Swift package manifest
  Sources/
    GemmaCLI.swift        # Main CLI implementation
  README.md               # This file
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (>= 0.30.3) — MLX framework for Apple Silicon
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (local) — VLM/LLM model loading and inference
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) (>= 1.5.0) — CLI argument parsing

## License

This tool is for research and evaluation purposes.
