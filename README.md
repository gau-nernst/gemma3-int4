# Gemma 3 INT4

Gemma 3 provides FP32 checkpoints trained with INT4 Quantization Aware Training (QAT). They have the following characteristics
- INT4 symmetric quantization i.e. no zero point, with group size = 32
- Tied embeddings

No established libraries (in PyTorch) can take full advantage of the above. Hence, this library aims to do it.

Plan:
- Use tensor subclass to support tied embeddings
- Safetensors-compatible serialization format. Transform plain tensors <-> tensor subclass
- HF / vLLM / SGLang integration

```
uv run convert_flax.py --ckpt_dir gemma3-1b-it-int4 --save_dir gemma-3-1b-it-int4
```
