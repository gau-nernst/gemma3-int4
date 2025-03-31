# Gemma 3 INT4

Gemma 3 provides FP32 checkpoints trained with INT4 Quantization Aware Training (QAT). They have the following characteristics
- INT4 symmetric quantization i.e. no zero point, with group size = 32
- Tied embeddings

No established libraries (in PyTorch) can take full advantage of the above. Hence, this library aims to do it.

Plan:
- vLLM / SGLang integration

Convert checkpoint

```bash
uv run convert_flax.py --ckpt_dir gemma3-1b-it-int4 --save_dir gemma-3-1b-it-int4 --format int4
# other possible --format values: awq, gguf
```

For GGUF, you will need a "donor" GGUF to copy the tokenizer over (I'm too lazy to re-implement the logic to serialize tokenizer to GGUF format).

```bash
# create the donor GGUF, where convert_hf_to_gguf.py is found in llama.cpp repo
python convert_hf_to_gguf.py $(huggingface-cli download google/gemma-3-4b-it) --outtype bf16 --outfile gemma-3-4b-it-BF16.gguf

# combine weights from convert_flax.py and metadata from convert_hf_to_gguf.py
python convert_gguf.py --ckpt gemma-3-4b-it-int4-gguf/model.safetensors --metadata gemma-3-4b-it-BF16.gguf --save_path gemma-3-4b-it-q4_0.gguf
```

Chat demo

```bash
python chat_hf.py --model gaunernst/gemma-3-1b-it-int4
```

Model       | Vision | INT4 link | AWQ link | GGUF link
------------|--------|-----------|----------|-----------
Gemma 3 1B  |        | [Link](https://huggingface.co/gaunernst/gemma-3-1b-it-int4) | [Link](https://huggingface.co/gaunernst/gemma-3-1b-it-int4-awq) | [Link](https://huggingface.co/gaunernst/gemma-3-1b-it-int4-gguf)
Gemma 3 4B  | ✅     | [Link](https://huggingface.co/gaunernst/gemma-3-4b-it-int4) | [Link](https://huggingface.co/gaunernst/gemma-3-4b-it-int4-awq) | [Link](https://huggingface.co/gaunernst/gemma-3-4b-it-int4-gguf)
Gemma 3 12B | ✅     | [Link](https://huggingface.co/gaunernst/gemma-3-12b-it-int4) | [Link](https://huggingface.co/gaunernst/gemma-3-12b-it-int4-awq) | [Link](https://huggingface.co/gaunernst/gemma-3-12b-it-int4-gguf)
Gemma 3 27B | ✅     | [Link](https://huggingface.co/gaunernst/gemma-3-27b-it-int4) | [Link](https://huggingface.co/gaunernst/gemma-3-27b-it-int4-awq) | [Link](https://huggingface.co/gaunernst/gemma-3-27b-it-int4-gguf)
