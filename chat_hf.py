import argparse

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3TextConfig, TextStreamer, pipeline
from transformers.modeling_utils import _get_resolved_checkpoint_files

import subclass


def create_new_init(old_init):
    def new_init(*args, **kwargs):
        kwargs.update(device="meta")
        old_init(*args, **kwargs)

    return new_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    nn.Linear.__init__ = create_new_init(nn.Linear.__init__)
    nn.Embedding.__init__ = create_new_init(nn.Embedding.__init__)

    cfg: Gemma3TextConfig = Gemma3TextConfig.from_pretrained(args.model)
    model = Gemma3ForCausalLM(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ckpts, _ = _get_resolved_checkpoint_files(
        args.model, "", None, None, False, False, True, None, False, None, False, None, None, None, None
    )
    state_dict = dict()
    for ckpt_path in ckpts:
        print(f"Load checkpoint {ckpt_path}")
        state_dict.update(load_file(ckpt_path))

    state_dict = subclass.from_plain(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    assert len(missing_keys) == 1 and missing_keys[0] == "lm_head.weight"
    assert len(unexpected_keys) == 0
    model.tie_weights()

    subclass._dequantize = torch.compile(subclass._dequantize)

    device = "cuda"
    model.to(device).eval()
    # TODO: figure out torch.compile
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    chat = []
    while True:
        chat.append(dict(role="user", content=input("User: ")))
        response = pipe(chat, streamer=streamer, max_new_tokens=100_000)
        chat = response[0]["generated_text"]


if __name__ == "__main__":
    main()
