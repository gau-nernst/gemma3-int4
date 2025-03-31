# /// script
# dependencies = [
#   "jaxlib",
#   "orbax-checkpoint",
#   "numpy",
#   "safetensors",
#   "tqdm",
# ]
# ///

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from safetensors.flax import save_file
from tqdm import tqdm

SIGLIP_PREFIX = "SigLiPFromPatches_0/siglip_encoder"


def flatten(x: np.ndarray, start: int = 0, end: int = -1):
    if start < 0:
        start += x.ndim
    if end < 0:
        end += x.ndim
    new_shape = x.shape[:start] + (-1,) + x.shape[end + 1 :]
    return x.reshape(new_shape)


def unflatten(x: np.ndarray, dim: int, sizes: tuple[int, ...]):
    new_shape = x.shape[:dim] + tuple(sizes) + x.shape[dim + 1 :]
    return x.reshape(new_shape)


# correct quantization parameters mean quantization error = 0 (or close to 0)
def check_groups(groups: np.ndarray, scales: np.ndarray, dim: int):
    # groups: (a, b, c, 32, d, e, f)
    # scales: (a, b, c,  1, d, e, f)
    inv_scale = 1.0 / scales.clip(1e-12)
    q_group = np.round(groups * inv_scale)
    max_diff = np.abs(q_group * scales - groups).max(dim, keepdims=True)
    return max_diff < 1e-6, max_diff


def find_scales(w: np.ndarray, dim: int):
    w = unflatten(w, dim, (-1, 32))
    group_range = w.max(dim + 1, keepdims=True) - w.min(dim + 1, keepdims=True)

    scales = np.zeros_like(group_range)
    for q in range(15, 0, -1):
        try_scale = group_range / q
        ok, _ = check_groups(w, try_scale, dim + 1)
        scales[ok] = try_scale[ok]

    ok, _ = check_groups(w, scales, dim + 1)
    assert ok.all()

    inv_scales = 1 / scales.clip(1e-12)
    qweight = np.round(w * inv_scales)
    assert qweight.max() <= 7
    assert qweight.min() >= -8

    return scales.squeeze(dim + 1), qweight


def convert_siglip(params, num_layers: int):
    state_dict = dict()

    def convert_layer(prefix: str, layer: dict[str, np.ndarray]):
        bias = layer["bias"]

        if "kernel" in layer:
            w = layer["kernel"]
            if w.ndim == 2:  # linear layer
                w = w.T

            elif w.ndim == 3:  # attn projection
                # qkv projection - (dim, num_heads, head_dim)
                if bias.ndim == 2:
                    w = flatten(w, 1, 2).T
                    bias = bias.reshape(-1)

                # o projection - (num_heads, head_dim, dim)
                elif bias.ndim == 1:
                    w = flatten(w, 0, 1).T

            elif w.ndim == 4:  # conv2d layer
                w = w.transpose(3, 2, 0, 1)

            else:
                raise RuntimeError(f"Unsupported {w.shape=}")

        elif "scale" in layer:  # layer norm
            w = layer["scale"]

        else:
            raise RuntimeError

        state_dict[f"{prefix}weight"] = w
        state_dict[f"{prefix}bias"] = bias

    convert_layer("embeddings.patch_embedding.", params[f"{SIGLIP_PREFIX}/embedding"])
    state_dict["embeddings.position_embedding.weight"] = params[SIGLIP_PREFIX]["pos_embedding"].squeeze(0)
    convert_layer("post_layernorm.", params[f"{SIGLIP_PREFIX}/Transformer/encoder_norm"])

    for layer_idx in range(num_layers):
        prefix = f"encoder.layers.{layer_idx}."
        layer_prefix = f"{SIGLIP_PREFIX}/Transformer/encoderblock_{layer_idx}/"

        convert_layer(f"{prefix}layer_norm1.", params[f"{layer_prefix}LayerNorm_0"])
        convert_layer(f"{prefix}layer_norm2.", params[f"{layer_prefix}LayerNorm_1"])

        attn_prefix = f"{layer_prefix}MultiHeadDotProductAttention_0/"
        convert_layer(f"{prefix}self_attn.q_proj.", params[f"{attn_prefix}query"])
        convert_layer(f"{prefix}self_attn.k_proj.", params[f"{attn_prefix}key"])
        convert_layer(f"{prefix}self_attn.v_proj.", params[f"{attn_prefix}value"])
        convert_layer(f"{prefix}self_attn.out_proj.", params[f"{attn_prefix}out"])

        mlp_prefix = f"{layer_prefix}MlpBlock_0/"
        convert_layer(f"{prefix}mlp.fc1.", params[f"{mlp_prefix}Dense_0"])
        convert_layer(f"{prefix}mlp.fc2.", params[f"{mlp_prefix}Dense_1"])

    return state_dict


# convert to HF format first, then apply quantization
def convert_to_hf(path: Path):
    path = path.absolute()  # orbax only works with absolute path
    ckpt = ocp.StandardCheckpointer()
    metadata = dict(ckpt.metadata(path))
    metadata = jax.tree.map(ocp.utils.to_shape_dtype_struct, metadata)

    num_layers = num_siglip_layers = 0
    while f"transformer/layer_{num_layers}/attn/_key_norm" in metadata:
        num_layers += 1
    while f"{SIGLIP_PREFIX}/Transformer/encoderblock_{num_siglip_layers}/LayerNorm_0" in metadata:
        num_siglip_layers += 1
    print(f"{num_layers=}")
    print(f"{num_siglip_layers=}")

    # NOTE: all gemma3 models use tied embeddings, even for the 27B version.
    params = ckpt.restore(path)
    state_dict = dict()

    if num_siglip_layers > 0:
        # HF append unused tokens for no reason???
        embed = params["transformer/embedder"]["input_embedding"]
        params["transformer/embedder"]["input_embedding"] = np.pad(embed, ((0, 64), (0, 0)))
        gemma_prefix = "language_model."

        prefix = "multi_modal_projector.mm_"
        jax_prefix = "transformer/embedder/"
        state_dict[f"{prefix}input_projection_weight"] = params[f"{jax_prefix}mm_input_projection"]["w"]
        state_dict[f"{prefix}soft_emb_norm.weight"] = params[f"{jax_prefix}mm_soft_embedding_norm"]["scale"]

    else:
        gemma_prefix = ""

    state_dict[f"{gemma_prefix}model.embed_tokens.weight"] = params["transformer/embedder"]["input_embedding"]
    state_dict[f"{gemma_prefix}model.norm.weight"] = params["transformer/final_norm"]["scale"]

    yield state_dict

    for layer_idx in range(num_layers):
        jax_prefix = f"transformer/layer_{layer_idx}/"

        state_dict = dict()
        prefix = f"{gemma_prefix}model.layers.{layer_idx}."
        state_dict[f"{prefix}input_layernorm.weight"] = params[f"{jax_prefix}pre_attention_norm"]["scale"]
        state_dict[f"{prefix}post_attention_layernorm.weight"] = params[f"{jax_prefix}post_attention_norm"]["scale"]
        state_dict[f"{prefix}pre_feedforward_layernorm.weight"] = params[f"{jax_prefix}pre_ffw_norm"]["scale"]
        state_dict[f"{prefix}post_feedforward_layernorm.weight"] = params[f"{jax_prefix}post_ffw_norm"]["scale"]

        prefix = f"{gemma_prefix}model.layers.{layer_idx}.self_attn."
        jax_prefix = f"transformer/layer_{layer_idx}/attn/"
        state_dict[f"{prefix}q_norm.weight"] = params[f"{jax_prefix}_query_norm"]["scale"]
        state_dict[f"{prefix}k_norm.weight"] = params[f"{jax_prefix}_key_norm"]["scale"]

        # (num_heads, hidden_size, head_dim) -> (num_heads * head_dim, hidden_size)
        state_dict[f"{prefix}q_proj.weight"] = flatten(params[f"{jax_prefix}q_einsum"]["w"].transpose(0, 2, 1), end=1)
        state_dict[f"{prefix}k_proj.weight"] = flatten(
            params[f"{jax_prefix}kv_einsum"]["w"][0].transpose(0, 2, 1), end=1
        )
        state_dict[f"{prefix}v_proj.weight"] = flatten(
            params[f"{jax_prefix}kv_einsum"]["w"][1].transpose(0, 2, 1), end=1
        )

        # (num_heads, head_dim, hidden_size) -> (hidden_size, num_heads * head_dim)
        state_dict[f"{prefix}o_proj.weight"] = flatten(params[f"{jax_prefix}attn_vec_einsum"]["w"], end=1).T

        prefix = f"{gemma_prefix}model.layers.{layer_idx}.mlp."
        jax_prefix = f"transformer/layer_{layer_idx}/mlp/"
        state_dict[f"{prefix}gate_proj.weight"] = params[f"{jax_prefix}gating_einsum"]["w"][0]
        state_dict[f"{prefix}up_proj.weight"] = params[f"{jax_prefix}gating_einsum"]["w"][1]
        state_dict[f"{prefix}down_proj.weight"] = params[f"{jax_prefix}linear"]["w"].T

        yield state_dict

    # vision tower
    if num_siglip_layers > 0:
        siglip_state_dict = convert_siglip(params, num_siglip_layers)
        for k, v in siglip_state_dict.items():
            state_dict[f"vision_tower.vision_model.{k}"] = v
        yield state_dict


def convert_int4(state_dict: dict[str, np.ndarray]):
    int4_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith(("vision_tower", "multi_modal_projector")) or v.ndim == 1:  # vision tower is not quantized
            int4_state_dict[k] = v.astype(jnp.bfloat16)
            continue

        assert v.ndim == 2
        N, K = v.shape
        scales, qweight = find_scales(v, dim=1)  # (N, K/32) and (N, K/32, 32)

        qweight_i8 = qweight.astype(np.int8)
        qweight_i4 = (qweight_i8[..., 0::2] & 0xF) | (qweight_i8[..., 1::2] << 4)

        int4_state_dict[f"{k}.int4_data"] = qweight_i4.reshape(N, K // 2)
        int4_state_dict[f"{k}.scales"] = scales.astype(jnp.bfloat16)

    return int4_state_dict


def convert_awq(state_dict: dict[str, np.ndarray]):
    awq_state_dict = dict()

    for k, v in state_dict.items():
        if (
            k.endswith("model.embed_tokens.weight")  # AWQ doesn't support INT4 embeddings
            or k.startswith(("vision_tower", "multi_modal_projector"))  # vision tower is not quantized
            or v.ndim == 1
        ):
            awq_state_dict[k] = v.astype(jnp.bfloat16)
            continue

        assert v.ndim == 2
        v = v.T  # AWQ transpose the weight

        K, N = v.shape
        scales, qweight = find_scales(v, dim=0)  # (K/32, N) and (K/32, 32, N)

        # AWQ is actually UINT4 (instead of INT4)
        # hence, we will shift qweight up by 8 (even though Google AQT only uses [-7,7])
        # and set zero_point = 8
        qweight = (qweight + 8).astype(np.uint32)

        # AWQ pack 8 int4 into UINT32 in the following layout (from high bits to low bits)
        # [7 5 3 1 6 4 2 0] along the 2nd dim
        qweight = qweight.reshape(K, N // 8, 8)
        qweight_packed = (
            (qweight[..., 7] << (7 * 4))
            | (qweight[..., 5] << (6 * 4))
            | (qweight[..., 3] << (5 * 4))
            | (qweight[..., 1] << (4 * 4))
            | (qweight[..., 6] << (3 * 4))
            | (qweight[..., 4] << (2 * 4))
            | (qweight[..., 2] << (1 * 4))
            | (qweight[..., 0] << (0 * 4))
        )
        qweight_packed = qweight_packed.view(np.int32).reshape(K, N // 8)

        prefix = k.removesuffix(".weight")
        awq_state_dict[f"{prefix}.qweight"] = qweight_packed
        awq_state_dict[f"{prefix}.qzeros"] = np.full((K // 32, N // 8), 0x8888_8888, dtype=np.uint32).view(np.int32)
        awq_state_dict[f"{prefix}.scales"] = scales.astype(jnp.bfloat16)

    return awq_state_dict


def convert_gguf(state_dict: dict[str, np.ndarray]):
    def map_key(k: str):
        k = k.replace("model.embed_tokens.", "token_embd.")
        k = k.replace("model.layers.", "blk.")
        k = k.replace(".input_layernorm.", ".attn_norm.")
        k = k.replace(".self_attn.q_norm.", ".attn_q_norm.")
        k = k.replace(".self_attn.k_norm.", ".attn_k_norm.")
        k = k.replace(".self_attn.q_proj.", ".attn_q.")
        k = k.replace(".self_attn.k_proj.", ".attn_k.")
        k = k.replace(".self_attn.v_proj.", ".attn_v.")
        k = k.replace(".self_attn.o_proj.", ".attn_output.")
        k = k.replace(".post_attention_layernorm.", ".post_attention_norm.")
        k = k.replace(".pre_feedforward_layernorm.", ".ffn_norm.")
        k = k.replace(".mlp.up_proj.", ".ffn_up.")
        k = k.replace(".mlp.gate_proj.", ".ffn_gate.")
        k = k.replace(".mlp.down_proj.", ".ffn_down.")
        k = k.replace(".post_feedforward_layernorm.", ".post_ffw_norm.")
        k = k.replace("model.norm.", "output_norm.")
        return k

    gguf_state_dict = dict()

    for k, v in state_dict.items():
        # skip vision tower
        if k.startswith(("vision_tower", "multi_modal_projector")):
            continue

        k = map_key(k.removeprefix("language_model."))

        if k == "token_embd.weight" and v.shape[1] > 1152:
            v = v[:-64]  # remove HF strange padding

        if v.ndim == 1:
            v = v.astype(np.float32)  # GGUF use FP32 for bias and norm weight/bias
            gguf_state_dict[k] = v + 1 if k.endswith("norm.weight") else v
            continue

        assert v.ndim == 2
        N, K = v.shape
        scales, qweight = find_scales(v, dim=1)  # (N, K/32, 1) and (N, K/32, 32)

        # Q4_0
        # https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
        qweight_u8 = (qweight + 8).astype(np.uint8)  # shift [-8,7] to [0,15]
        qweight_u4 = qweight_u8[..., :16] | (qweight_u8[..., 16:] << 4)  # (N, K/32, 16)

        scales_f16 = scales[..., None].astype(np.float16).view(np.uint8)
        data = np.concatenate([scales_f16, qweight_u4], axis=-1)  # (N, K/32, 2 + 16)
        gguf_state_dict[k] = data.reshape(N, K // 32 * (2 + 16))

    return gguf_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, type=Path)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    converter = dict(
        int4=convert_int4,
        awq=convert_awq,
        gguf=convert_gguf,
    )[args.format]

    total_size = 0
    weight_map = dict()

    state_dict = dict()
    size = 0
    shard_idx = 0
    filename = f"model-{shard_idx + 1:05d}.safetensors"
    for sub_state_dict in tqdm(convert_to_hf(args.ckpt_dir)):
        sub_state_dict = converter(sub_state_dict)
        new_size = sum(v.nbytes for v in sub_state_dict.values())

        if size + new_size > 5e9:
            save_file(state_dict, args.save_dir / filename)
            state_dict = dict()
            size = 0
            shard_idx += 1
            filename = f"model-{shard_idx + 1:05d}.safetensors"

        # assume that new_size < 5e9
        size += new_size
        total_size += new_size
        for k, v in sub_state_dict.items():
            state_dict[k] = v
            weight_map[k] = filename

    save_file(state_dict, args.save_dir / filename)
    json.dump(
        dict(metadata=dict(total_size=total_size), weight_map=weight_map),
        open(args.save_dir / "model.safetensors.index.json", "w"),
    )
