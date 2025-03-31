# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "safetensors",
# ]
# ///

import argparse
import struct
import typing
from pathlib import Path

from safetensors.torch import load_file


def _read_number(f: typing.BinaryIO, dtype: str):
    format, nbyte = dict(
        u32=("<L", 4),
        u64=("<Q", 8),
    )[dtype]
    return struct.unpack_from(format, f.read(nbyte))[0]


def _write_number(f: typing.BinaryIO, dtype: str, value: float | int):
    format = dict(
        u8="<B",
        u32="<L",
        u64="<Q",
        f32="<f",
    )[dtype]
    return f.write(struct.pack(format, value))


def _read_str(f: typing.BinaryIO):
    return f.read(_read_number(f, "u64")).decode()


def _write_str(f: typing.BinaryIO, value: str):
    value_b = value.encode()
    _write_number(f, "u64", len(value_b))
    f.write(value_b)


def _write_metadata_value(f: typing.BinaryIO, v):
    if type(v) is int:
        _write_number(f, "u32", 4)
        _write_number(f, "u32", v)
    elif type(v) is str:
        _write_number(f, "u32", 8)
        _write_str(f, v)
    else:
        raise NotImplementedError(f"{type(v)}")


def _copy_metadata_value(fin: typing.BinaryIO, fout: typing.BinaryIO, value_type: int | None = None):
    if value_type is None:
        _write_number(fout, "u32", value_type := _read_number(fin, "u32"))

    lookup = [
        "u8",
        "i8",
        "u16",
        "i16",
        "u32",
        "i32",
        "f32",
        "u8",  # bool
        "str",
        "array",
        "u64",
        "i64",
        "f64",
    ]
    assert value_type < len(lookup), value_type
    value_type = lookup[value_type]

    if value_type == "str":
        _write_str(fout, _read_str(fin))
    elif value_type == "array":
        _write_number(fout, "u32", elem_type := _read_number(fin, "u32"))
        _write_number(fout, "u64", count := _read_number(fin, "u64"))
        for _ in range(count):
            _copy_metadata_value(fin, fout, elem_type)
    else:
        nbytes = int(value_type[1:]) // 8
        fout.write(fin.read(nbytes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()

    state_dict = load_file(args.ckpt)

    num_params = sum(p.numel() for p in state_dict.values())
    model_size = None
    for size in [1, 4, 12, 27]:
        if num_params < size * 1e9:
            model_size = f"{size}b"
            break
    assert model_size is not None

    basic_metadata = {
        "general.architecture": "gemma3",
        "general.quantization_version": 2,
        "general.name": f"gemma-3-{model_size}-it-int4",
    }
    num_metadata_out = len(basic_metadata)

    args.save_path.parent.mkdir(exist_ok=True, parents=True)
    fin = open(args.metadata, "rb")
    fout = open(args.save_path, "wb")

    assert fin.read(4) == b"GGUF"
    assert _read_number(fin, "u32") == 3
    _read_number(fin, "u64")  # num_tensors
    num_metadata_in = _read_number(fin, "u64")  # num_metadata

    fout.write(b"GGUF")
    _write_number(fout, "u32", 3)  # version
    _write_number(fout, "u64", len(state_dict))  # num_tensors
    _write_number(fout, "u64", 0)  # num_metadata. we will write rewind and rewrite this later

    for k, v in basic_metadata.items():
        print(f"{k}: {v}")
        _write_str(fout, k)
        _write_metadata_value(fout, v)

    # copy metadata over
    for _ in range(num_metadata_in):
        key = _read_str(fin)

        if not key.startswith(("gemma3.", "tokenizer.")):
            print(f"Skipping key {key}")
            _copy_metadata_value(fin, open("/dev/null", "wb"))
            continue

        print(f"Copying key {key}")
        _write_str(fout, key)
        _copy_metadata_value(fin, fout)
        num_metadata_out += 1

    fin.close()

    # write the correct num_metadata
    curr_pos = fout.tell()
    fout.seek(4 + 4 + 8)
    _write_number(fout, "u64", num_metadata_out)
    fout.seek(curr_pos)

    offset = 0
    alignment = 32

    for k, v in state_dict.items():
        _write_str(fout, k)
        _write_number(fout, "u32", v.ndim)

        if v.ndim == 1:
            _write_number(fout, "u64", v.shape[0])
            _write_number(fout, "u32", 0)  # GGML_TYPE_F32
        elif v.ndim == 2:
            # get original shape from Q4_0
            shape = list(v.shape)
            shape[-1] = shape[-1] * 32 // (2 + 16)
            for dim in shape[::-1]:  # reversed order
                _write_number(fout, "u64", dim)
            _write_number(fout, "u32", 2)  # GGML_TYPE_Q4_0
        else:
            raise NotImplementedError

        _write_number(fout, "u64", offset)
        offset = (offset + v.nbytes + alignment - 1) // alignment * alignment

    curr_pos = fout.tell()
    pad_amt = (curr_pos + alignment - 1) // alignment * alignment - curr_pos
    fout.write(b"\0" * pad_amt)

    for k, v in state_dict.items():
        fout.write(v.numpy().tobytes())
        pad_amt = (v.nbytes + alignment - 1) // alignment * alignment - v.nbytes
        fout.write(b"\0" * pad_amt)


if __name__ == "__main__":
    main()
