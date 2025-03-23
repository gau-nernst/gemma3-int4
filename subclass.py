import torch
import torch.nn.functional as F
from torch import Tensor


def _dequantize(int4_data: Tensor, scales: Tensor):
    # int4_data: (..., N / 2), in int8
    # scales: (..., N / block_size)
    int8_data = torch.stack([int4_data << 4 >> 4, int4_data >> 4], dim=-1)
    fp32_data = int8_data.float().view(*scales.shape, -1) * scales.unsqueeze(-1)
    return fp32_data.flatten(-2).to(scales.dtype)


class Int4Tensor(Tensor):
    @staticmethod
    def __new__(cls, int4_data: Tensor, scales: Tensor):
        shape = int4_data.shape
        return Tensor._make_wrapper_subclass(
            cls,
            shape[:-1] + (shape[-1] * 2,),
            dtype=scales.dtype,
            device=scales.device,
        )

    def __init__(self, int4_data: Tensor, scales: Tensor) -> None:
        self.int4_data = int4_data
        self.scales = scales

    def __tensor_flatten__(self):
        return ["int4_data", "scales"], []

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*tensor_data_dict.values(), *tensor_attributes)

    def __repr__(self):
        fields = dict(
            shape=tuple(self.shape),
            block_size=self.int4_data.shape[-1] * 2 // self.scales.shape[-1],
            device=self.device,
        )
        fields_str = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fields_str})"

    def dequantize(self):
        return _dequantize(self.int4_data, self.scales)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            x: Tensor = args[0]
            w: Int4Tensor = args[1]
            b: Tensor | None = args[2] if len(args) > 2 else None
            return F.linear(x, w.dequantize(), b)

        elif func is F.embedding:
            input: Tensor = args[0]
            weight: Int4Tensor = args[1]
            return _dequantize(F.embedding(input, weight.int4_data), F.embedding(input, weight.scales))

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        aten = torch.ops.aten

        if func is aten.detach.default:
            x: Int4Tensor = args[0]
            return Int4Tensor(x.int4_data, x.scales)

        msg = f"{cls.__name__} dispatch: {func} is not implemented"
        for i, arg in enumerate(args):
            msg += f"\n- args[{i}]={arg}"
        for k, v in kwargs.items():
            msg += f"\n- {k}={v}"
        raise NotImplementedError(msg)


def to_plain(state_dict: dict[str, Tensor]):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if isinstance(v, Int4Tensor):
            subkeys, _ = v.__tensor_flatten__()
            for subkey in subkeys:
                new_state_dict[f"{k}.{subkey}"] = getattr(v, subkey)
        else:
            assert type(v) == Tensor
            new_state_dict[k] = v
    return new_state_dict


def from_plain(state_dict: dict[str, Tensor]):
    state_dict = dict(state_dict)  # shallow clone
    for k in list(state_dict.keys()):
        if k.endswith(".int4_data"):
            prefix = k.removesuffix(".int4_data")
            int4_data = state_dict.pop(k)
            scales = state_dict.pop(f"{prefix}.scales")
            state_dict[prefix] = Int4Tensor(int4_data, scales)
    return state_dict
