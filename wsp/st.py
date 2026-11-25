"""Read and write .safetensors files with numpy only.

numpy has no bfloat16 type, and the models here are stored in it, so a bfloat16 tensor is handled as its raw 16-bit
patterns ("bits", uint16) and converted to float32 exactly when needed. A bfloat16 value is the top 16 bits of a
float32, so bits -> float32 is exact; float32 -> bits rounds to nearest, ties to even.

    f = File(path)            # memory-mapped, read-only
    f.names                   # tensor names, in file order
    f.info(name)              # (dtype string such as "BF16", shape)
    f.bits(name)              # the stored array (uint16 bit patterns for BF16), a read-only view
    f.f32(name)               # a float32 copy
    patch(base, out, {name: float32 array})   # copy `base` to `out` with these tensors replaced
"""
import json
import shutil
import struct

import numpy as np

NP = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
      "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "?"}


def bf16_to_f32(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    x = np.ascontiguousarray(x, np.float32)
    assert np.isfinite(x).all(), "non-finite value"
    b = x.view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


class File:
    def __init__(self, path, mode="r"):
        self.path = path
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            head = json.loads(fh.read(n))
        self.meta = head.pop("__metadata__", {})
        self.head = head
        self.base = 8 + n
        self.names = sorted(head, key=lambda k: head[k]["data_offsets"][0])
        self.map = np.memmap(path, dtype=np.uint8, mode=mode)

    def info(self, name):
        t = self.head[name]
        return t["dtype"], tuple(t["shape"])

    def bits(self, name):
        t = self.head[name]
        a, b = t["data_offsets"]
        return self.map[self.base + a:self.base + b].view(NP[t["dtype"]]).reshape(t["shape"])

    def f32(self, name):
        dt, _ = self.info(name)
        x = self.bits(name)
        return bf16_to_f32(x) if dt == "BF16" else np.asarray(x, np.float32)


class Dir:
    """The tensors of a model folder, whichever of its .safetensors files each one is in."""

    def __init__(self, model_dir):
        import glob
        import os
        self.files = [File(p) for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))]
        assert self.files, f"no .safetensors file in {model_dir}"
        self.of = {n: f for f in self.files for n in f.names}
        self.names = [n for f in self.files for n in f.names]

    def info(self, name):
        return self.of[name].info(name)

    def bits(self, name):
        return self.of[name].bits(name)

    def f32(self, name):
        return self.of[name].f32(name)


def patch(base_path, out_path, tensors):
    """Copy base_path to out_path, then overwrite the named tensors (float32 arrays, same shapes) in place.
    Returns {name: number of stored values that changed}."""
    shutil.copyfile(base_path, out_path)
    f = File(out_path, mode="r+")
    changed = {}
    for name, x in tensors.items():
        dt, shape = f.info(name)
        assert tuple(np.shape(x)) == shape, (name, np.shape(x), shape)
        new = f32_to_bf16(x) if dt == "BF16" else np.asarray(x, NP[dt])
        old = f.bits(name)
        changed[name] = int((old != new).sum())
        old[...] = new
    f.map.flush()
    return changed
