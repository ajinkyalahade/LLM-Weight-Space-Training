"""Which tensors differ between two checkpoints, and how.

    python -I wsp/diff.py <base .safetensors> <other .safetensors>

For each tensor that differs: how many stored values changed, which columns and rows, the size of the change relative
to the tensor, and whether the change is rank one (one direction subtracted from the changed columns).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st  # noqa: E402


def main(pa, pb):
    a, b = st.File(pa), st.File(pb)
    only = sorted(set(a.names) ^ set(b.names))
    print(f"tensors: {len(a.names)} and {len(b.names)}; in only one file: {only[:5] or 'none'}")
    total, kinds, differ = 0, {}, []
    for k in a.names:
        if k not in b.head:
            continue
        x, y = a.bits(k), b.bits(k)
        total += x.size
        kinds[a.info(k)[0]] = kinds.get(a.info(k)[0], 0) + x.size
        if a.info(k) != b.info(k) or not np.array_equal(x, y):
            differ.append(k)
    print(f"values: {total / 1e6:.1f}M; by stored type: " + ", ".join(f"{d} {n / 1e6:.1f}M" for d, n in kinds.items()))
    print(f"tensors that differ: {len(differ)}")
    for k in differ:
        x, y = a.f32(k), b.f32(k)
        d = y - x
        print(f"\n{k}\n  shape {x.shape}, stored as {a.info(k)[0]}; values changed: {int((d != 0).sum()):,}")
        if d.ndim != 2:
            continue
        cols, rows = np.where(np.abs(d).sum(0) > 0)[0], np.where(np.abs(d).sum(1) > 0)[0]
        s = np.linalg.svd(d[:, cols], compute_uv=False)
        print(f"  columns changed: {len(cols)} of {x.shape[1]}; rows changed: {len(rows)} of {x.shape[0]}")
        print(f"  columns: {cols.tolist()}")
        print(f"  size of change / size of tensor: {np.linalg.norm(d) / np.linalg.norm(x):.4f}; "
              f"/ size of the changed columns: {np.linalg.norm(d[:, cols]) / np.linalg.norm(x[:, cols]):.4f}")
        print(f"  top singular values of the change: {np.round(s[:4], 4).tolist()}; "
              f"share in the first: {s[0] ** 2 / (s ** 2).sum():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
