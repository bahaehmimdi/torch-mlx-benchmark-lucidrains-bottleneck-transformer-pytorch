"""Test lucidrains/bottleneck-transformer-pytorch BottleStack with torch-mlx.

Byte-for-byte load of the upstream module. The unique feature is the
*transposed/bottleneck* attention: q,k,v projected to a small bottleneck dim
and attention computed as einsum over 4D/5D tensors — stresses whether
torch-mlx supports multi-operand einsum + einops rearrange on mlx tensors.
"""
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))                 # so `import bottleneck_transformer_pytorch` resolves
sys.path.insert(0, str(P.parent.parent))   # torch-mlx root

from bottleneck_transformer_pytorch.bottleneck_transformer_pytorch import BottleStack, BottleBlock
import torch
import torch.nn as nn

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")
    return cond


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    print("=" * 60)
    print("lucidrains/bottleneck-transformer-pytorch — BottleStack (BiT)")
    print("=" * 60)

    stack = BottleStack(
        dim=32, fmap_size=8, dim_out=64,
        proj_factor=4, num_layers=2, heads=4, dim_head=8,
        downsample=True, rel_pos_emb=False, activation=nn.ReLU(),
    )
    n = count_params(stack)
    check("BottleStack built", n > 0)
    print(f"    BottleStack: {n:,} params")

    # input (B, 32, 8, 8) -> after downsample block: (B, 64, 4, 4)
    x = torch.randn(2, 32, 8, 8)
    out = stack(x)
    check("output shape (downsample block)", tuple(out.shape) == (2, 64, 4, 4))
    check("output finite", bool(torch.isfinite(out).all().item()))

    # relative position embedding variant
    stack_rel = BottleStack(
        dim=16, fmap_size=6, dim_out=16,
        proj_factor=2, num_layers=1, heads=2, dim_head=8,
        downsample=False, rel_pos_emb=True, activation=nn.ReLU(),
    )
    x2 = torch.randn(2, 16, 6, 6)
    out2 = stack_rel(x2)
    check("rel_pos_emb output shape", tuple(out2.shape) == (2, 16, 6, 6))
    check("rel_pos_emb finite", bool(torch.isfinite(out2).all().item()))

    # backward
    loss = out.sum()
    loss.backward()
    grads = sum(1 for p in stack.parameters() if p.grad is not None)
    total = sum(1 for p in stack.parameters())
    check(f"backward {grads}/{total} grads", grads == total)

    opt = torch.optim.Adam(stack.parameters(), lr=1e-3)
    opt.step()
    check("Adam step ran", True)

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
