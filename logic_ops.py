import torch

def all_logic_ops(a, b):
    ops = torch.stack([
        torch.zeros_like(a),         # 0: FALSE
        a * b,                       # 1: AND
        a - a * b,                   # 2: not(A => B)
        a,                           # 3: A
        b - a * b,                   # 4: not(B => A)
        b,                           # 5: B
        a + b - 2 * a * b,           # 6: XOR
        a + b - a * b,               # 7: OR
        1 - (a + b - a * b),         # 8: NOR
        1 - (a + b - 2 * a * b),     # 9: XNOR
        1 - b,                       # 10: NOT B
        1 - b + a * b,               # 11: B => A
        1 - a,                       # 12: NOT A
        1 - a + a * b,               # 13: A => B
        1 - a * b,                   # 14: NAND
        torch.ones_like(a),         # 15: TRUE
    ], dim=1)
    return ops

