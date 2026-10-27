"""Write a deterministic 3DGS fixture: python synthetic_ply.py <output.ply>."""

import itertools
import math
from pathlib import Path
import sys


def write_fixture(path):
    names = ["x", "y", "z", *[f"f_dc_{i}" for i in range(3)],
             *[f"f_rest_{i}" for i in range(45)], "opacity",
             *[f"scale_{i}" for i in range(3)], *[f"rot_{i}" for i in range(4)]]
    rows = []
    for x, y, z in itertools.product(range(-2, 3), repeat=3):
        rows.append([0.7 * x, 0.7 * y, 0.7 * z,
                     0.2 * x, 0.2 * y, 0.2 * z, *([0.0] * 45),
                     math.log(9.0), *([math.log(0.15)] * 3), 1.0, 0.0, 0.0, 0.0])
    header = ("ply\nformat ascii 1.0\nelement vertex 125\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n")
    Path(path).write_text(header + "\n".join(" ".join(map(str, row)) for row in rows) + "\n")


if __name__ == "__main__":
    write_fixture(sys.argv[1])
