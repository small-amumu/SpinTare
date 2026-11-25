# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Paired bootstrap of the accuracy difference between two lc_reason_vllm.py result files (same
items).

Usage: python reason_ci.py <base.json> <test.json>   -> one JSON line
"""

import json
import random
import sys


def main() -> None:
    a = {r["i"]: r for r in json.load(open(sys.argv[1]))["records"]}
    b = {r["i"]: r for r in json.load(open(sys.argv[2]))["records"]}
    items = sorted(set(a) & set(b))
    d = [int(b[i]["ok"]) - int(a[i]["ok"]) for i in items]
    n = len(d)
    rng = random.Random(0)
    boots = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    gen_a = sum(a[i]["n_gen"] for i in items) / n
    gen_b = sum(b[i]["n_gen"] for i in items) / n
    print(
        json.dumps(
            {
                "n": n,
                "acc_base": round(100 * sum(a[i]["ok"] for i in items) / n, 2),
                "acc_test": round(100 * sum(b[i]["ok"] for i in items) / n, 2),
                "diff": round(100 * sum(d) / n, 2),
                "ci95": [round(boots[249], 2), round(boots[9749], 2)],
                "gen_ratio": round(gen_b / gen_a, 2),
                "cap_share_test": round(100 * sum(b[i]["truncated"] for i in items) / n, 1),
                "loop_share_test": round(100 * sum(b[i]["loop"] for i in items) / n, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
