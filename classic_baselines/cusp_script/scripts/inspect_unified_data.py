# -*- coding: utf-8 -*-
"""Inspect Table2Graph unified_data.pt: label histogram and class count."""
import argparse
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.table2graph_dataset import TABLE2GRAPH_FOLDER, _coerce_data, DEFAULT_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to unified_data.pt; or omit and use --dataset to resolve under exptable2graph",
    )
    p.add_argument("--dataset", type=str, default=None, choices=list(TABLE2GRAPH_FOLDER.keys()))
    args = p.parse_args()

    if args.path:
        pt = args.path
    elif args.dataset:
        sub = TABLE2GRAPH_FOLDER[args.dataset]
        pt = os.path.join(DEFAULT_ROOT, sub, "unified_data.pt")
    else:
        print("Pass unified_data.pt path or --dataset Hockey")
        sys.exit(1)

    if not os.path.isfile(pt):
        print("File not found:", pt)
        sys.exit(1)

    raw = torch.load(pt, map_location="cpu", weights_only=False)
    data = _coerce_data(raw)
    y = data.y.view(-1).long()
    n = y.numel()
    u, cnt = torch.unique(y, return_counts=True)
    print("file:", os.path.normpath(pt))
    print("num_nodes:", n)
    print("num_classes (max+1):", int(y.max().item()) + 1)
    print("label -> count:")
    for lab, c in zip(u.tolist(), cnt.tolist()):
        pct = 100.0 * c / n
        print("  %s : %d (%.2f%%)" % (lab, c, pct))
    if len(cnt) > 1:
        imb = float(cnt.max() / cnt.min())
        print("max_count / min_count (imbalance hint): %.2f" % imb)
    print("\nNote: CUSP reads data.y only; alignment with your table->graph script is your responsibility.")
    print("Few samples per class -> high Acc but low macro-F1 is common.")


if __name__ == "__main__":
    main()
