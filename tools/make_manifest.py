#!/usr/bin/env python
"""Create a reduced manifest by keeping every N-th line.

Ported from the research utility that produced the reduced-data manifests.

    python tools/make_manifest.py data/DIFFall_train.txt --interval 5

**The interval is a line stride, not a percentage.** Because the research
manifests were built with ``if i % interval == 0``, a file named ``sub5`` holds
about 20% of the lines and ``sub20`` about 5%. The manuscript flags the affected
low-data rows as unreconciled for exactly this reason, so the naming trap is
preserved here with the semantics made explicit.
"""

from __future__ import annotations

import argparse

from _common import read_manifest_lines


def subsample(lines: list[str], interval: int) -> list[str]:
    if interval < 1:
        raise ValueError("interval must be >= 1")
    return [line for index, line in enumerate(lines) if index % interval == 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest")
    parser.add_argument("--interval", type=int, default=5,
                        help="keep every N-th line (N is a line stride, not a percentage)")
    parser.add_argument("--out", default=None,
                        help="output path (default: <manifest stem>_sub<interval>.txt)")
    parser.add_argument("--stats", action="store_true", help="print the resulting proportion")
    args = parser.parse_args()

    lines = read_manifest_lines(args.manifest)
    kept = subsample(lines, args.interval)

    out = args.out
    if out is None:
        stem, _, _ = args.manifest.rpartition(".")
        out = f"{stem}_sub{args.interval}.txt"
    with open(out, "w", encoding="utf-8") as handle:
        handle.writelines(kept)

    proportion = len(kept) / max(len(lines), 1)
    print(f"{args.manifest}: {len(lines)} lines -> {out}: {len(kept)} lines ({proportion:.3%})")
    if args.stats:
        print(f"  interval {args.interval} keeps approximately {100 / args.interval:.2f}% of the lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
