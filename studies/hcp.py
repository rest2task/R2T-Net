from __future__ import annotations

import argparse
from pathlib import Path

from .plan import manifest_cmd, train_cmd
from .specs import REPRESENTATIONS, TRAINING_MODES


def parse_args():
    p = argparse.ArgumentParser(description="HCP commands")
    p.add_argument("--mode", choices=sorted(TRAINING_MODES))
    p.add_argument("--representation", choices=[*sorted(REPRESENTATIONS), "all"], default="grayord")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--participants", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--manifest-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(manifest_cmd("hcp", args.data_dir, args.source_root, args.participants))
    if args.manifest_only:
        return
    # `--representation all` fans out across every defined representation; otherwise just one.
    reps = sorted(REPRESENTATIONS) if args.representation == "all" else [args.representation]
    # Default is a full four-mode sweep unless caller fixes a single mode.
    modes = [args.mode] if args.mode else ["rs_only", "t_only", "synthetic_t", "r2t"]
    for rep in reps:
        for mode in modes:
            print(train_cmd("hcp", mode, rep, args.data_dir, args.out_dir))


if __name__ == "__main__":
    main()
