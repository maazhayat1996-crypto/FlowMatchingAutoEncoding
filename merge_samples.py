"""
Combine all sample sets in RUN_DIR/samples/ (written by trainer.py and
sample_posterior.py) into the files plot_posterior.py reads:

    python merge_samples.py RUN_DIR

Writes RUN_DIR/G_samples.npy, chi2.npy, G_chi2_lt_45.npy, G_chi2_lt_2500.npy and
merged.json, and prints the totals.  Idempotent: it only reads samples/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sample_posterior as SP


def merge(run: Path, log=lambda s: print(s, flush=True)) -> dict:
    d = run / "samples"
    tags = sorted(p.name[:-len("_G.npy")] for p in d.glob("*_G.npy"))
    if not tags:
        raise SystemExit(f"no samples/*_G.npy in {run}")
    Gs, cs, n_drawn, seeds = [], [], 0, {}
    for tag in tags:
        G, c = np.load(d / f"{tag}_G.npy"), np.load(d / f"{tag}_chi2.npy")
        if len(G) != len(c):
            raise SystemExit(f"{tag}: {len(G)} samples but {len(c)} chi2 values")
        meta_path = d / f"{tag}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        n_drawn += meta.get("n_drawn", len(G))
        seed = meta.get("seed")
        if seed is not None and seed in seeds:
            log(f"WARNING: tags {seeds[seed]} and {tag} used the same seed {seed}; "
                f"their samples are identical copies, not independent draws")
        seeds.setdefault(seed, tag)
        log(f"  {tag:>8}: {len(G):>10,} samples, {int(np.sum(c < 45)):>7,} with chi2 < 45")
        Gs.append(G); cs.append(c)
    G, c = np.concatenate(Gs), np.concatenate(cs)
    del Gs, cs
    np.save(run / "G_samples.npy", G)
    np.save(run / "chi2.npy", c)
    np.save(run / "G_chi2_lt_45.npy", G[c < 45])
    np.save(run / "G_chi2_lt_2500.npy", G[c < 2500])
    st = dict(tags=tags, **SP.stats(c, n_drawn=n_drawn))
    (run / "merged.json").write_text(json.dumps(st, indent=2))
    log(f"merged {len(tags)} sets: " + SP.format_stats(st))
    return st


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    merge(p.parse_args().run)


if __name__ == "__main__":
    main()
