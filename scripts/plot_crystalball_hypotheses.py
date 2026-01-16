#!/usr/bin/env python3
"""Plot Crystal Ball distributions for multiple (alpha, n) hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import crystalball


@dataclass(frozen=True)
class Hypothesis:
    alpha: float
    n: float

    @property
    def label(self) -> str:
        return rf"$\alpha={self.alpha},\,n={self.n}$"


DEFAULT_HYPOTHESES = [
    Hypothesis(alpha=1.0, n=2.0),
    Hypothesis(alpha=1.5, n=3.0),
    Hypothesis(alpha=2.0, n=5.0),
    Hypothesis(alpha=2.5, n=10.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Crystal Ball distributions for multiple hypotheses."
    )
    parser.add_argument(
        "--resolution-sigma",
        type=float,
        default=4.0,
        help="Gaussian core sigma (GeV).",
    )
    parser.add_argument(
        "--n-mc-total",
        type=int,
        default=10_000_000,
        help="Total Monte Carlo events per hypothesis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=250,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=(-30.0, 30.0),
        metavar=("MIN", "MAX"),
        help="Histogram range (GeV).",
    )
    parser.add_argument(
        "--hypothesis",
        action="append",
        default=[],
        help="Hypothesis in 'alpha,n' format (can be repeated).",
    )
    parser.add_argument(
        "--output",
        default="crystalball_hypotheses.png",
        help="Output image filename.",
    )
    return parser.parse_args()


def parse_hypotheses(values: list[str]) -> list[Hypothesis]:
    if not values:
        return DEFAULT_HYPOTHESES
    hypotheses: list[Hypothesis] = []
    for item in values:
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid hypothesis '{item}'. Use 'alpha,n'.")
        alpha, n = (float(part) for part in parts)
        hypotheses.append(Hypothesis(alpha=alpha, n=n))
    return hypotheses


def main() -> None:
    args = parse_args()
    hypotheses = parse_hypotheses(args.hypothesis)
    rng = np.random.default_rng(args.seed)

    fig, ax = plt.subplots(figsize=(9, 6))
    for hypothesis in hypotheses:
        smear = crystalball.rvs(
            beta=hypothesis.alpha,
            m=hypothesis.n,
            loc=0.0,
            scale=args.resolution_sigma,
            size=args.n_mc_total,
            random_state=rng,
        )
        ax.hist(
            smear,
            bins=args.bins,
            range=args.range,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=hypothesis.label,
        )

    ax.set_title(
        "Crystal Ball smearing hypotheses",
        fontsize=13,
    )
    ax.set_xlabel("Smear [GeV]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
