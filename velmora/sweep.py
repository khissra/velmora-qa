"""Phase 3 bis - Joint sweep of tau (coverage threshold) and rho (sentence ratio).

Runs the assistant on every (tau, rho) pair of a grid, then writes:
  - sweep_grid.csv     one row per (tau, rho) with all counts and rates
  - sweep_heatmaps.png 4 heatmaps over the tau x rho grid
  - sweep_curves.png   status counts along tau (rho fixed) and along rho (tau fixed)

Usage (from the folder that contains the velmora package):
    python -m velmora.sweep --kb ai/kb --eval ai/eval.jsonl --method embeddings --rerank cross
    python -m velmora.sweep --kb ai/kb --eval ai/eval.jsonl --method bm25 --taus 0.1 0.2 0.3 0.4 0.5 0.6 0.7

Needs matplotlib and numpy:  pip install matplotlib numpy
Note: the 40 questions are used to draw these plots, so every number is in-sample.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from velmora.answer import DEFAULTS, Assistant
from velmora.evaluate import evaluate_question, load_questions, mean
from velmora.ingest import DEFAULT_MODEL, build_knowledge_base
from velmora.rerank import make_reranker

STATUSES = ["good_answer", "incomplete_answer", "false_refusal",
            "correct_refusal", "false_answer"]
COLORS = {
    "good_answer": "#2e9e5b",
    "incomplete_answer": "#f0a030",
    "false_refusal": "#d9534f",
    "correct_refusal": "#4a90c2",
    "false_answer": "#6a1b1b",
}

DEFAULT_GRIDS = {
    "embeddings": [round(0.30 + 0.05 * i, 2) for i in range(10)],  # 0.30 .. 0.75
    "bm25": [round(0.10 * i, 2) for i in range(1, 10)],            # 0.10 .. 0.90
}
DEFAULT_RHOS = [round(0.2 + 0.1 * i, 2) for i in range(8)]         # 0.2 .. 0.9


# ---------- run ----------

class CachedReranker:
    """Same scores for the same (question, texts): the grid reuses them."""

    def __init__(self, inner):
        self.inner = inner
        self.cache = {}

    def scores(self, question, texts):
        key = (question, tuple(texts))
        if key not in self.cache:
            self.cache[key] = self.inner.scores(question, texts)
        return self.cache[key]


def run_grid(kb, questions, method, reranker, taus, rhos):
    grid = {}
    for tau in taus:
        for rho in rhos:
            assistant = Assistant(kb, tau=tau, rho=rho, method=method, reranker=reranker)
            rows = [evaluate_question(assistant, item) for item in questions]
            counts = Counter(r["status"] for r in rows)
            answerable = [r for r in rows if r["answerable"]]
            unanswerable = [r for r in rows if not r["answerable"]]
            answered_ok = [r for r in answerable if r["answered"]]
            grid[(tau, rho)] = {
                "tau": tau,
                "rho": rho,
                **{s: counts[s] for s in STATUSES},
                "correct": counts["good_answer"] + counts["correct_refusal"],
                "answer_rate": mean([r["answered"] for r in answerable]),
                "refusal_rate": mean([not r["answered"] for r in unanswerable]),
                "keyword_score": mean([r["keyword_score"] for r in answered_ok]),
            }
        c = grid[(tau, rhos[0])]
        print(f"tau={tau:<5} done (rho={rhos[0]}: correct={c['correct']}/{len(questions)})")
    return grid


def save_csv(grid, path):
    rows = list(grid.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------- analysis ----------

def matrix(grid, taus, rhos, key):
    """rows = rho, columns = tau."""
    return np.array([[grid[(t, r)][key] for t in taus] for r in rhos], dtype=float)


def nearest(values, target):
    return int(np.argmin(np.abs(np.array(values) - target)))


def robust_best(mat):
    """Cell with the best mean over its 3x3 neighbourhood: the centre of a plateau,
    not an isolated peak (less sensitive to the 40 questions)."""
    padded = np.pad(mat, 1, constant_values=np.nan)
    smooth = np.empty_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            smooth[i, j] = np.nanmean(padded[i:i + 3, j:j + 3])
    i, j = np.unravel_index(np.nanargmax(smooth), smooth.shape)
    return int(i), int(j)


# ---------- plots ----------

def heatmap(ax, taus, rhos, mat, title, cmap, default_ij, robust_ij):
    im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:g}" for t in taus], fontsize=8)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f"{r:g}" for r in rhos], fontsize=8)
    ax.set_xlabel("tau (threshold)")
    ax.set_ylabel("rho (sentence ratio)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    lo, hi = np.nanmin(mat), np.nanmax(mat)
    span = (hi - lo) or 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            dark = (mat[i, j] - lo) / span > 0.6
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "black")

    di, dj = default_ij
    ax.add_patch(Rectangle((dj - 0.5, di - 0.5), 1, 1, fill=False,
                           edgecolor="black", linewidth=2.2, linestyle="--"))
    ri, rj = robust_ij
    ax.plot(rj + 0.3, ri + 0.3, marker="*", markersize=13,
            markerfacecolor="gold", markeredgecolor="black", linestyle="none")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def plot_heatmaps(grid, taus, rhos, n_questions, default_ij, robust_ij, path):
    panels = [
        ("correct", f"Correct answers (/{n_questions})", "YlGn"),
        ("false_refusal", "False refusals (answerable, refused)", "Reds"),
        ("false_answer", "False answers (unanswerable, answered)", "Reds"),
        ("incomplete_answer", "Incomplete answers", "Oranges"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, cmap) in zip(axes.ravel(), panels):
        heatmap(ax, taus, rhos, matrix(grid, taus, rhos, key), title, cmap,
                default_ij, robust_ij)
    fig.suptitle("tau x rho sweep  |  dashed square = current default, "
                 "star = most stable region (best 3x3 mean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stacked(ax, xs, series, xlabel, title, n_questions, marker_x, marker_label):
    ax.stackplot(xs, *[series[s] for s in STATUSES],
                 labels=STATUSES, colors=[COLORS[s] for s in STATUSES], alpha=0.9)
    ax.axvline(marker_x, color="black", linestyle="--", linewidth=1.5, label=marker_label)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0, n_questions)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of questions")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)


def plot_curves(grid, taus, rhos, n_questions, default_tau, default_rho, path):
    rho_fixed = rhos[nearest(rhos, default_rho)]
    tau_fixed = taus[nearest(taus, default_tau)]

    along_tau = {s: [grid[(t, rho_fixed)][s] for t in taus] for s in STATUSES}
    along_rho = {s: [grid[(tau_fixed, r)][s] for r in rhos] for s in STATUSES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    stacked(ax1, taus, along_tau, "tau (threshold)",
            f"Behaviour along tau (rho = {rho_fixed:g})", n_questions,
            tau_fixed, f"default tau = {tau_fixed:g}")
    stacked(ax2, rhos, along_rho, "rho (sentence ratio)",
            f"Behaviour along rho (tau = {tau_fixed:g})", n_questions,
            rho_fixed, f"default rho = {rho_fixed:g}")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------- report ----------

def print_summary(grid, taus, rhos, n_questions, default_ij, robust_ij):
    cells = sorted(grid.values(), key=lambda c: (c["correct"], c["answer_rate"]), reverse=True)
    best = cells[0]["correct"]
    ties = sum(c["correct"] == best for c in cells)

    print(f"\nBest raw score: {best}/{n_questions} reached by {ties} of {len(cells)} cells")
    print("Top 5 cells (tau, rho):")
    for c in cells[:5]:
        print(f"  tau={c['tau']:<5} rho={c['rho']:<4} correct={c['correct']}  "
              f"false_refusal={c['false_refusal']}  false_answer={c['false_answer']}  "
              f"incomplete={c['incomplete_answer']}")

    di, dj = default_ij
    d = grid[(taus[dj], rhos[di])]
    print(f"\nCurrent default (tau={taus[dj]}, rho={rhos[di]}): correct={d['correct']}/{n_questions}")
    ri, rj = robust_ij
    r = grid[(taus[rj], rhos[ri])]
    print(f"Most stable region (tau={taus[rj]}, rho={rhos[ri]}): correct={r['correct']}/{n_questions}")
    print("\nReminder: tuned and measured on the same 40 questions -> in-sample, optimistic.")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Joint tau/rho sweep with plots")
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--eval", default="eval.jsonl")
    parser.add_argument("--method", default="embeddings", choices=["bm25", "embeddings"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank", choices=["cross", "llm", "semantic"])
    parser.add_argument("--taus", type=float, nargs="+", help="tau values (default depends on method)")
    parser.add_argument("--rhos", type=float, nargs="+", help="rho values (default 0.2 .. 0.9)")
    parser.add_argument("--out-dir", default="sweep_out")
    args = parser.parse_args()

    taus = sorted(args.taus) if args.taus else DEFAULT_GRIDS[args.method]
    rhos = sorted(args.rhos) if args.rhos else DEFAULT_RHOS
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kb = build_knowledge_base(
        args.kb, use_embeddings=(args.method == "embeddings"), model_name=args.model
    )
    inner = make_reranker(args.rerank)
    reranker = CachedReranker(inner) if inner else None
    questions = load_questions(args.eval)

    print(f"Sweep: {len(taus)} tau x {len(rhos)} rho = {len(taus) * len(rhos)} runs "
          f"on {len(questions)} questions")
    grid = run_grid(kb, questions, args.method, reranker, taus, rhos)

    default_tau, default_rho = DEFAULTS[args.method]["tau"], DEFAULTS[args.method]["rho"]
    default_ij = (nearest(rhos, default_rho), nearest(taus, default_tau))
    robust_ij = robust_best(matrix(grid, taus, rhos, "correct"))

    save_csv(grid, out / "sweep_grid.csv")
    plot_heatmaps(grid, taus, rhos, len(questions), default_ij, robust_ij, out / "sweep_heatmaps.png")
    plot_curves(grid, taus, rhos, len(questions), default_tau, default_rho, out / "sweep_curves.png")
    print_summary(grid, taus, rhos, len(questions), default_ij, robust_ij)
    print(f"\nFiles written in {out}/: sweep_grid.csv, sweep_heatmaps.png, sweep_curves.png")


if __name__ == "__main__":
    main()
