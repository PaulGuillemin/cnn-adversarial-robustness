"""Extract the recorded evaluation checkpoints and plot the final accuracies."""

import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = json.loads((ROOT / "results/archived_run.json").read_text())
    source = ROOT / metadata["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("The archived notebook differs from the recorded source")
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cell = notebook["cells"][metadata["source_cell_index"]]
    text = "\n".join("".join(o.get("text", [])) for o in cell["outputs"]
                     if o.get("output_type") == "stream")
    values = re.findall(
        r"test acc on (clean examples|FGM adversarial examples|PGD adversarial examples) \(%\): ([0-9.]+)", text)
    if len(values) != 30:
        raise ValueError("Expected ten cumulative checkpoints with three accuracies each")
    rows = []
    for i in range(0, len(values), 3):
        if [v[0] for v in values[i:i + 3]] != ["clean examples", "FGM adversarial examples", "PGD adversarial examples"]:
            raise ValueError("Unexpected metric order")
        rows.append({"evaluated_images": (i // 3 + 1) * 1000,
                     "clean_accuracy_percent": float(values[i][1]),
                     "fgm_accuracy_percent": float(values[i + 1][1]),
                     "pgd_accuracy_percent": float(values[i + 2][1])})
    final = rows[-1]
    recorded = metadata["final_accuracy_percent"]
    for name in ("clean", "fgm", "pgd"):
        if final[f"{name}_accuracy_percent"] != recorded[name]:
            raise ValueError("Notebook output and summary disagree")
    with (ROOT / "results/archived_accuracy.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    labels = ["Clean", "FGM", "PGD (40 steps)"]
    accuracies = [recorded[name] for name in ("clean", "fgm", "pgd")]
    bars = ax.bar(labels, accuracies, width=0.55, color=["#444444", "#929292", "#c6c6c6"],
                  edgecolor="#333333", linewidth=0.7)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in accuracies], padding=5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CNN robustness on CIFAR-10", loc="left", pad=15)
    ax.set_xlabel(r"Archived run · 10,000 test images · $L_\infty$ budget = 0.05", labelpad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)
    path = ROOT / "docs/media/archived_accuracy.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    print("Verified ten archived checkpoints and regenerated the accuracy figure.")


if __name__ == "__main__":
    main()
