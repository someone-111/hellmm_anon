"""Adsorption energy bar chart — one panel per catalyst.

Reads:  eval_adsorption_energy_state.json
Output: plot_adsorption_energy.png
"""

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CANONICAL = {
    "OER":   {"*OH", "*O", "*OOH"},
    "ORR":   {"*OOH", "*O", "*OH"},
    "HER":   {"*H"},
    "CO2RR": {"*CO2", "*COOH", "*CO2H", "*CO"},
    "NRR":   {"*N2", "*NNH", "*NH", "*NH2", "*NH3"},
}

REACTION_COLORS = {
    "OER":   "#d6604d",
    "ORR":   "#762a83",
    "HER":   "#4393c3",
    "CO2RR": "#4dac26",
    "NRR":   "#e08214",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open("eval_adsorption_energy_state.json") as f:
    state = json.load(f)

mlip_model  = state["mlip_model"]
all_results = state["results"]   # dict: catalyst_key → {context, energy_results, ...}
n_cats      = len(all_results)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5.5), sharey=False)
if n_cats == 1:
    axes = [axes]
fig.patch.set_facecolor("white")

for ax, (cat_key, cat_data) in zip(axes, all_results.items()):
    rxn       = cat_data["context"]["reaction"]
    energies  = cat_data["energy_results"]      # label → float (eV)
    canonical = CANONICAL.get(rxn, set())
    color_hi  = REACTION_COLORS.get(rxn, "#555555")

    # Canonical first, then others alphabetically; skip bare surface *
    labels_can   = [l for l in energies if l in canonical and l != "*"]
    labels_other = sorted(l for l in energies if l not in canonical and l != "*")
    labels       = labels_can + labels_other
    values       = [energies[l] for l in labels]
    colors       = [color_hi if l in canonical else "#cccccc" for l in labels]

    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--", alpha=0.5, zorder=2)

    # Annotate canonical bars only
    for xi, (lbl, val) in enumerate(zip(labels, values)):
        if lbl in canonical:
            va     = "bottom" if val >= 0 else "top"
            offset = 0.04 if val >= 0 else -0.04
            ax.text(xi, val + offset, f"{val:+.2f}", ha="center", va=va,
                    fontsize=7, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ΔE$_{ads}$ (eV)", fontsize=10)
    ax.set_title(f"{cat_key} — {rxn}", fontsize=11, fontweight="bold", pad=6)
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=color_hi,  edgecolor="#666", label="Canonical pathway"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#666", label="Other kept intermediates"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best", framealpha=0.85)

fig.suptitle(
    f"Adsorption energies  ·  {mlip_model}",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
out = "plot_adsorption_energy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
