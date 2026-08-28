"""CHE free energy staircase — one panel per catalyst.

Reads per-step ΔG data saved by eval_che.py (includes ZPE/TS corrections).
Three potentials are plotted: U=0, U=U_ideal, U=U_onset.

Reads:  eval_che_state.json
Output: plot_che.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REACTION_COLORS = {
    "OER":   "#d6604d",
    "ORR":   "#762a83",
    "HER":   "#4393c3",
    "CO2RR": "#4dac26",
    "NRR":   "#e08214",
}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

with open("eval_che_state.json") as f:
    state = json.load(f)

mlip_model = state["mlip_model"]
results    = state["results"]

# Verify the state file has per-step data
if results and "step_energies" not in results[0]:
    raise RuntimeError(
        "eval_che_state.json does not contain per-step ΔG data. "
        "Re-run eval_che.py first (step_energies field is now saved)."
    )

# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def draw_staircase(
    ax,
    labels: list[str],
    dg_steps_by_U: dict[str, list[float]],
    title: str,
    limiting_pair: tuple[str, str],
    rxn: str,
) -> None:
    color   = REACTION_COLORS.get(rxn, "#555555")
    n       = len(labels)
    x       = np.arange(n, dtype=float)
    shelf_w = 0.30

    U_styles = {
        "U = 0 V":         dict(color="#aaaaaa", lw=1.5, ls="--", alpha=0.85),
        "U = U$_{ideal}$": dict(color=color,     lw=2.0, ls="-",  alpha=0.90),
        "U = U$_{onset}$": dict(color="#222222", lw=2.3, ls="-",  alpha=1.00),
    }

    for u_label, dg_steps in dg_steps_by_U.items():
        style = U_styles[u_label]
        cumg  = np.concatenate(([0.0], np.cumsum(dg_steps)))
        for i in range(n):
            ax.plot([x[i] - shelf_w, x[i] + shelf_w], [cumg[i], cumg[i]],
                    zorder=3, **style)
        for i in range(n - 1):
            ax.plot([x[i] + shelf_w, x[i+1] - shelf_w], [cumg[i], cumg[i+1]],
                    color=style["color"], lw=0.7, ls=":",
                    alpha=style["alpha"] * 0.5, zorder=2)
        ax.plot([], [], label=u_label, **style)

    # Highlight limiting step at U_ideal
    ideal_key = "U = U$_{ideal}$"
    if ideal_key in dg_steps_by_U:
        dg_ideal   = dg_steps_by_U[ideal_key]
        cumg_ideal = np.concatenate(([0.0], np.cumsum(dg_ideal)))
        for i, (par, prod) in enumerate(zip(labels[:-1], labels[1:])):
            if (par, prod) == limiting_pair:
                y_lo, y_hi = cumg_ideal[i], cumg_ideal[i + 1]
                ax.annotate(
                    "", xy=(x[i] + shelf_w + 0.04, y_hi),
                    xytext=(x[i] + shelf_w + 0.04, y_lo),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6),
                    zorder=5,
                )
                ax.text(
                    x[i] + shelf_w + 0.12, (y_lo + y_hi) / 2,
                    f"ΔG = {y_hi - y_lo:+.2f} eV\n(limiting)",
                    ha="left", va="center", fontsize=7.5,
                    color="#c0392b", fontweight="bold", zorder=5,
                )
                break

    ax.axhline(0, color="#888888", lw=0.7, ls="--", alpha=0.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cumulative ΔG (eV)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.65, n - 0.35)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

n_cats = len(results)
fig, axes = plt.subplots(1, n_cats, figsize=(6.5 * n_cats, 5.5))
if n_cats == 1:
    axes = [axes]
fig.patch.set_facecolor("white")

for ax, row in zip(axes, results):
    catalyst_key  = row["catalyst"]
    rxn           = row["reaction"]
    u_ideal       = row["U_ideal"]
    u_onset       = row["U_onset"]
    labels        = row["best_pathway"]
    step_data     = row["step_energies"]   # list of dicts with delta_e/zpe/ts/g/n_electrons

    # Sign: anodic reactions use +U, cathodic use −U
    # Infer from majority n_electrons sign in path steps
    ne_sum = sum(s["n_electrons"] for s in step_data)
    u_sign = 1 if ne_sum > 0 else -1

    limiting_pair = tuple(row["limiting_step"].split(" → ", 1))

    # Compute cumulative ΔG at three potentials from saved ΔE/ZPE/TS per step.
    # ΔG(U) = ΔE + ΔZPE − ΔTS − n_e · U   (same formula as che.py)
    def steps_at_U(U_val):
        return [
            s["delta_e"] + s["delta_zpe"] - s["delta_ts"] - s["n_electrons"] * U_val
            for s in step_data
        ]

    dg_steps_by_U = {
        "U = 0 V":         steps_at_U(0.0),
        "U = U$_{ideal}$": steps_at_U(u_sign * u_ideal),
        "U = U$_{onset}$": steps_at_U(u_sign * u_onset),
    }

    title = (
        f"{catalyst_key} — {rxn}\n"
        f"η = {row['overpotential']:.3f} eV  ·  "
        f"U$_{{onset}}$ = {u_sign * u_onset:+.3f} V  ·  "
        f"U$_{{ideal}}$ = {u_sign * u_ideal:+.2f} V"
    )
    draw_staircase(ax, labels, dg_steps_by_U, title, limiting_pair, rxn)

fig.suptitle(
    f"CHE free energy diagram  ·  {mlip_model}  ·  ΔE + ZPE − TΔS",
    fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout()
out = "plot_che.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
