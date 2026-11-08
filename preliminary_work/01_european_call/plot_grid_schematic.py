import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Limits
ax.set_xlim(-1.2, 6.2)
ax.set_ylim(-1.0, 4.6)
ax.axis("off")

# Axes with arrows
ax.annotate("", xy=(6.0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.8))
ax.annotate("", xy=(0, 4.4), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.8))

# Axis labels
ax.text(6.05, -0.02, r"$S$", fontsize=18, va="center")
ax.text(0.00, 4.48, r"$t$", fontsize=18, ha="center")

# Grid box
xmax, ymax = 5, 4
ax.plot([0, xmax], [0, 0], ls=(0, (5, 5)), color="0.7")
ax.plot([0, xmax], [ymax, ymax], ls=(0, (5, 5)), color="0.7")
ax.plot([0, 0], [0, ymax], ls=(0, (5, 5)), color="0.7")
ax.plot([xmax, xmax], [0, ymax], ls=(0, (5, 5)), color="0.7")

# Interior grid lines
for x in [1, 2, 4]:
    ax.plot([x, x], [0, ymax], ls=(0, (5, 5)), color="0.75", linewidth=1)
for y in [1, 2]:
    ax.plot([0, xmax], [y, y], ls=(0, (5, 5)), color="0.75", linewidth=1)

# Ticks on x-axis
for x in [0, 1, 2, 4, 5]:
    ax.plot([x, x], [-0.05, 0.05], color="black", linewidth=1)

# Ticks on y-axis
for y in [0, 1, 2, 3, 4]:
    ax.plot([-0.05, 0.05], [y, y], color="black", linewidth=1)

# x labels
ax.text(0, -0.18, r"$0$", ha="center", va="top", fontsize=16)
ax.text(1, -0.18, r"$\Delta S$", ha="center", va="top", fontsize=16)
ax.text(2, -0.18, r"$2\Delta S$", ha="center", va="top", fontsize=16)
ax.text(3, -0.15, r"$\cdots$", ha="center", va="top", fontsize=18)
ax.text(4.0, -0.20, r"$S_i=i\Delta S$", ha="center", va="top", fontsize=16)
ax.text(5.0, -0.20, r"$S_{\max}=M\Delta S$", ha="center", va="top", fontsize=16)

# y labels
ax.text(-0.12, 0, r"$0$", ha="right", va="center", fontsize=16)
ax.text(-0.12, 1, r"$\Delta t$", ha="right", va="center", fontsize=16)
ax.text(-0.12, 2, r"$2\Delta t$", ha="right", va="center", fontsize=16)
ax.text(-0.12, 3, r"$\cdots$", ha="right", va="center", fontsize=18)
ax.text(-0.12, 4, r"$T=N\Delta t$", ha="right", va="center", fontsize=16)

# Mark representative point
ax.plot(4, 2, "ko", markersize=7)
ax.text(4.08, 2.08, r"$(S_i,t_n)$", fontsize=18, ha="left", va="bottom")

# Brace-like annotations using simple line + text
# Horizontal brace substitute
ax.annotate("",
            xy=(1, -0.55), xytext=(0, -0.55),
            arrowprops=dict(linewidth=0.2, mutation_scale=10))
ax.text(0.5, -0.78, r"$\Delta S=\dfrac{S_{\max}}{M}$",
        ha="center", va="top", fontsize=18)

# Vertical brace substitute
ax.annotate("",
            xy=(-0.32, 1), xytext=(-0.32, 0),
            arrowprops=dict(linewidth=0.2, mutation_scale=10))
ax.text(-0.85, 0.5, r"$\Delta t=\dfrac{T}{N}$",
        ha="center", va="center", fontsize=18)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "grid_schematic.png", dpi=300)
plt.show()