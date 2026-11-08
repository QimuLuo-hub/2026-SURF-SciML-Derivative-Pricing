import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Exact obstacle problem figure
# -----------------------------
# We want a truly exact picture satisfying:
# 1. u(-1) = 0, u(1) = 0
# 2. u'' = 0 outside [A, B]  -> straight lines
# 3. u(x) = f(x) on [A, B]
# 4. smooth matching at A and B:
#       u(A)=f(A), u'(A)=f'(A)
#       u(B)=f(B), u'(B)=f'(B)
# -----------------------------

# Choose symmetric contact points
A = -0.5
B = 0.5

# Obstacle chosen so that the tangent lines at A and B
# pass exactly through (-1,0) and (1,0)
def f(x):
    return 0.75 - x**2

def fp(x):
    return -2 * x

# Domain ONLY on [-1,1], since those are the true endpoints
x = np.linspace(-1.0, 1.0, 1200)

# Build u(x) piecewise exactly
u = np.zeros_like(x)

left = x < A
middle = (x >= A) & (x <= B)
right = x > B

# Left straight line: through (-1,0) and tangent to f at x=A
# Since A=-0.5, slope = f'(A)=1
u[left] = x[left] + 1

# Middle contact region
u[middle] = f(x[middle])

# Right straight line: through (1,0) and tangent to f at x=B
# Since B=0.5, slope = f'(B)=-1
u[right] = 1 - x[right]

# Check exact conditions numerically
print("u(-1) =", u[0])
print("u(1)  =", u[-1])
print("f(A) =", f(A), ", left line at A =", A + 1)
print("f(B) =", f(B), ", right line at B =", 1 - B)
print("f'(A) =", fp(A), ", left slope = 1")
print("f'(B) =", fp(B), ", right slope = -1")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

# Curves
ax.plot(x, u, linewidth=3.5, label=r"String $u(x)$")
ax.plot(x, f(x), linestyle="--", linewidth=2.2, label=r"Obstacle $f(x)$")

# Axis
ax.axhline(0, linewidth=1)

# Endpoints
ax.scatter([-1, 1], [0, 0], s=60, zorder=5)
ax.text(-1.0, -0.045, r"$(-1,0)$", fontsize=12, ha="center", va="top")
ax.text(1.0, -0.045, r"$(1,0)$", fontsize=12, ha="center", va="top")

# Contact points
ax.scatter([A, B], [f(A), f(B)], s=70, zorder=6)
ax.text(A, f(A) + 0.04, "A", fontsize=14, ha="center")
ax.text(B, f(B) + 0.04, "B", fontsize=14, ha="center")

# Region labels
ax.text(-1.0, 0.22, r"$u''=0$", fontsize=14)
ax.text(-0.2, 0.82, r"$u(x)=f(x)$", fontsize=14, ha="center")
ax.text(1.0, 0.22, r"$u''=0$", fontsize=14, ha="right")

# Smooth-pasting annotations
ax.annotate(
    r"$u(A)=f(A)$" + "\n" + r"$u'(A)=f'(A)$",
    xy=(A, f(A)),
    xytext=(-0.92, 0.64),
    arrowprops=dict(arrowstyle="->", lw=1.1),
    fontsize=12
)

ax.annotate(
    r"$u(B)=f(B)$" + "\n" + r"$u'(B)=f'(B)$",
    xy=(B, f(B)),
    xytext=(0.56, 0.64),
    arrowprops=dict(arrowstyle="->", lw=1.1),
    fontsize=12
)

# Labels
# ax.set_title("Exact Obstacle Problem with Smooth Contact", fontsize=17, pad=14)
ax.set_xlabel(r"$x$", fontsize=15)
ax.set_ylabel(r"$u(x)$", fontsize=15)

# Limits
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-0.08, 0.92)

# Legend
ax.legend(loc="lower center", fontsize=11, frameon=True)

# Clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "obstacle_schematic.png", dpi = 300)
plt.show()
