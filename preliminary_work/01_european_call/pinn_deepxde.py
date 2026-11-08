"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Adapted from the DeepXDE Black-Scholes European call example (DeepXDE PR #2056).

Black-Scholes PDE for European call option pricing using PINN.

This example solves the Black-Scholes PDE in time-to-maturity form:
    dV/dtau = 0.5*sigma^2*S^2*d^2V/dS^2 + r*S*dV/dS - r*V

with boundary conditions:
    V(S, 0) = max(S - K, 0)           [Initial condition at maturity]
    V(0, tau) = 0                      [Lower boundary]
    V(S_max, tau) = S_max - K*exp(-r*tau)  [Upper boundary]

Reference:
    Tanios, R. (2021). "Physics Informed Neural Networks in Computational Finance:
    High Dimensional Forward & Inverse Option Pricing". ETH Zurich Master's Thesis.
    https://doi.org/10.3929/ethz-b-000491555
"""

import deepxde as dde
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path


# Problem parameters
K = 100.0      # Strike price
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility
T = 1.0       # Time to maturity
S_max = 300.0 # Maximum stock price

results_dir = Path(__file__).resolve().parent / "results" / "pinn"
results_dir.mkdir(parents=True, exist_ok=True)


def pde(x, y):
    """Black-Scholes PDE residual."""
    S = x[:, 0:1]
    dy_tau = dde.grad.jacobian(y, x, i=0, j=1)
    dy_S = dde.grad.jacobian(y, x, i=0, j=0)
    dy_SS = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tau - (0.5 * sigma**2 * S**2 * dy_SS + r * S * dy_S - r * y)


def black_scholes_call(S, tau):
    """Analytical Black-Scholes formula."""
    tau = np.maximum(tau, 1e-8)
    S = np.maximum(S, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def gen_testdata():
    """Generate test data with analytical solution."""
    S = np.linspace(0, S_max, 256).reshape(256, 1)
    tau = np.linspace(0, T, 201).reshape(201, 1)
    SS, TT = np.meshgrid(S, tau)
    X = np.vstack((np.ravel(SS), np.ravel(TT))).T
    y = np.array([black_scholes_call(X[i, 0], X[i, 1]) for i in range(len(X))])
    return X, y[:, None]


# Computational geometry
geom = dde.geometry.Interval(0, S_max)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Boundary and initial conditions
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.maximum(x[:, 0:1] - K, 0.0),
    lambda _, on_initial: on_initial,
)
bc_lower = dde.icbc.DirichletBC(
    geomtime,
    lambda x: np.zeros((len(x), 1)),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
)
bc_upper = dde.icbc.DirichletBC(
    geomtime,
    lambda x: x[:, 0:1] - K * np.exp(-r * x[:, 1:2]),
    lambda x, on_boundary: on_boundary and np.isclose(x[0], S_max),
)

# Define PDE problem
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_lower, bc_upper],
    num_domain=2047,
    num_boundary=63,
    num_initial=127,
    num_test=2047,
    train_distribution="Sobol",
)

# Neural network
net = dde.nn.FNN([2] + [28] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Training
model.compile("adam", lr=1e-3)
model.train(iterations=20000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# Plot and save results
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=str(results_dir))

# Validation
X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt(results_dir / "test.dat", np.hstack((X, y_true, y_pred)))
np.savetxt(results_dir / "residual.dat", np.hstack((X, f)))

# Plot a representative slice at tau = 0.5 and save it for later reuse
S_plot = np.linspace(0, S_max, 300).reshape(-1, 1)
tau_plot = 0.5 * np.ones_like(S_plot)
X_plot = np.hstack((S_plot, tau_plot))
V_exact = np.array([black_scholes_call(X_plot[i, 0], X_plot[i, 1]) for i in range(len(X_plot))]).reshape(-1, 1)
V_pred = model.predict(X_plot)
np.savetxt(results_dir / "slice_tau_0p5.dat", np.hstack((S_plot, tau_plot, V_exact, V_pred)))

plt.figure(figsize=(8, 5))
plt.plot(S_plot, V_exact, label="Exact Black-Scholes")
plt.plot(S_plot, V_pred, "--", label="DeepXDE PINN")
plt.xlabel("Asset price S")
plt.ylabel("Option value V(S, tau)")
plt.title("European Call Option via DeepXDE (tau = 0.5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "exact_vs_pinn_tau_0p5.png", dpi=300, bbox_inches="tight")
plt.show()
