import numpy as np
import math


# ===============================
# Parameters (same as PSOR)
# ===============================
STRIKE = 10.0
RATE = 0.10
SIGMA = 0.40

ASSET_PRICES = np.array([0.0, 2.0, 4.0, 6.0, 8.0,
                         10.0, 12.0, 14.0, 16.0])


# ===============================
# Binomial American Put
# ===============================
def american_put_binomial(S0, T, N):
    if S0 == 0:
        return STRIKE  # immediate exercise

    dt = T / N
    u = math.exp(SIGMA * math.sqrt(dt))
    d = 1.0 / u

    disc = math.exp(-RATE * dt)
    p = (math.exp(RATE * dt) - d) / (u - d)

    # Terminal stock prices
    prices = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    values = np.maximum(STRIKE - prices, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        # recompute stock prices at time step i
        prices = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)])

        continuation = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        exercise = np.maximum(STRIKE - prices, 0.0)

        values = np.maximum(continuation, exercise)

    return values[0]


# ===============================
# Build Table (same format)
# ===============================
def build_table(N):
    rows = []

    for S in ASSET_PRICES:
        payoff = max(STRIKE - S, 0.0)

        val_3m = american_put_binomial(S, 0.25, N)
        val_6m = american_put_binomial(S, 0.50, N)

        rows.append((S, payoff, val_3m, val_6m))

    return rows


# ===============================
# Print Table (same style)
# ===============================
def print_table(rows):
    header = (
        f"{'Asset Price':>12}"
        f"{'Payoff Value':>16}"
        f"{'3 Months Amer.':>18}"
        f"{'6 Months Amer.':>18}"
    )
    print(header)
    print("-" * len(header))

    for s, payoff, a3, a6 in rows:
        print(f"{s:12.2f}{payoff:16.4f}{a3:18.4f}{a6:18.4f}")


# ===============================
# Main
# ===============================
def main():
    N = 4000  # tree steps (increase for accuracy)

    print("Binomial Model Parameters")
    print(f"  Strike E   = {STRIKE}")
    print(f"  Rate r     = {RATE}")
    print(f"  Sigma      = {SIGMA}")
    print(f"  Steps N    = {N}")
    print()

    rows = build_table(N)
    print_table(rows)


if __name__ == "__main__":
    main()