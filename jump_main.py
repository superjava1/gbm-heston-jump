import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
T = 1.0              # Time horizon in years
N = 252              # Number of time steps (e.g., daily steps)
dt = T / N           # Time increment
n_paths = 1000       # Number of Monte Carlo simulation paths

# Heston Model parameters
S0 = 100.0           # Initial asset price
v0 = 0.04            # Initial variance
mu = 0.05            # Drift for the asset price
kappa = 2.0          # Mean reversion speed of variance
theta = 0.04         # Long-run average variance
sigma_v = 0.3        # Volatility of the variance process (vol-of-vol)
rho = -0.7           # Correlation between asset and variance Brownian motions

# Jump parameters (Merton Jump-Diffusion extension)
lamb = 0.5           # Intensity of jumps (expected number of jumps per year)
mu_j = -0.1          # Mean of log jump size
sigma_j = 0.2        # Standard deviation of log jump size

# Pre-calculate constants
sqrt_dt = np.sqrt(dt)

def simulate_heston_with_jumps():
    # Initialize arrays for asset prices and variance
    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # Pre-generate random variables for efficiency:
    # Correlated Brownian increments for asset and variance processes
    Z1 = np.random.normal(size=(n_paths, N))
    Z2 = np.random.normal(size=(n_paths, N))
    dW_v = sqrt_dt * Z2
    dW_S = sqrt_dt * (rho * Z2 + np.sqrt(1 - rho**2) * Z1)

    # For jumps: simulate the number of jumps for each time step and each path.
    # Using Poisson approximation for dt.
    # We'll also simulate the jump sizes from a log-normal distribution.
    jumps = np.random.poisson(lamb * dt, size=(n_paths, N))
    jump_sizes = np.exp(mu_j + sigma_j * np.random.normal(size=(n_paths, N))) - 1.0

    for t in range(1, N + 1):
        # Previous variance (ensure non-negative by full truncation Euler)
        v_prev = np.maximum(v[:, t - 1], 0)

        # Variance process (Heston)
        dv = kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * dW_v[:, t - 1]
        v[:, t] = v_prev + dv
        # Ensure variance remains positive (full truncation)
        v[:, t] = np.maximum(v[:, t], 0)

        # Asset price process:
        # Drift & diffusion term with stochastic volatility:
        dS_diff = (mu - lamb * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * S[:, t - 1] * dt \
                  + np.sqrt(v_prev) * S[:, t - 1] * dW_S[:, t - 1]
        # Jump term: if jumps[t-1] > 0, then asset price jumps
        dS_jump = S[:, t - 1] * jump_sizes[:, t - 1] * jumps[:, t - 1]
        S[:, t] = S[:, t - 1] + dS_diff + dS_jump

    return S, v

if __name__ == "__main__":
    # Run simulation
    S_paths, v_paths = simulate_heston_with_jumps()

    # Plot a sample of asset price paths
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(np.linspace(0, T, N + 1), S_paths[i, :], lw=1)
    plt.title("Monte Carlo Simulation: Heston Model with Jumps")
    plt.xlabel("Time (years)")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.show()
