# Monte Carlo Simulation using the Heston Model with Jumps

This project provides a Monte Carlo simulation of asset prices using an advanced stochastic volatility model based on the Heston model with jumps (a jump-diffusion extension). The simulation is implemented in Python and uses Euler discretization for the numerical solution.

## Overview

The model used in this simulation extends the classical Geometric Brownian Motion (GBM) by incorporating:
- **Stochastic Volatility:** The variance follows the Heston process.
- **Jumps:** The asset price process includes jumps following a Poisson process with lognormally distributed jump sizes.

## Model Formulation

### 1. Heston Stochastic Volatility Model

The asset price \( S_t \) and variance \( v_t \) are modeled as follows:

#### Asset Price Process:
\[
dS_t = \left(\mu - \lambda \left(e^{\mu_j + \frac{1}{2}\sigma_j^2} - 1\right)\right) S_t dt + \sqrt{v_t} S_t dW_t^S + S_t \, dJ_t
\]

#### Variance Process:
\[
dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^v
\]

Where:
- \( \mu \) is the drift of the asset.
- \( \lambda \) is the jump intensity (expected number of jumps per year).
- \( \mu_j \) and \( \sigma_j \) are the mean and standard deviation of the logarithm of the jump size, respectively.
- \( \kappa \) is the rate at which \( v_t \) reverts to its long-term mean \( \theta \).
- \( \sigma_v \) is the volatility of the variance (vol-of-vol).
- \( dW_t^S \) and \( dW_t^v \) are two correlated Brownian motions with correlation coefficient \( \rho \).
- \( dJ_t \) represents the jump component in the asset price.

### 2. Jump Component

The jump component is modeled using a Poisson process:
- The number of jumps in a small time interval \( dt \) is given by a Poisson random variable with mean \( \lambda \, dt \).
- When a jump occurs, the jump size is modeled as:
  \[
  J = e^{\mu_j + \sigma_j Z} - 1, \quad Z \sim \mathcal{N}(0,1)
  \]
  
Thus, the asset price adjustment due to jumps in a time step is:
\[
\text{Jump Contribution} = S_t \times J \times \text{(number of jumps)}
\]

## Numerical Simulation

The simulation uses the Euler-Maruyama method with full truncation to ensure non-negative variance. At each time step:
1. The variance \( v_t \) is updated using:
   \[
   v_{t+dt} = v_t + \kappa (\theta - \max(v_t, 0)) dt + \sigma_v \sqrt{\max(v_t, 0)} \, dW_t^v
   \]
2. The asset price \( S_t \) is updated by combining the drift, diffusion, and jump components:
   \[
   S_{t+dt} = S_t + \left(\mu - \lambda \left(e^{\mu_j + \frac{1}{2}\sigma_j^2} - 1\right)\right) S_t dt + \sqrt{\max(v_t, 0)} S_t \, dW_t^S + S_t \times J \times \text{(number of jumps)}
   \]

## How to Run

1. **Prerequisites:**
   - Python 3.x
   - Required libraries: `numpy`, `matplotlib`

2. **Install Required Packages:**

   You can install the necessary packages using pip:

   ```bash
   pip install numpy matplotlib
   ```

3. **Run the Simulation:**

   Execute the Python script:

   ```bash
   python monte_carlo_heston_jumps.py
   ```

   This will run the simulation and display a plot of sample asset price paths.

## Parameters

Below are the key parameters used in the simulation:
- **Time Parameters:**
  - \( T \): Time horizon (years)
  - \( N \): Number of time steps
- **Asset and Variance:**
  - \( S_0 \): Initial asset price
  - \( v_0 \): Initial variance
  - \( \mu \): Asset drift
- **Heston Model:**
  - \( \kappa \): Mean reversion speed
  - \( \theta \): Long-term variance mean
  - \( \sigma_v \): Volatility of variance
  - \( \rho \): Correlation coefficient between the asset and variance Brownian motions
- **Jump Model:**
  - \( \lambda \): Jump intensity (jumps per year)
  - \( \mu_j \): Mean log jump size
  - \( \sigma_j \): Standard deviation of log jump size

## Conclusion

This simulation offers an advanced framework for modeling asset prices that account for both stochastic volatility and jumps, providing a richer dynamic than the standard GBM. It is particularly useful for risk management, option pricing, and quantitative analysis in finance.

Feel free to modify the parameters and extend the simulation as needed.
