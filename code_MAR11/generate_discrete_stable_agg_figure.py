#!/usr/bin/env python3
"""
Generate Figure: Simulated stable aggregate dynamics with two components.

This script generates a multi-panel figure showing:
- Top left: Distribution of weights for the two components
- Top right: Aggregated process trajectory
- Middle/Bottom: Individual latent component processes

Based on the discrete mixture model from stable_aggregates.py with MAR(1,1) extension.

Author: Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gamma
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'text.usetex': False,
    'font.family': 'serif',
})


def simulate_alpha_stable(n: int, alpha: float, beta: float = 0.0, 
                          sigma: float = 1.0, mu: float = 0.0,
                          random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from an alpha-stable distribution S(alpha, beta, sigma, mu).
    
    Parameters
    ----------
    n : int
        Number of samples to generate.
    alpha : float
        Stability parameter (0 < alpha <= 2).
    beta : float
        Skewness parameter (-1 <= beta <= 1).
    sigma : float
        Scale parameter (sigma > 0).
    mu : float
        Location parameter.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Array of n samples from the alpha-stable distribution.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    samples = stats.levy_stable.rvs(
        alpha=alpha,
        beta=beta,
        loc=mu,
        scale=sigma,
        size=n,
        random_state=random_state
    )
    return samples


def generate_mar11_process(n: int, phi: float, psi: float, 
                           alpha: float, beta: float = 0.0,
                           sigma: float = 1.0, burnin: int = 500,
                           max_lag: int = 300, 
                           random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate a MAR(1,1) process with alpha-stable innovations.
    
    The MAR(1,1) model is defined as:
        (1 - phi * L)(1 - psi * L^{-1}) X_t = epsilon_t
    
    where L is the lag operator and L^{-1} is the lead operator.
    
    The MA(infinity) representation coefficients are:
        d_k = psi^k / (1 - phi*psi) for k >= 0  (noncausal part)
        d_k = phi^|k| / (1 - phi*psi) for k < 0 (causal part)
    
    Parameters
    ----------
    n : int
        Length of the series to generate.
    phi : float
        Causal AR coefficient (|phi| < 1).
    psi : float
        Noncausal AR coefficient (|psi| < 1).
    alpha : float
        Stability parameter (0 < alpha <= 2).
    beta : float
        Skewness parameter (-1 <= beta <= 1).
    sigma : float
        Scale parameter of innovations.
    burnin : int
        Burn-in period to discard.
    max_lag : int
        Maximum number of lags for MA truncation.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray
        Simulated MAR(1,1) process of length n.
    """
    if abs(phi) >= 1 or abs(psi) >= 1:
        raise ValueError("Both |phi| and |psi| must be strictly less than 1")
    
    total_len = n + burnin + 2 * max_lag
    
    # Generate alpha-stable innovations
    innovations = simulate_alpha_stable(
        total_len, alpha, beta, sigma, random_state=random_state
    )
    
    # Compute MA(infinity) coefficients
    # Normalization factor
    norm_factor = 1.0 / (1.0 - phi * psi)
    
    # Build the process using MA representation
    X = np.zeros(n + burnin)
    
    for t in range(max_lag, n + burnin + max_lag):
        weighted_sum = 0.0
        
        # Noncausal part: k >= 0 (future innovations)
        for k in range(max_lag):
            d_k = (psi ** k) * norm_factor
            if t + k < total_len:
                weighted_sum += d_k * innovations[t + k]
        
        # Causal part: k < 0 (past innovations)
        for k in range(1, max_lag):
            d_minus_k = (phi ** k) * norm_factor
            if t - k >= 0:
                weighted_sum += d_minus_k * innovations[t - k]
        
        X[t - max_lag] = weighted_sum
    
    return X[burnin:]


def generate_mar11_aggregate(n: int, phis: np.ndarray, psis: np.ndarray,
                              pis: np.ndarray, alpha: float, 
                              beta: float = 0.0, sigma: float = 1.0,
                              burnin: int = 500, max_lag: int = 300,
                              random_state: Optional[int] = None
                              ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate an aggregated MAR(1,1) process as a weighted sum of J components.
    
    The aggregate is defined as:
        X_t = sum_{j=1}^J pi_j * X_{j,t}
    
    where each X_{j,t} follows a MAR(1,1) with parameters (phi_j, psi_j).
    
    Parameters
    ----------
    n : int
        Length of the series to generate.
    phis : np.ndarray
        Array of causal AR coefficients for each component.
    psis : np.ndarray
        Array of noncausal AR coefficients for each component.
    pis : np.ndarray
        Array of aggregation weights (should sum to 1).
    alpha : float
        Common stability parameter.
    beta : float
        Common skewness parameter.
    sigma : float
        Common scale parameter.
    burnin : int
        Burn-in period.
    max_lag : int
        Maximum lag for MA truncation.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[np.ndarray, List[np.ndarray]]
        The aggregated process and list of individual components.
    """
    J = len(phis)
    assert len(psis) == J, "phis and psis must have the same length"
    assert len(pis) == J, "pis must have the same length as phis"
    
    components = []
    
    for j in range(J):
        seed_j = random_state + j * 1000 if random_state is not None else None
        component = generate_mar11_process(
            n, phis[j], psis[j], alpha, beta, sigma, 
            burnin, max_lag, random_state=seed_j
        )
        components.append(component)
    
    # Aggregate: weighted sum of components
    aggregated = sigma * np.sum(
        [pis[j] * components[j] for j in range(J)], axis=0
    )
    
    return aggregated, components


def generate_discrete_stable_agg_figure(
    T: int = 500,
    psi1: float = 0.90,
    psi2: float = 0.70,
    pi1: float = 0.40,
    pi2: float = 0.60,
    phi1: float = 0.30,
    phi2: float = 0.20,
    alpha: float = 1.5,
    beta: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate the multi-panel figure for discrete stable aggregate dynamics.
    
    Parameters
    ----------
    T : int
        Sample size for the simulation.
    psi1, psi2 : float
        Noncausal AR coefficients for components 1 and 2.
    pi1, pi2 : float
        Aggregation weights for components 1 and 2.
    phi1, phi2 : float
        Causal AR coefficients for components 1 and 2.
    alpha : float
        Stability parameter.
    beta : float
        Skewness parameter.
    sigma : float
        Scale parameter.
    seed : int
        Random seed for reproducibility.
    save_path : str, optional
        Path to save the figure (PDF format recommended).
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object.
    """
    # Parameters
    phis = np.array([phi1, phi2])
    psis = np.array([psi1, psi2])
    pis = np.array([pi1, pi2])
    
    # Generate the aggregate and components
    np.random.seed(seed)
    aggregate, components = generate_mar11_aggregate(
        n=T, phis=phis, psis=psis, pis=pis, 
        alpha=alpha, beta=beta, sigma=sigma,
        burnin=500, max_lag=300, random_state=seed
    )
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
    # Color scheme
    colors = {
        'component1': '#1f77b4',  # Blue
        'component2': '#ff7f0e',  # Orange
        'aggregate': '#2ca02c',   # Green
        'weight1': '#1f77b4',
        'weight2': '#ff7f0e'
    }
    
    # ========== Top Left: Distribution of weights ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create a simple bar chart showing the weights
    components_labels = ['Component 1', 'Component 2']
    bar_width = 0.35
    x_pos = np.arange(len(components_labels))
    
    # Weights (pi)
    bars1 = ax1.bar(x_pos - bar_width/2, [pi1, pi2], bar_width, 
                    label=r'Weight $\pi_j$', color=[colors['weight1'], colors['weight2']],
                    alpha=0.7, edgecolor='black')
    
    # Persistence (psi)
    bars2 = ax1.bar(x_pos + bar_width/2, [psi1, psi2], bar_width,
                    label=r'Persistence $\psi_j$', color=[colors['weight1'], colors['weight2']],
                    alpha=0.4, edgecolor='black', hatch='//')
    
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Parameter Value')
    ax1.set_title('Distribution of Weights and Persistence')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components_labels)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add parameter values as text
    for i, (p, w) in enumerate([(psi1, pi1), (psi2, pi2)]):
        ax1.annotate(f'$\\psi_{i+1}={p:.2f}$\n$\\pi_{i+1}={w:.2f}$', 
                     xy=(i, max(p, w) + 0.05), ha='center', fontsize=9)
    
    # ========== Top Right: Aggregated process ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    time_index = np.arange(T)
    ax2.plot(time_index, aggregate, color=colors['aggregate'], 
             linewidth=0.8, alpha=0.9)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel(r'$\mathcal{X}_t$')
    ax2.set_title(r'Aggregated Process $\mathcal{X}_t$')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T)
    
    # Add annotation for parameters
    param_text = f'$\\alpha={alpha}$, $\\beta={beta}$, $\\sigma={sigma}$'
    ax2.text(0.98, 0.95, param_text, transform=ax2.transAxes, 
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== Middle: Component 1 ==========
    ax3 = fig.add_subplot(gs[1, :])
    
    ax3.plot(time_index, components[0], color=colors['component1'], 
             linewidth=0.7, alpha=0.9)
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel(r'$X_{1,t}$')
    ax3.set_title(f'Component 1: $\\phi_1={phi1:.2f}$, $\\psi_1={psi1:.2f}$, $\\pi_1={pi1:.2f}$ (High Persistence)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, T)
    
    # ========== Bottom: Component 2 ==========
    ax4 = fig.add_subplot(gs[2, :])
    
    ax4.plot(time_index, components[1], color=colors['component2'], 
             linewidth=0.7, alpha=0.9)
    ax4.set_xlabel('Time $t$')
    ax4.set_ylabel(r'$X_{2,t}$')
    ax4.set_title(f'Component 2: $\\phi_2={phi2:.2f}$, $\\psi_2={psi2:.2f}$, $\\pi_2={pi2:.2f}$ (Moderate Persistence)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, T)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    
    return fig


def main():
    """Main function to generate and display the figure."""
    
    print("=" * 60)
    print("Generating: Discrete Stable Aggregate Dynamics Figure")
    print("=" * 60)
    
    # Parameters from the paper description
    # Component 1: High persistence (psi_1 = 0.90)
    # Component 2: Moderate persistence (psi_2 = 0.70)
    
    fig = generate_discrete_stable_agg_figure(
        T=500,
        psi1=0.20,      # Noncausal persistence for component 1
        psi2=0.70,      # Noncausal persistence for component 2
        pi1=0.40,       # Weight for component 1
        pi2=0.60,       # Weight for component 2
        phi1=0.90,      # Causal persistence for component 1
        phi2=0.50,      # Causal persistence for component 2
        alpha=1.5,      # Tail index
        beta=0.0,       # Symmetric distribution
        sigma=1.0,      # Scale parameter
        seed=42,        # Reproducibility
        save_path='discretestableagg.pdf'
    )
    
    print("\nFigure generation complete!")
    print("Parameters used:")
    print(f"  Component 1: psi_1=0.90, phi_1=0.30, pi_1=0.40")
    print(f"  Component 2: psi_2=0.70, phi_2=0.20, pi_2=0.60")
    print(f"  Distribution: alpha=1.5, beta=0.0, sigma=1.0")
    print(f"  Sample size: T=500")
    
    plt.show()


if __name__ == "__main__":
    main()
