import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import warnings
import multiprocessing
from joblib import Parallel, delayed
from itertools import permutations
from scipy import stats, optimize
from scipy.special import gamma
# from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Union, Any

warnings.filterwarnings('ignore')

class StableAggregates:
    """
    Class encapsulating functionalities for estimation of aggregations
    of non-causal AR(1) processes with alpha-stable innovations.

    This class implements:
    - Simulation of alpha-stable distributions
    - Generation of non-causal AR(1) processes
    - Parameter estimation for discrete and continuous mixtures
    - Performance evaluation of estimators via Monte Carlo simulations
    """
    
    def __init__(self, model_type: str = 'discrete', seed: Optional[int] = None) -> None:
        """
        Initializes the StableAggregates object.

        Parameters:
        -----------
        model_type : str
            Model type: 'discrete' or 'continuous'
        seed : int or None
            Seed for random generation
        """
        if model_type not in ['discrete', 'continuous']:
            raise ValueError("model_type must be 'discrete' or 'continuous'")
            
        self.model_type = model_type
        
        if seed is not None:
            np.random.seed(seed)
        
        self.results_dir = 'results'
        os.makedirs(self.results_dir, exist_ok=True)

    
    # --- Simulation and characterization of alpha-stable distributions ---

    
    def simulate_alpha_stable(self, n: int, alpha: float, beta: float, sigma: float = 1.0, 
                            mu: float = 0.0, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generates samples from an alpha-stable distribution S(alpha, beta, sigma, mu)
        using scipy.stats.levy_stable.

        Parameters:
        -----------
        n : int
            Number of samples to generate
        alpha : float (0 < alpha <= 2)
            Stability parameter (tail thickness)
        beta : float (-1 <= beta <= 1)
            Asymmetry parameter
        sigma : float (sigma > 0)
            Scale parameter
        mu : float
            Location parameter
        random_state : int or None
            Seed for random generation
            
        Returns:
        --------
        numpy.ndarray
            Samples from the alpha-stable distribution
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

    
    def estimate_ecf(self, X: np.ndarray, u: Union[float, np.ndarray], 
                    v: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        """
        Calculates the empirical characteristic function (ECF) of a time series.

        Parameters:
        -----------
        X : numpy.ndarray
            Time series
        u : float or numpy.ndarray
            Evaluation points for the characteristic function
        v : float or numpy.ndarray or None
            Additional evaluation points for the joint characteristic function
            
        Returns:
        --------
        numpy.ndarray
            Empirical characteristic function evaluated at points u (and v if specified)
        """
        if v is None:  # Univariate case
            return np.mean(np.exp(1j * u[:, np.newaxis] * X), axis=1)
        else:  # Bivariate case
            return np.mean(np.exp(1j * (u * X[1:] + v * X[:-1])))

    
    def moment_function(self, lambda_vals: np.ndarray, ecf: callable, 
                        u0: float, alpha: float) -> np.ndarray:
        """
        Calculates the moment function g(λ) = log|Ψ(u0, λu0)|/|u0|^α.

        Parameters:
        -----------
        lambda_vals : array_like
            Values of λ for which to calculate g(λ)
        ecf : function
            Empirical characteristic function
        u0 : float
            Small parameter for evaluation near the origin
        alpha : float
            Stability parameter
            
        Returns:
        --------
        numpy.ndarray
            Values of g(λ)
        """
        g_lambda = []
        for lam in lambda_vals:
            psi = ecf(u0, lam * u0)
            log_abs_psi = np.log(np.abs(psi))
            g_lambda.append(log_abs_psi / (np.abs(u0)**alpha))
        
        return np.array(g_lambda)


    def char_function_discrete(self, u: float, v: float, theta: np.ndarray, 
                            alpha: float, beta: float) -> complex:
        """
        Calculates the joint characteristic function of (Y_t, Y_{t-1}) for discrete mixture.
        """
        J = len(theta) // 2
        phis = theta[:J]
        varsigmas = theta[J:]
        
        c = np.sum(varsigmas)
        pis = varsigmas / c
        
        exponent = 0.0 + 0.0j
        
        for j in range(J):
            phi_j = phis[j]
            pi_j = pis[j]
            
            if alpha != 1:
                component_exponent = -pi_j**alpha * c**alpha * (
                    np.abs(u + phi_j*v)**alpha * (1 - 1j * beta * np.sign(u + phi_j*v) * np.tan(np.pi * alpha / 2)) / (1-np.abs(phi_j)**alpha) 
                    + np.abs(v)**alpha * (1 - 1j * beta * np.sign(v) * np.tan(np.pi * alpha / 2))
                )
            else:
                component_exponent = -pi_j * c * (
                    np.abs(u + phi_j*v) * (1 + 1j * beta * np.sign(u + phi_j*v) * 2/np.pi * np.log(np.abs(u + phi_j*v) + 1e-10)) / (1-np.abs(phi_j)) 
                    + np.abs(v) * (1 + 1j * beta * np.sign(v) * 2/np.pi * np.log(np.abs(v) + 1e-10))
                )
            
            exponent += component_exponent
        
        return np.exp(exponent)


    def char_function_beta(self, u: float, v: float, theta: np.ndarray, 
                        alpha: float, beta: float, num_points: int = 100) -> complex:
        """
        Calculates the joint characteristic function for beta mixture.
        """
        c, a, b = theta
        
        phi_points = np.linspace(0.001, 0.999, num_points)
        pi_density = self.beta_pdf(phi_points, a, b)
        dphi = phi_points[1] - phi_points[0]
        
        exponent = 0.0 + 0.0j
        
        for j, phi in enumerate(phi_points):
            if alpha != 1:
                component_exponent = -c**alpha * pi_density[j] * dphi * (
                    np.abs(u + phi*v)**alpha * (1 - 1j * beta * np.sign(u + phi*v) * np.tan(np.pi * alpha / 2)) / (1-phi**alpha) 
                    + np.abs(v)**alpha * (1 - 1j * beta * np.sign(v) * np.tan(np.pi * alpha / 2))
                )
            else:
                component_exponent = -c * pi_density[j] * dphi * (
                    np.abs(u + phi*v) * (1 + 1j * beta * np.sign(u + phi*v) * 2/np.pi * np.log(np.abs(u + phi*v) + 1e-10)) / (1-phi) 
                    + np.abs(v) * (1 + 1j * beta * np.sign(v) * 2/np.pi * np.log(np.abs(v) + 1e-10))
                )
            
            exponent += component_exponent
        
        return np.exp(exponent)


    def theoretical_moment_function_beta(self, lambda_vals: np.ndarray, theta: np.ndarray, 
                                        alpha: float, num_points: int = 100) -> np.ndarray:
        """
        Calculates the theoretical moment function for a mixture with beta distribution
        using only the limiting behavior (β is not involved).

        Parameters:
        -----------
        lambda_vals : array_like
            Values of λ for which to calculate the function
        theta : array_like
            Model parameters (without β)
        alpha : float
            Stability parameter
        num_points : int
            Number of points for numerical integration
            
        Returns:
        --------
        numpy.ndarray
            Values of g(λ; θ) at the limit
        """
        c, a, b = theta
        
        # Check constraints
        if c <= 0 or a <= 0 or b <= 0:
            return np.ones_like(lambda_vals) * np.inf
        
        # Quadrature points for numerical integration
        phi_points = np.linspace(0.001, 0.999, num_points)
        
        # Calculate density at quadrature points
        pi_density = self.beta_pdf(phi_points, a, b)
        
        # Interval width for integration
        dphi = phi_points[1] - phi_points[0]
        
        g_lambda = np.zeros_like(lambda_vals, dtype=float)
        
        for i, lam in enumerate(lambda_vals):
            integral_sum = 0
            
            for j, phi in enumerate(phi_points):
                # Limiting behavior: β terms disappear
                term1 = np.abs(1 + phi * lam)**alpha / (1 - phi**alpha)
                term2 = np.abs(lam)**alpha
                integrand = pi_density[j] * (term1 + term2)
                
                integral_sum += integrand * dphi
            
            g_lambda[i] = -c**alpha * integral_sum
            
        return g_lambda


    # --- Generation of noncausal AR(1) processes and their aggregates ---

    def generate_noncausal_ar1(self, n: int, phi: float, alpha: float, beta: float, 
                            sigma: float = 1.0, burnin: int = 1000, max_lag: int = 500) -> np.ndarray:
        """
        Generates a non-causal AR(1) process with alpha-stable innovations.

        Parameters:
        -----------
        n : int
            Length of the series to generate
        phi : float (|phi| < 1)
            AR coefficient
        alpha : float (0 < alpha <= 2)
            Stability parameter
        beta : float (-1 <= beta <= 1)
            Asymmetry parameter
        sigma : float (sigma > 0)
            Scale parameter of innovations
        burnin : int
            Burn-in period
        max_lag : int
            Maximum number of lags for MA approximation
            
        Returns:
        --------
        numpy.ndarray
            Simulated non-causal AR(1) process
        """
        total_len = n + burnin
        
        # Generate the alpha-stable innovations
        innovations = self.simulate_alpha_stable(total_len + max_lag, alpha, beta, sigma)
        
        # Initialise the process as a weighted sum of innovations (MA representation)
        X = np.zeros(total_len)
        
        # Build the process from its MA(infinity) representation
        # X_t = ∑_j=0^∞ φ^j ε_{t+j}
        for t in range(total_len):
            weighted_sum = 0
            for j in range(max_lag):
                if t + j < total_len + max_lag:
                    weighted_sum += (phi ** j) * innovations[t + j]
            X[t] = weighted_sum
        
        # Return the series without the burn-in period
        return X[burnin:]


    def generate_mixture(self, n: int, params_dict: Dict[str, Any], 
                        burnin: int = 1000, max_lag: int = 500) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generates an aggregated process according to the model type (discrete or continuous).
        Modified to handle both parameterization options for discrete model.

        Parameters:
        -----------
        n : int
            Length of the series to generate
        params_dict : dict
            Dictionary containing model parameters
        burnin : int
            Burn-in period
        max_lag : int
            Maximum number of lags for MA approximation
            
        Returns:
        --------
        numpy.ndarray
            Aggregated process
        list
            List of individual components
        """
        if self.model_type == 'discrete':
            # Check available parameters for the discrete model
            phis = params_dict['phis']
            alpha = params_dict['alpha']
            beta = params_dict['beta']
            sigma = params_dict.get('sigma', 1.0)
            
            # Handle different parameterization options
            varsigmas = params_dict.get('varsigmas', None)
            c = params_dict.get('c', None)
            pis = params_dict.get('pis', None)
            
            return self.generate_discrete_mixture(
                n, phis, alpha, beta, 
                varsigmas=varsigmas, c=c, pis=pis,
                sigma=sigma, burnin=burnin, max_lag=max_lag
            )
        else:  # continuous
            # generate_beta_mixture returns (aggregated, components, phis)
            # but we only want (aggregated, components) for uniformity
            aggregated, components, phis = self.generate_beta_mixture(
                n,
                params_dict['a'],
                params_dict['b'],
                params_dict['c'],
                params_dict['alpha'],
                params_dict['beta'],
                params_dict.get('sigma', 1.0),
                burnin,
                max_lag,
                params_dict.get('n_components', 50)
            )
            return aggregated, components


    def generate_discrete_mixture(self, n: int, phis: np.ndarray, 
                            alpha: float, beta: float, 
                            varsigmas: Optional[np.ndarray] = None,
                            c: Optional[float] = None, 
                            pis: Optional[np.ndarray] = None,
                            sigma: float = 1.0, 
                            burnin: int = 1000, 
                            max_lag: int = 500) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generates an aggregated process as a weighted sum of J non-causal AR(1) processes.
        Accepts either varsigma_j parameterization or traditional (c, pi_j) parameterization.

        Parameters:
        -----------
        n : int
            Length of the series to generate
        phis : array_like
            List of AR coefficients for each component
        alpha : float
            Common stability parameter
        beta : float
            Common asymmetry parameter
        varsigmas : array_like, optional
            Combined scale parameters for each component (varsigma_j = c * pi_j)
        c : float, optional
            Global scale parameter (used with pis)
        pis : array_like, optional
            Weights of each component (must sum to 1, used with c)
        sigma : float
            Additional global scale parameter (multiplies the entire process)
        burnin, max_lag : int
            Parameters for AR(1) process generation
            
        Returns:
        --------
        numpy.ndarray
            Aggregated process
        list
            List of individual components
        
        Notes:
        ------
        Either varsigmas or both c and pis must be provided. If both parameterizations 
        are given, varsigmas takes precedence.
        """
        J = len(phis)
        
        # Handle parameterization options
        if varsigmas is not None:
            # Use varsigmas parameterization
            assert len(varsigmas) == J, "The lists of phis and varsigmas must have the same length"
            # Calculate c and pis for internal use
            # c_internal = np.sum(varsigmas)
            # pis_internal = varsigmas / c_internal
        elif c is not None and pis is not None:
            # Use traditional (c, pi_j) parameterization
            assert len(pis) == J, "The lists of phis and pis must have the same length"
            assert np.isclose(np.sum(pis), 1.0), "The weights pis must sum to 1"
            # Calculate varsigmas for internal use
            # c_internal = c
            # pis_internal = pis
            varsigmas = c * pis
        else:
            raise ValueError("Either varsigmas or both c and pis must be provided")
        
        # Generate each non-causal AR(1) component
        components = []
        for j in range(J):
            component = self.generate_noncausal_ar1(n, phis[j], alpha, beta, 1.0, burnin, max_lag)
            components.append(component)
        
        # Aggregate components according to their weights (varsigma_j)
        aggregated = sigma * np.sum([varsigmas[j] * components[j] for j in range(J)], axis=0)
        
        return aggregated, components


    def generate_beta_mixture(self, n: int, a: float, b: float, c: float, alpha: float, 
                            beta: float, sigma: float = 1.0, burnin: int = 1000, 
                            max_lag: int = 500, n_components: int = 50) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Generates an aggregated process with a mixing measure following a transformed beta distribution.

        Parameters:
        -----------
        n : int
            Length of the series to generate
        a, b : float
            Parameters of the transformed beta distribution
        c : float
            Global scale parameter
        alpha, beta : float
            Parameters of the alpha-stable distribution
        sigma : float
            Scale parameter of innovations
        burnin, max_lag : int
            Parameters for AR(1) process generation
        n_components : int
            Number of components to approximate the integral
            
        Returns:
        --------
        tuple
            (aggregated_process, list_of_components, list_of_phis)
        """
        # Draw the phis from the transformed beta distribution
        # Distribution on [0, 1] only (not [-1, 1])
        phis = np.random.beta(a, b, n_components)
        
        # Equal weights for each component (approximation of the integral)
        pis = np.ones(n_components) / n_components
        
        # Generate the aggregated process with the discrete-mixture generator
        aggregated, components = self.generate_discrete_mixture(
            n=n, 
            phis=phis, 
            alpha=alpha, 
            beta=beta,
            c=c,
            pis=pis,
            sigma=sigma, 
            burnin=burnin, 
            max_lag=max_lag
        )
        
        return aggregated, components, phis
    

    # --- Sequential estimation (discrete and continuous mixtures) ---


    def estimate_alpha_beta(self, X: np.ndarray, step: float = 0.01, m: float = 1.0, 
                        use_robust_scale: bool = True, verbose: bool = False) -> Tuple[float, float]:
        """
        Estimates both alpha and beta parameters simultaneously using Kogon and Williams (1998) approach 
        with robust normalization.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series
        step : float
            Step size for the grid of u values
        m : float
            Maximum value for the grid of u values
        use_robust_scale : bool
            If True, uses IQR for normalization, otherwise standard deviation
        verbose : bool
            If True, prints warning messages
            
        Returns:
        --------
        tuple
            (alpha_estimate, beta_estimate)
        """
        
        # Robust scale estimation using IQR
        if use_robust_scale:
            q75, q25 = np.quantile(X, [0.75, 0.25])
            iqr = q75 - q25
            sigma0 = iqr / 2
            
            if sigma0 > 1e-10:
                X_norm = (X - np.median(X)) / sigma0
            else:
                # Fallback to standard normalization
                X_norm = (X - np.mean(X)) / np.std(X)
                sigma0 = np.std(X)
        else:
            # Standard normalization
            sigma0 = np.std(X)
            X_norm = (X - np.mean(X)) / sigma0
        
        # Create u grid
        u = np.arange(step, m + step, step)
        
        # Calculate empirical characteristic function using existing method
        ecf = self.estimate_ecf(X_norm, u)  # Univariate case (v=None)
        ecf1 = np.real(ecf)  # Real part for phase calculation
        ecf2 = np.imag(ecf)  # Imaginary part for phase calculation
        
        # Filter valid points for alpha estimation
        valid_mask = (np.abs(ecf) > 1e-10) & (np.abs(ecf) < 0.999)
        
        if np.sum(valid_mask) < 3:
            if verbose:
                print("Warning: Not enough valid points for alpha/beta estimation")
            return 1.5, 0.0  # Default values
        
        u_valid = u[valid_mask]
        ecf_valid = ecf[valid_mask]
        ecf1_valid = ecf1[valid_mask]
        ecf2_valid = ecf2[valid_mask]
        
        try:
            # Estimate alpha from the log-log plot
            x = np.log(-np.log(np.abs(ecf_valid) + 1e-10))
            y = np.log(np.abs(u_valid) + 1e-10)
            parhat = np.polyfit(y, x, 1)
            
            alpha = parhat[0]
            # Constrain alpha to be in (0, 2)
            alpha = max(0.01, min(1.99, alpha))
            
            sigma = np.exp(parhat[1]/alpha)
            
            # Estimate beta
            if alpha != 1:
                eta = np.tan(np.pi * alpha / 2) * (np.abs(u_valid) - np.abs(u_valid)**alpha)
            else:
                eta = 2 / np.pi * u_valid * np.log(np.abs(u_valid) + 1e-10)
            
            # Calculate beta using the phase information
            x_phase = np.arctan2(ecf2_valid, ecf1_valid)
            y_eta = -(sigma**alpha) * eta
            
            # Filter out invalid eta values
            valid_eta_mask = np.isfinite(y_eta) & (np.abs(y_eta) > 1e-10)
            
            if np.sum(valid_eta_mask) < 3:
                if verbose:
                    print("Warning: Not enough valid eta points for beta estimation")
                beta = 0.0
            else:
                x_phase_valid = x_phase[valid_eta_mask]
                y_eta_valid = y_eta[valid_eta_mask]
                
                parhat_beta = np.polyfit(x_phase_valid, y_eta_valid, 1)
                beta = parhat_beta[0]
                
                # Ensure beta is in [-1, 1]
                beta = np.clip(beta, -1.0, 1.0)
            
            return alpha, beta
            
        except Exception as e:
            if verbose:
                print(f"Error in alpha/beta estimation: {e}")
            return 1.5, 0.0  # Default values


    def beta_pdf(self, phi: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Calculates the beta distribution density on [0, 1].
        """
        norm = gamma(a + b) / (gamma(a) * gamma(b))
        beta_density = norm * (phi**(a-1)) * ((1-phi)**(b-1))
        return beta_density


    def estimate_mixture_params(self, X: np.ndarray, alpha: float, beta: float,
                            params_dict: Optional[Dict[str, Any]] = None, 
                            lambda_vals: Optional[np.ndarray] = None, 
                            u0: float = 0.01, method: str = 'nls',
                            max_iter: int = 200) -> Dict[str, Any]:
        """
        Estimates mixture parameters using characteristic function matching.
        Now includes beta parameter and uses complete characteristic function.

        Parameters:
        -----------
        X : numpy.ndarray
            Time series
        alpha : float
            Estimated stability parameter
        beta : float
            Estimated asymmetry parameter
        params_dict : dict or None
            Dictionary containing model-specific parameters
        lambda_vals : array_like or None
            Not used (kept for compatibility)
        u0 : float
            Not used (kept for compatibility)
        method : str
            Not used (kept for compatibility)
        max_iter : int
            Maximum number of iterations for optimization
            
        Returns:
        --------
        dict
            Dictionary containing estimated parameters (including β)
        """
        if self.model_type == 'discrete':
            if params_dict is None:
                J = 3  # Default value
            else:
                J = params_dict.get('J', 3)
            return self.estimate_discrete_mixture_params(X, alpha, beta, J, lambda_vals, u0, max_iter, method)
        else:
            num_points = 100  # Default value
            if params_dict is not None:
                num_points = params_dict.get('n_subintervals', 100)
                
            return self.estimate_beta_mixture_params(X, alpha, beta, lambda_vals, u0, max_iter, num_points)


    def estimate_discrete_mixture_params(self, X: np.ndarray, alpha: float, beta: float, J: int, 
                                    lambda_vals: Optional[np.ndarray] = None, 
                                    u0: float = 0.01, max_iter: int = 200,
                                    method: str = 'md') -> Dict[str, Any]:
        """
        Estimates parameters of a discrete mixture using either NLS or
        original minimum distance method.
        
        Parameters:
        -----------
        X : np.ndarray
            Time series data
        alpha : float
            Estimated stability parameter
        J : int
            Number of mixture components
        lambda_vals : np.ndarray, optional
            Lambda values for moment conditions
        u0 : float
            Small parameter for characteristic function evaluation
        max_iter : int
            Maximum iterations for optimization
        method : str
            Estimation method: 'nls' for NLS or 'md' for minimum distance
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing estimated parameters
        """
        if method == 'nls':
            return self.estimate_nls(X, alpha, beta, J, lambda_vals, u0, max_iter)
        elif method == 'md':
            return self.estimate_md(X, alpha, beta, J, lambda_vals, u0, max_iter)
        else:
            raise ValueError("method must be either 'nls' or 'md'")


    def estimate_beta_mixture_params(self, X: np.ndarray, alpha: float, beta: float,
                                    lambda_vals: Optional[np.ndarray] = None, 
                                    u0: float = 0.01, max_iter: int = 200, 
                                    num_points: int = 200) -> Dict[str, Any]:
        """
        Estimates parameters of a mixture with beta distribution using the limiting behavior.
        """
        if lambda_vals is None:
            lambda_vals = np.linspace(0.1, 1.5, 10)
        
        # Calculate empirical moment function
        empirical_g = []
        for lam in lambda_vals:
            psi = self.estimate_ecf(X, u0, lam * u0)
            log_abs_psi = np.log(np.abs(psi))
            g_lambda = log_abs_psi / (np.abs(u0)**alpha)
            empirical_g.append(g_lambda)
        
        empirical_g = np.array(empirical_g)
        
        # Objective function for optimization (without β)
        def objective(theta):
            theoretical_g = self.theoretical_moment_function_beta(lambda_vals, theta, alpha, num_points)
            return np.sum((empirical_g - theoretical_g)**2)
        
        # Initial parameters and bounds
        initial_params = np.array([1.0, 7.0, 6.0])  # [c, a, b]
        bounds = [(0.01, 10.0), (0.1, 10.0), (0.1, 10.0)]
        
        # Optimization
        try:
            result = optimize.minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
            
            estimated_params = result.x
        except Exception:
            # In case of failure, try another method
            try:
                result = optimize.differential_evolution(
                    objective,
                    bounds,
                    maxiter=max_iter//10,
                    tol=1e-4
                )
                estimated_params = result.x
            except Exception:
                # Last attempt
                result = optimize.minimize(
                    objective,
                    initial_params,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter}
                )
                estimated_params = result.x
        
        # Extract estimated parameters
        c_est, a_est, b_est = estimated_params
        
        return {
            'c': c_est,
            'a': a_est,
            'b': b_est,
            'alpha': alpha,
            'beta': beta
        }


    def estimate_md(self, X: np.ndarray, alpha: float, beta: float, J: int, 
                    lambda_vals: Optional[np.ndarray] = None, 
                    u0: float = 0.01, max_iter: int = 200) -> Dict[str, Any]:
        """
        Estimates parameters of a discrete mixture using the minimum distance method
        with uniform weights.
        
        Uses the traditional (c, pi_j) parameterization with explicit constraint sum(pi_j) = 1.
        
        Parameters:
        -----------
        X : np.ndarray
            Time series data
        alpha : float
            Estimated stability parameter
        J : int
            Number of mixture components
        lambda_vals : np.ndarray, optional
            Lambda values for moment conditions
        u0 : float
            Small parameter for characteristic function evaluation
        max_iter : int
            Maximum iterations for optimization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing estimated parameters
        """
        # Objective function using characteristic functions (like stablemixture.py)
        def objective(theta):
            # Check parameter validity
            phis = theta[:J]
            varsigmas = theta[J:]
            
            if np.any(np.abs(phis) >= 1) or np.any(varsigmas <= 0):
                return 1e10
            
            # Define grid for (u, v)
            if lambda_vals is None:
                u_grid = np.linspace(-1.5, 1.5, 15)
                v_grid = np.linspace(-1.5, 1.5, 15)
            else: 
                u_grid = u0 * lambda_vals
                v_grid = u0 * lambda_vals
            
            diff_sum = 0
            count = 0
            
            for u in u_grid:
                for v in v_grid:
                    # Theoretical characteristic function
                    cf_theo = self.char_function_discrete(u, v, theta, alpha, beta)
                    
                    # Empirical characteristic function
                    cf_emp = np.mean(np.exp(1j * (u * X[1:] + v * X[:-1])))
                    
                    # Squared difference
                    diff_sum += np.abs(cf_emp - cf_theo)**2
                    count += 1
            
            return diff_sum / count
        
        # Initial parameters and bounds
        phi_init = np.linspace(0.6, 0.8, J)
        varsigma_init = np.linspace(1.0, 0.7, J) # np.ones(J) / J
        initial_params = np.concatenate([phi_init, varsigma_init])
        
        # Bounds for parameters
        phi_bounds = [(-0.99, 0.99)] * J
        varsigma_bounds = [(0.001, 10.0)] * J
        bounds = phi_bounds + varsigma_bounds
        
        # Optimization
        try:
            result = optimize.minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
            estimated_params = result.x
        except Exception:
            try:
                bounds_de = list(zip(*bounds))
                result = optimize.differential_evolution(
                    objective,
                    bounds_de,
                    maxiter=max_iter//10,
                    tol=1e-4
                )
                estimated_params = result.x
            except Exception:
                result = optimize.minimize(
                    objective,
                    initial_params,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter}
                )
                estimated_params = result.x
        
        # Extract estimated parameters
        phi_est = estimated_params[:J]
        varsigma_est = estimated_params[J:]
        
        # Calculate c and pis from varsigmas
        c_est = np.sum(varsigma_est)
        pi_est = varsigma_est / c_est
        
        return {
            'phis': phi_est,
            'varsigmas': varsigma_est,
            'c': c_est,
            'pis': pi_est,
            'alpha': alpha,
            'beta': beta,
            'J': J
        }


    def estimate_nls(self, X: np.ndarray, alpha: float, beta: float, J: int, 
                lambda_vals: Optional[np.ndarray] = None, 
                u0: float = 0.01, max_iter: int = 200) -> Dict[str, Any]:
        """
        Estimates parameters of a discrete mixture using Nonlinear Least Squares
        with varsigma_j = c * pi_j parameterization for better numerical stability.
        
        This implements the reparameterization strategy from the paper:
        - Uses varsigma_j instead of (c, pi_j) to eliminate constraints
        - Employs multi-stage optimization (grid search + Gauss-Newton + Levenberg-Marquardt)
        - Computes analytical Jacobian for efficiency
        
        Parameters:
        -----------
        X : np.ndarray
            Time series data
        alpha : float
            Estimated stability parameter
        J : int
            Number of mixture components
        lambda_vals : np.ndarray, optional
            Lambda values for moment conditions
        u0 : float
            Small parameter for characteristic function evaluation
        max_iter : int
            Maximum iterations for optimization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing estimated parameters with both parameterizations
        """
        if lambda_vals is None:
            # Ensure we have enough moment conditions for identification
            lambda_vals = np.linspace(0.5, 1.5, 2*J + 5)
        
        # Calculate empirical moment function
        empirical_g = []
        for lam in lambda_vals:
            psi = self.estimate_ecf(X, u0, lam * u0)
            log_abs_psi = np.log(np.abs(psi) + 1e-15)  # Avoid log(0)
            g_lambda = log_abs_psi / (np.abs(u0)**alpha)
            empirical_g.append(g_lambda)
        
        empirical_g = np.array(empirical_g)
        
        # Multi-stage optimization approach
        
        # Stage 1: Grid initialization for phi parameters
        # initial_params = self._grid_search_initialization_nls(J, empirical_g, lambda_vals, alpha)
        # print("grid params:", initial_params)
        
        phi_init = np.linspace(0.5, 0.9, J)
        varsigma_init = np.linspace(1.0, 0.7, J) # np.ones(J) / J
        
        initial_params = np.concatenate([varsigma_init, phi_init])
        
        # Stage 2: Gauss-Newton optimization
        try:
            gn_params = self._gauss_newton_step_nls(
                initial_params, empirical_g, lambda_vals, alpha, max_iter // 2
            )
        except Exception:
            gn_params = initial_params
        
        # Stage 3: Levenberg-Marquardt regularization
        try:
            final_params = self._levenberg_marquardt_step_nls(
                gn_params, empirical_g, lambda_vals, alpha, max_iter // 2
            )
        except Exception:
            final_params = gn_params
        
        # Extract results and convert between parameterizations
        varsigmas = final_params[:J]
        phis = final_params[J:]
        
        # Calculate c and pis from varsigmas
        c_est = np.sum(varsigmas)
        pi_est = varsigmas / c_est
        
        return {
            'phis': phis,
            'varsigmas': varsigmas,
            'c': c_est,
            'pis': pi_est,
            'alpha': alpha,
            'beta': beta,
            'J': J
        }


    def _grid_search_initialization_nls(self, J: int, empirical_g: np.ndarray,
                                    lambda_vals: np.ndarray, alpha: float) -> np.ndarray:
        """
        Grid search initialization for NLS optimization.
        Uses coarse grid to find reasonable starting values for phi parameters.
        """
        phi_grid = np.linspace(0.1, 0.9, 15)
        best_obj = np.inf
        best_params = None
        
        # Try different combinations (limited for computational efficiency)
        np.random.seed(42)  # For reproducible results
        n_trials = min(100, 15**J if J <= 3 else 100)
        
        for _ in range(n_trials):
            # Random sample from grid
            trial_phis = np.random.choice(phi_grid, J, replace=False)
            trial_phis = np.sort(trial_phis)  # Sort for consistency
            
            # Initialize varsigmas with equal weights
            trial_varsigmas = np.ones(J) / J
            trial_params = np.concatenate([trial_varsigmas, trial_phis])
            
            # Evaluate objective
            obj_val = self._objective_varsigma_nls(trial_params, empirical_g, lambda_vals, alpha)
            
            if obj_val < best_obj:
                best_obj = obj_val
                best_params = trial_params
        
        # Fallback if no good parameters found
        if best_params is None:
            default_varsigmas = np.ones(J) / J
            default_phis = np.linspace(0.2, 0.8, J)
            best_params = np.concatenate([default_varsigmas, default_phis])
        
        return best_params


    def _objective_varsigma_nls(self, params: np.ndarray, empirical_g: np.ndarray,
                            lambda_vals: np.ndarray, alpha: float) -> float:
        """Objective function for NLS optimization using varsigma parameterization."""
        J = len(params) // 2
        varsigmas = params[:J]
        phis = params[J:]
        
        # Check parameter validity
        if np.any(varsigmas <= 0) or np.any(np.abs(phis) >= 1):
            return 1e12
        
        # Compute theoretical moment function
        theoretical_g = self._theoretical_moment_function_varsigma_nls(
            lambda_vals, varsigmas, phis, alpha
        )
        
        return np.sum((empirical_g - theoretical_g)**2)


    def _theoretical_moment_function_varsigma_nls(self, lambda_vals: np.ndarray,
                                                varsigmas: np.ndarray, phis: np.ndarray,
                                                alpha: float) -> np.ndarray:
        """
        Theoretical moment function using varsigma parameterization for NLS.
        
        g(λ; θ*) = -∑_{j=1}^J varsigma_j^α * [|1 + φ_j*λ|^α / (1-|φ_j|^α) + |λ|^α]
        """
        J = len(varsigmas)
        g_lambda = np.zeros_like(lambda_vals, dtype=float)
        
        for i, lam in enumerate(lambda_vals):
            sum_term = 0
            for j in range(J):
                term1 = np.abs(1 + phis[j] * lam)**alpha / (1 - np.abs(phis[j])**alpha)
                term2 = np.abs(lam)**alpha
                sum_term += varsigmas[j]**alpha * (term1 + term2)
            
            g_lambda[i] = -sum_term
        
        return g_lambda


    def _compute_jacobian_analytical_nls(self, params: np.ndarray, lambda_vals: np.ndarray,
                                        alpha: float) -> np.ndarray:
        """
        Compute analytical Jacobian matrix for Gauss-Newton optimization.
        
        Jacobian elements:
        ∂g/∂varsigma_j = -α * varsigma_j^(α-1) * [|1 + φ_j*λ|^α/(1-|φ_j|^α) + |λ|^α]
        ∂g/∂φ_j = -varsigma_j^α * [∂/∂φ_j(|1 + φ_j*λ|^α/(1-|φ_j|^α))]
        """
        J = len(params) // 2
        varsigmas = params[:J]
        phis = params[J:]
        
        M = len(lambda_vals)  # Number of moment conditions
        jacobian = np.zeros((M, 2*J))
        
        for i, lam in enumerate(lambda_vals):
            for j in range(J):
                # Derivative w.r.t. varsigma_j
                term1 = np.abs(1 + phis[j] * lam)**alpha / (1 - np.abs(phis[j])**alpha)
                term2 = np.abs(lam)**alpha
                jacobian[i, j] = -alpha * varsigmas[j]**(alpha-1) * (term1 + term2)
                
                # Derivative w.r.t. phi_j (using finite differences for stability)
                if np.abs(phis[j]) < 0.95:
                    delta = 1e-6
                    phi_plus = phis[j] + delta
                    phi_minus = phis[j] - delta
                    
                    # Ensure we stay in bounds
                    phi_plus = min(phi_plus, 0.99)
                    phi_minus = max(phi_minus, -0.99)
                    
                    term_plus = np.abs(1 + phi_plus * lam)**alpha / (1 - np.abs(phi_plus)**alpha)
                    term_minus = np.abs(1 + phi_minus * lam)**alpha / (1 - np.abs(phi_minus)**alpha)
                    
                    jacobian[i, J + j] = -varsigmas[j]**alpha * (term_plus - term_minus) / (phi_plus - phi_minus)
                else:
                    jacobian[i, J + j] = 0  # Avoid numerical issues at boundary
        
        return jacobian


    def _gauss_newton_step_nls(self, initial_params: np.ndarray, empirical_g: np.ndarray,
                            lambda_vals: np.ndarray, alpha: float, max_iter: int = 50) -> np.ndarray:
        """
        Gauss-Newton optimization step.
        
        Update formula: θ^(k+1) = θ^(k) - (J_k^T J_k)^(-1) J_k^T r_k
        """
        params = initial_params.copy()
        J = len(params) // 2
        
        for iteration in range(max_iter):
            # Compute residuals
            theoretical_g = self._theoretical_moment_function_varsigma_nls(
                lambda_vals, params[:J], params[J:], alpha
            )
            residuals = empirical_g - theoretical_g
            
            # Check convergence
            if np.linalg.norm(residuals) < 1e-8:
                break
            
            try:
                # Compute Jacobian and update
                jacobian = self._compute_jacobian_analytical_nls(params, lambda_vals, alpha)
                JtJ = jacobian.T @ jacobian
                Jtr = jacobian.T @ residuals
                
                # Solve normal equations
                update = np.linalg.solve(JtJ, Jtr)
                new_params = params + update
                
                # Project onto feasible region
                new_params = self._project_parameters_nls(new_params, J)
                
                # Check if objective improved
                new_obj = self._objective_varsigma_nls(new_params, empirical_g, lambda_vals, alpha)
                old_obj = self._objective_varsigma_nls(params, empirical_g, lambda_vals, alpha)
                
                if new_obj < old_obj:
                    params = new_params
                else:
                    # Try reduced step sizes
                    for step_size in [0.5, 0.1, 0.01]:
                        test_params = params + step_size * update
                        test_params = self._project_parameters_nls(test_params, J)
                        test_obj = self._objective_varsigma_nls(test_params, empirical_g, lambda_vals, alpha)
                        
                        if test_obj < old_obj:
                            params = test_params
                            break
                    else:
                        break  # No improvement possible
            
            except np.linalg.LinAlgError:
                break
        
        return params


    def _levenberg_marquardt_step_nls(self, initial_params: np.ndarray, empirical_g: np.ndarray,
                                    lambda_vals: np.ndarray, alpha: float, max_iter: int = 50) -> np.ndarray:
        """
        Levenberg-Marquardt optimization with adaptive damping.
        
        Update formula: θ^(k+1) = θ^(k) - (J_k^T J_k + μ_k I)^(-1) J_k^T r_k
        """
        params = initial_params.copy()
        J = len(params) // 2
        mu = 1e-3  # Initial damping parameter
        
        for iteration in range(max_iter):
            # Compute residuals
            theoretical_g = self._theoretical_moment_function_varsigma_nls(
                lambda_vals, params[:J], params[J:], alpha
            )
            residuals = empirical_g - theoretical_g
            
            # Check convergence
            if np.linalg.norm(residuals) < 1e-8:
                break
            
            try:
                # Compute Jacobian and update with damping
                jacobian = self._compute_jacobian_analytical_nls(params, lambda_vals, alpha)
                JtJ = jacobian.T @ jacobian
                Jtr = jacobian.T @ residuals
                
                # Add damping
                update = np.linalg.solve(JtJ + mu * np.eye(len(params)), Jtr)
                new_params = params + update
                
                # Project onto feasible region
                new_params = self._project_parameters_nls(new_params, J)
                
                # Check improvement
                new_obj = self._objective_varsigma_nls(new_params, empirical_g, lambda_vals, alpha)
                old_obj = self._objective_varsigma_nls(params, empirical_g, lambda_vals, alpha)
                
                if new_obj < old_obj:
                    # Accept update and decrease damping
                    params = new_params
                    mu = max(mu / 10, 1e-12)
                else:
                    # Reject update and increase damping
                    mu = min(mu * 10, 1e6)
            
            except np.linalg.LinAlgError:
                mu = min(mu * 10, 1e6)
        
        return params


    def _project_parameters_nls(self, params: np.ndarray, J: int) -> np.ndarray:
        """Project parameters onto feasible region."""
        projected = params.copy()
        
        # Project varsigmas to positive values
        projected[:J] = np.maximum(projected[:J], 1e-6)
        
        # Project phis to (-1, 1)
        projected[J:] = np.clip(projected[J:], -0.99, 0.99)
        
        return projected



    # --- Main Estimation function ---

    def sequential_estimation(self, X: np.ndarray, params_dict: Optional[Dict[str, Any]] = None, 
                            lambda_vals: Optional[np.ndarray] = None, u0: float = 0.01,
                            step: float = 0.01, m: float = 1.0,
                            use_robust_scale: bool = True, verbose: bool = False,
                            method: str = 'md') -> Dict[str, Any]:
        """
        Performs complete sequential estimation using characteristic function matching.
        Step 1: Estimate alpha and beta using Kogon-Williams method
        Step 2: Estimate mixture parameters using complete characteristic function
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series
        params_dict : dict or None
            Model-specific parameters
        lambda_vals : array_like or None
            Not used (kept for compatibility)
        u0 : float
            Not used (kept for compatibility)
        step : float
            Step size for alpha/beta estimation
        m : float
            Maximum u value for alpha/beta estimation
        use_robust_scale : bool
            If True, uses robust normalization for alpha and beta estimation
        verbose : bool
            If True, displays progress messages
        method : str
            Estimation method: 'nls' for NLS or 'md' for minimum distance
        
        Returns:
        ---------
        dict
            Dictionary containing all estimated parameters
        """
        # Step 1: Estimate alpha and beta simultaneously using Kogon-Williams method
        alpha_est, beta_est = self.estimate_alpha_beta(X, step=step, m=m, 
                                                    use_robust_scale=use_robust_scale, 
                                                    verbose=verbose)
        
        if verbose:
            print(f"Alpha estimation: {alpha_est:.4f}")
            print(f"Beta estimation: {beta_est:.4f}")
        
        # Step 2: Estimate mixture parameters using the estimated alpha and beta
        mixture_params = self.estimate_mixture_params(X, alpha_est, beta_est, params_dict, lambda_vals, u0, method)
        
        if verbose:
            if self.model_type == 'discrete':
                print(f"Estimation of phis: {mixture_params['phis']}")
                print(f"Estimation of varsigmas: {mixture_params['varsigmas']}")
                print(f"Implied c: {mixture_params['c']:.4f}")
                print(f"Implied pis: {mixture_params['pis']}")
            else:
                print(f"Estimation of c: {mixture_params['c']:.4f}")
                print(f"Estimation of a: {mixture_params['a']:.4f}")
                print(f"Estimation of b: {mixture_params['b']:.4f}")
            print(f"Alpha and beta used: α={alpha_est:.4f}, β={beta_est:.4f}")
        
        # Ensure both parameterizations are available for discrete model
        if self.model_type == 'discrete':
            if 'varsigmas' in mixture_params and ('c' not in mixture_params or 'pis' not in mixture_params):
                varsigmas = mixture_params['varsigmas']
                mixture_params['c'] = np.sum(varsigmas)
                mixture_params['pis'] = varsigmas / mixture_params['c']
            elif 'c' in mixture_params and 'pis' in mixture_params and 'varsigmas' not in mixture_params:
                mixture_params['varsigmas'] = mixture_params['c'] * mixture_params['pis']
        
        return mixture_params


    def evaluate_estimation(self, true_params: Dict[str, Any], 
                            estimated_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates estimation performance according to the model type.
        Dispatches to the appropriate evaluation method.
        
        Parameters:
        -----------
        true_params : dict
            True parameters
        estimated_params : dict
            Estimated parameters
            
        Returns:
        --------
        dict
            Dictionary containing estimation errors
        """
        if self.model_type == 'discrete':
            return self.evaluate_estimation_discrete(true_params, estimated_params)
        else:
            return self.evaluate_estimation_beta(true_params, estimated_params)


    def evaluate_estimation_discrete(self, true_params: Dict[str, Any], 
                                    estimated_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates estimation performance for a discrete mixture.
        Modified to handle varsigma parameterization.
        """
        # Get J from parameters or from length of phis
        if 'J' in true_params:
            J = true_params['J']
        elif 'phis' in true_params:
            J = len(true_params['phis'])
        else:
            raise KeyError("Cannot determine J from parameters")
        
        # Absolute errors for scalar parameters
        alpha_error = np.abs(estimated_params['alpha'] - true_params['alpha'])
        beta_error = np.abs(estimated_params['beta'] - true_params['beta'])
        c_error = np.abs(estimated_params['c'] - true_params['c'])
        
        # For phis and varsigmas, association is not unique
        true_phis = true_params['phis']
        true_varsigmas = true_params.get('varsigmas', true_params['pis'] * true_params['c'])
        est_phis = estimated_params['phis']
        est_varsigmas = estimated_params['varsigmas']
        
        # Calculate all possible permutations
        min_phi_error = float('inf')
        min_varsigma_error = float('inf')
        min_pi_error = float('inf')  # Still calculate pi errors for reporting
        
        for p in permutations(range(J)):
            perm_est_phis = est_phis[list(p)]
            perm_est_varsigmas = est_varsigmas[list(p)]
            
            # Calculate errors
            phi_err = np.mean(np.abs(perm_est_phis - true_phis))
            varsigma_err = np.mean(np.abs(perm_est_varsigmas - true_varsigmas))
            
            # Calculate pi error for reporting
            true_pis = true_params.get('pis', true_varsigmas / np.sum(true_varsigmas))
            est_pis = estimated_params.get('pis', perm_est_varsigmas / np.sum(perm_est_varsigmas))
            pi_err = np.mean(np.abs(est_pis - true_pis))
            
            if phi_err < min_phi_error:
                min_phi_error = phi_err
            
            if varsigma_err < min_varsigma_error:
                min_varsigma_error = varsigma_err
                
            if pi_err < min_pi_error:
                min_pi_error = pi_err
        
        return {
            'alpha_error': alpha_error,
            'beta_error': beta_error,
            'c_error': c_error,
            'phi_error': min_phi_error,
            'varsigma_error': min_varsigma_error,
            'pi_error': min_pi_error  # Keep pi error for backward compatibility
        }

    
    def evaluate_estimation_beta(self, true_params: Dict[str, Any], 
                                estimated_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates estimation performance for a beta mixture.
        """
        # Erreurs absolues
        alpha_error = np.abs(estimated_params['alpha'] - true_params['alpha'])
        beta_error = np.abs(estimated_params['beta'] - true_params['beta'])
        c_error = np.abs(estimated_params['c'] - true_params['c'])
        a_error = np.abs(estimated_params['a'] - true_params['a'])
        b_error = np.abs(estimated_params['b'] - true_params['b'])
        
        # Erreurs relatives
        alpha_rel_error = alpha_error / true_params['alpha'] if true_params['alpha'] != 0 else np.inf
        beta_rel_error = beta_error / true_params['beta'] if true_params['beta'] != 0 else np.inf
        c_rel_error = c_error / true_params['c']
        a_rel_error = a_error / true_params['a']
        b_rel_error = b_error / true_params['b']
        
        return {
            'alpha_error': alpha_error,
            'beta_error': beta_error,
            'c_error': c_error,
            'a_error': a_error,
            'b_error': b_error,
            'alpha_rel_error': alpha_rel_error,
            'beta_rel_error': beta_rel_error,
            'c_rel_error': c_rel_error,
            'a_rel_error': a_rel_error,
            'b_rel_error': b_rel_error
        }
        

    # --- Main Inference function ---


    def inference(self, X: np.ndarray, estimated_params: Optional[Dict[str, Any]] = None,
                alpha: float = 0.05, method: str = 'md', 
                step: float = 0.01, m: float = 1.0) -> pd.DataFrame:
        """
        Calculates inference statistics for the estimated parameters from sequential estimation.

        Returns a DataFrame with, for each parameter, the estimate, its standard
        error, the t-statistic and a (1 - alpha) confidence interval.
        """
        if estimated_params is None:
            raise ValueError("estimated_params must be provided. Run sequential_estimation first.")
        
        # Extract basic parameters
        alpha_est = estimated_params['alpha']
        beta_est = estimated_params['beta']
        
        # Initialize results list
        results_list = []
        
        # Confidence level label
        ci_level = int(100*(1-alpha))
        
        # 1. INFERENCE FOR ALPHA AND BETA (from Kogon-Williams estimation)
        try:
            alpha_se, beta_se = self._compute_alpha_beta_standard_errors(
                X, alpha_est, beta_est, step, m
            )
        except Exception as e:
            print(f"Warning: Could not compute standard errors for alpha/beta: {e}")
            alpha_se = 0.1  # Default fallback
            beta_se = 0.2   # Default fallback
        
        # Alpha statistics
        alpha_t = alpha_est / alpha_se if alpha_se > 0 else np.nan
        alpha_p = 2 * (1 - stats.norm.cdf(abs(alpha_t))) if not np.isnan(alpha_t) else np.nan
        z_critical = stats.norm.ppf(1 - alpha/2)
        alpha_lower = alpha_est - z_critical * alpha_se
        alpha_upper = alpha_est + z_critical * alpha_se
        
        results_list.append({
            'Parameter': 'alpha',
            'Estimate': alpha_est,
            'Std. Error': alpha_se,
            't-statistic': alpha_t,
            'p-value': alpha_p,
            f'CI {ci_level}% Lower': alpha_lower,
            f'CI {ci_level}% Upper': alpha_upper
        })
        
        # Beta statistics
        beta_t = beta_est / beta_se if beta_se > 0 else np.nan
        beta_p = 2 * (1 - stats.norm.cdf(abs(beta_t))) if not np.isnan(beta_t) else np.nan
        beta_lower = beta_est - z_critical * beta_se
        beta_upper = beta_est + z_critical * beta_se
        
        results_list.append({
            'Parameter': 'beta',
            'Estimate': beta_est,
            'Std. Error': beta_se,
            't-statistic': beta_t,
            'p-value': beta_p,
            f'CI {ci_level}% Lower': beta_lower,
            f'CI {ci_level}% Upper': beta_upper
        })
        
        # 2. INFERENCE FOR MIXTURE PARAMETERS
        if self.model_type == 'discrete':
            mixture_results = self._compute_discrete_mixture_inference(
                X, estimated_params, alpha, method, ci_level
            )
        else:  # continuous
            mixture_results = self._compute_continuous_mixture_inference(
                X, estimated_params, alpha, ci_level
            )
        
        results_list.extend(mixture_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)
        
        return results_df


    def _compute_alpha_beta_standard_errors(self, X: np.ndarray, alpha_est: float, 
                                        beta_est: float, step: float = 0.01, 
                                        m: float = 1.0) -> Tuple[float, float]:
        """
        Computes standard errors for alpha and beta estimates using bootstrap or 
        asymptotic approximation based on the regression approach.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series data
        alpha_est : float
            Estimated alpha parameter
        beta_est : float
            Estimated beta parameter
        step : float
            Step size for characteristic function evaluation
        m : float
            Maximum u value for characteristic function evaluation
            
        Returns:
        --------
        tuple
            (alpha_standard_error, beta_standard_error)
        """
        n = len(X)
        
        # Robust scale estimation (same as in estimate_alpha_beta)
        q75, q25 = np.quantile(X, [0.75, 0.25])
        iqr = q75 - q25
        sigma0 = iqr / 2 if iqr > 1e-10 else np.std(X)
        X_norm = (X - np.median(X)) / sigma0
        
        # Create u grid
        u = np.arange(step, m + step, step)
        
        # Calculate empirical characteristic function
        ecf = self.estimate_ecf(X_norm, u)
        valid_mask = (np.abs(ecf) > 1e-10) & (np.abs(ecf) < 0.999)
        
        if np.sum(valid_mask) < 3:
            # Not enough points for reliable inference
            return 0.1, 0.2
        
        u_valid = u[valid_mask]
        ecf_valid = ecf[valid_mask]
        
        # For alpha: use regression variance
        try:
            x = np.log(-np.log(np.abs(ecf_valid) + 1e-10))
            y = np.log(np.abs(u_valid) + 1e-10)
            
            # OLS regression for alpha
            X_mat = np.column_stack([np.ones(len(y)), y])
            beta_ols = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ x)
            
            # Residuals and variance estimate
            residuals = x - X_mat @ beta_ols
            mse = np.sum(residuals**2) / (len(residuals) - 2)
            
            # Variance-covariance matrix
            vcv = mse * np.linalg.inv(X_mat.T @ X_mat)
            alpha_se = np.sqrt(vcv[1, 1])  # Variance of slope coefficient
            
            # For beta: approximate using the phase regression
            # This is more complex, so we use a simpler approximation
            beta_se = 0.2 / np.sqrt(n/100)  # Rough approximation based on sample size
            
            return max(alpha_se, 0.01), max(beta_se, 0.01)
            
        except Exception:
            # Fallback to simple approximations
            alpha_se = 0.2 / np.sqrt(n/100)
            beta_se = 0.3 / np.sqrt(n/100)
            return alpha_se, beta_se


    def _compute_discrete_mixture_inference(self, X: np.ndarray, estimated_params: Dict[str, Any],
                                        alpha: float, method: str, ci_level: int) -> List[Dict[str, Any]]:
        """
        Computes inference statistics for discrete mixture parameters.
        """
        results = []
        
        # Extract parameters
        J = estimated_params.get('J', len(estimated_params['phis']))
        phis = estimated_params['phis']
        varsigmas = estimated_params['varsigmas'] 
        c_est = estimated_params['c']
        pis = estimated_params['pis']
        alpha_est = estimated_params['alpha']
        beta_est = estimated_params['beta']
        
        # Compute variance-covariance matrix based on method
        try:
            if method == 'md':
                vcv_matrix = self._compute_vcv_matrix_md(
                    X, phis, varsigmas, alpha_est, beta_est
                )
            else:  # nls
                vcv_matrix = self._compute_vcv_matrix_nls(
                    X, phis, varsigmas, alpha_est, beta_est
                )
                
            # Extract standard errors
            n_params = len(phis) + len(varsigmas)  # phi_j + varsigma_j
            
            if vcv_matrix.shape[0] >= n_params:
                std_errors = np.sqrt(np.diag(vcv_matrix[:n_params]))
            else:
                # Fallback to rough approximations
                std_errors = np.concatenate([
                    0.1 * np.abs(phis) + 0.01,  # For phis
                    0.1 * np.abs(varsigmas) + 0.01  # For varsigmas
                ])
                
        except Exception as e:
            print(f"Warning: Could not compute mixture parameter standard errors: {e}")
            # Use rough approximations
            std_errors = np.concatenate([
                0.1 * np.abs(phis) + 0.01,  # For phis
                0.1 * np.abs(varsigmas) + 0.01  # For varsigmas
            ])
        
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Statistics for phi parameters
        for j in range(J):
            phi_j = phis[j]
            se_j = std_errors[j] if j < len(std_errors) else 0.1
            
            t_stat = phi_j / se_j if se_j > 0 else np.nan
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            lower_ci = phi_j - z_critical * se_j
            upper_ci = phi_j + z_critical * se_j
            
            results.append({
                'Parameter': f'phi_{j+1}',
                'Estimate': phi_j,
                'Std. Error': se_j,
                't-statistic': t_stat,
                'p-value': p_val,
                f'CI {ci_level}% Lower': lower_ci,
                f'CI {ci_level}% Upper': upper_ci
            })
        
        # Statistics for varsigma parameters
        for j in range(J):
            varsigma_j = varsigmas[j]
            se_j = std_errors[J + j] if J + j < len(std_errors) else 0.1 * varsigma_j
            
            t_stat = varsigma_j / se_j if se_j > 0 else np.nan
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            lower_ci = varsigma_j - z_critical * se_j
            upper_ci = varsigma_j + z_critical * se_j
            
            results.append({
                'Parameter': f'varsigma_{j+1}',
                'Estimate': varsigma_j,
                'Std. Error': se_j,
                't-statistic': t_stat,
                'p-value': p_val,
                f'CI {ci_level}% Lower': lower_ci,
                f'CI {ci_level}% Upper': upper_ci
            })
        
        # Statistics for c parameter
        c_se = np.sqrt(np.sum(std_errors[J:]**2))  # c = sum(varsigmas)
        c_t = c_est / c_se if c_se > 0 else np.nan
        c_p = 2 * (1 - stats.norm.cdf(abs(c_t))) if not np.isnan(c_t) else np.nan
        c_lower = c_est - z_critical * c_se
        c_upper = c_est + z_critical * c_se
        
        results.append({
            'Parameter': 'c',
            'Estimate': c_est,
            'Std. Error': c_se,
            't-statistic': c_t,
            'p-value': c_p,
            f'CI {ci_level}% Lower': c_lower,
            f'CI {ci_level}% Upper': c_upper
        })
        
        # Statistics for pi parameters (derived from varsigmas)
        for j in range(J):
            pi_j = pis[j]
            
            # Delta method for pi_j = varsigma_j / c
            if J + j < len(std_errors):
                varsigma_se = std_errors[J + j]
                # Simplified approximation (ignoring covariances)
                pi_se = np.sqrt((1/c_est)**2 * varsigma_se**2 + (varsigmas[j]/c_est**2)**2 * c_se**2)
            else:
                pi_se = 0.1 * pi_j
            
            pi_t = pi_j / pi_se if pi_se > 0 else np.nan
            pi_p = 2 * (1 - stats.norm.cdf(abs(pi_t))) if not np.isnan(pi_t) else np.nan
            pi_lower = pi_j - z_critical * pi_se
            pi_upper = pi_j + z_critical * pi_se
            
            results.append({
                'Parameter': f'pi_{j+1}',
                'Estimate': pi_j,
                'Std. Error': pi_se,
                't-statistic': pi_t,
                'p-value': pi_p,
                f'CI {ci_level}% Lower': pi_lower,
                f'CI {ci_level}% Upper': pi_upper
            })
        
        return results


    def _compute_continuous_mixture_inference(self, X: np.ndarray, estimated_params: Dict[str, Any],
                                            alpha: float, ci_level: int) -> List[Dict[str, Any]]:
        """
        Computes inference statistics for continuous mixture (beta) parameters.
        """
        results = []
        
        # Extract parameters
        c_est = estimated_params['c']
        a_est = estimated_params['a']
        b_est = estimated_params['b']
        alpha_est = estimated_params['alpha']
        beta_est = estimated_params['beta']
        
        # Compute variance-covariance matrix
        try:
            vcv_matrix = self._compute_vcv_matrix_beta(X, c_est, a_est, b_est, alpha_est, beta_est)
            std_errors = np.sqrt(np.diag(vcv_matrix))
        except Exception as e:
            print(f"Warning: Could not compute beta mixture standard errors: {e}")
            # Use rough approximations
            std_errors = np.array([
                0.1 * c_est + 0.01,    # c
                0.2 * a_est + 0.05,    # a  
                0.2 * b_est + 0.05     # b
            ])
        
        z_critical = stats.norm.ppf(1 - alpha/2)
        params = [c_est, a_est, b_est]
        param_names = ['c', 'a', 'b']
        
        # Statistics for c, a, b parameters
        for i, (param_name, param_value) in enumerate(zip(param_names, params)):
            se = std_errors[i] if i < len(std_errors) else 0.1 * abs(param_value)
            
            t_stat = param_value / se if se > 0 else np.nan
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            lower_ci = param_value - z_critical * se
            upper_ci = param_value + z_critical * se
            
            results.append({
                'Parameter': param_name,
                'Estimate': param_value,
                'Std. Error': se,
                't-statistic': t_stat,
                'p-value': p_val,
                f'CI {ci_level}% Lower': lower_ci,
                f'CI {ci_level}% Upper': upper_ci
            })
        
        # Derived parameters for beta distribution
        mean_phi = a_est / (a_est + b_est)
        # variance_phi = (a_est * b_est) / ((a_est + b_est)**2 * (a_est + b_est + 1))
        
        # Delta method for mean_phi = a/(a+b)
        if len(std_errors) >= 2:
            a_se = std_errors[1]  # a
            b_se = std_errors[2]  # b
            # ∂mean/∂a = b/(a+b)², ∂mean/∂b = -a/(a+b)²
            dmean_da = b_est / (a_est + b_est)**2
            dmean_db = -a_est / (a_est + b_est)**2
            mean_phi_se = np.sqrt(dmean_da**2 * a_se**2 + dmean_db**2 * b_se**2)
        else:
            mean_phi_se = 0.05
        
        # Statistics for mean_phi
        mean_phi_t = mean_phi / mean_phi_se if mean_phi_se > 0 else np.nan
        mean_phi_p = 2 * (1 - stats.norm.cdf(abs(mean_phi_t))) if not np.isnan(mean_phi_t) else np.nan
        mean_phi_lower = mean_phi - z_critical * mean_phi_se
        mean_phi_upper = mean_phi + z_critical * mean_phi_se
        
        results.append({
            'Parameter': 'mean_phi',
            'Estimate': mean_phi,
            'Std. Error': mean_phi_se,
            't-statistic': mean_phi_t,
            'p-value': mean_phi_p,
            f'CI {ci_level}% Lower': mean_phi_lower,
            f'CI {ci_level}% Upper': mean_phi_upper
        })
        
        return results


    def _compute_vcv_matrix_md(self, X: np.ndarray, phis: np.ndarray, varsigmas: np.ndarray,
                            alpha: float, beta: float) -> np.ndarray:
        """
        Computes variance-covariance matrix for discrete mixture using minimum distance method.
        Based on the Hessian of the characteristic function objective.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series data
        phis : numpy.ndarray
            Estimated phi parameters
        varsigmas : numpy.ndarray
            Estimated varsigma parameters
        alpha : float
            Estimated alpha parameter (treated as fixed)
        beta : float
            Estimated beta parameter (treated as fixed)
            
        Returns:
        --------
        numpy.ndarray
            Variance-covariance matrix for [phi_1, ..., phi_J, varsigma_1, ..., varsigma_J]
        """
        n = len(X)
        J = len(phis)
        n_params = 2 * J  # phi_j + varsigma_j
        
        # Define grid for (u, v) - same as in estimation
        u_grid = np.linspace(-1.5, 1.5, 15)
        v_grid = np.linspace(-1.5, 1.5, 15)
        
        # Combine parameters for objective function
        theta = np.concatenate([phis, varsigmas])
        
        # Compute Hessian numerically
        hessian = np.zeros((n_params, n_params))
        epsilon = 1e-5
        
        try:
            # Compute objective at current point
            f0 = self._objective_md_inference(theta, X, alpha, beta, u_grid, v_grid)
            
            # Compute second derivatives
            for i in range(n_params):
                for j in range(i, n_params):
                    # Create perturbation vectors
                    ei = np.zeros(n_params)
                    ej = np.zeros(n_params)
                    ei[i] = epsilon
                    ej[j] = epsilon
                    
                    if i == j:
                        # Diagonal element: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
                        f_plus = self._objective_md_inference(theta + ei, X, alpha, beta, u_grid, v_grid)
                        f_minus = self._objective_md_inference(theta - ei, X, alpha, beta, u_grid, v_grid)
                        hessian[i, j] = (f_plus - 2*f0 + f_minus) / (epsilon**2)
                    else:
                        # Off-diagonal element: mixed derivative
                        f_pp = self._objective_md_inference(theta + ei + ej, X, alpha, beta, u_grid, v_grid)
                        f_pm = self._objective_md_inference(theta + ei - ej, X, alpha, beta, u_grid, v_grid)
                        f_mp = self._objective_md_inference(theta - ei + ej, X, alpha, beta, u_grid, v_grid)
                        f_mm = self._objective_md_inference(theta - ei - ej, X, alpha, beta, u_grid, v_grid)
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
                        hessian[j, i] = hessian[i, j]  # Symmetry
            
            # Regularize if needed
            eigenvals = np.linalg.eigvals(hessian)
            if np.min(eigenvals) <= 0:
                # Add small positive value to diagonal
                hessian += np.abs(np.min(eigenvals)) * 1.1 * np.eye(n_params)
            
            # Variance-covariance matrix (inverse of Hessian, scaled by sample size)
            vcv_matrix = np.linalg.inv(hessian) / n
            
            return vcv_matrix
            
        except Exception as e:
            print(f"Warning: Hessian computation failed for MD method: {e}")
            # Return diagonal matrix with rough approximations
            variances = np.concatenate([
                (0.1 * np.abs(phis) + 0.01)**2,      # For phis
                (0.1 * np.abs(varsigmas) + 0.01)**2  # For varsigmas
            ])
            return np.diag(variances)


    def _objective_md_inference(self, theta: np.ndarray, X: np.ndarray, alpha: float, beta: float,
                            u_grid: np.ndarray, v_grid: np.ndarray) -> float:
        """
        Objective function for minimum distance estimation (for Hessian computation).
        """
        J = len(theta) // 2
        phis = theta[:J]
        varsigmas = theta[J:]
        
        # Check parameter validity
        if np.any(np.abs(phis) >= 1) or np.any(varsigmas <= 0):
            return 1e10
        
        diff_sum = 0
        count = 0
        
        for u in u_grid:
            for v in v_grid:
                try:
                    # Theoretical characteristic function
                    cf_theo = self.char_function_discrete(u, v, theta, alpha, beta)
                    
                    # Empirical characteristic function
                    cf_emp = np.mean(np.exp(1j * (u * X[1:] + v * X[:-1])))
                    
                    # Squared difference
                    diff_sum += np.abs(cf_emp - cf_theo)**2
                    count += 1
                except Exception:
                    diff_sum += 1e6  # Penalty for numerical issues
                    count += 1
        
        return diff_sum / count if count > 0 else 1e10


    def _compute_vcv_matrix_nls(self, X: np.ndarray, phis: np.ndarray, varsigmas: np.ndarray,
                            alpha: float, beta: float) -> np.ndarray:
        """
        Computes variance-covariance matrix for discrete mixture using NLS method.
        Based on the Jacobian of the moment conditions.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series data
        phis : numpy.ndarray
            Estimated phi parameters
        varsigmas : numpy.ndarray
            Estimated varsigma parameters
        alpha : float
            Estimated alpha parameter (treated as fixed)
        beta : float
            Estimated beta parameter (treated as fixed)
            
        Returns:
        --------
        numpy.ndarray
            Variance-covariance matrix for [phi_1, ..., phi_J, varsigma_1, ..., varsigma_J]
        """
        n = len(X)
        J = len(phis)
        n_params = 2 * J
        
        # Define lambda values for moment conditions
        lambda_vals = np.linspace(0.5, 1.5, 2*J + 5)
        u0 = 0.01
        
        try:
            # Calculate empirical moment function
            empirical_g = []
            for lam in lambda_vals:
                psi = self.estimate_ecf(X, u0, lam * u0)
                log_abs_psi = np.log(np.abs(psi) + 1e-15)
                g_lambda = log_abs_psi / (np.abs(u0)**alpha)
                empirical_g.append(g_lambda)
            
            empirical_g = np.array(empirical_g)
            
            # Compute Jacobian of moment conditions
            jacobian = self._compute_jacobian_nls_inference(
                varsigmas, phis, lambda_vals, alpha
            )
            
            # Compute residuals
            theoretical_g = self._theoretical_moment_function_varsigma_nls(
                lambda_vals, varsigmas, phis, alpha
            )
            residuals = empirical_g - theoretical_g
            
            # Estimate residual variance
            sigma2 = np.mean(residuals**2)
            
            # Variance-covariance matrix: σ²(J'J)⁻¹
            JtJ = jacobian.T @ jacobian
            
            # Regularize if needed
            eigenvals = np.linalg.eigvals(JtJ)
            if np.min(eigenvals) <= 1e-12:
                JtJ += 1e-8 * np.eye(n_params)
            
            vcv_matrix = sigma2 * np.linalg.inv(JtJ) / n
            
            return vcv_matrix
            
        except Exception as e:
            print(f"Warning: Jacobian computation failed for NLS method: {e}")
            # Return diagonal matrix with rough approximations
            variances = np.concatenate([
                (0.1 * np.abs(phis) + 0.01)**2,      # For phis
                (0.1 * np.abs(varsigmas) + 0.01)**2  # For varsigmas
            ])
            return np.diag(variances)


    def _compute_jacobian_nls_inference(self, varsigmas: np.ndarray, phis: np.ndarray,
                                    lambda_vals: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute Jacobian matrix for NLS moment conditions.
        """
        J = len(varsigmas)
        M = len(lambda_vals)  # Number of moment conditions
        jacobian = np.zeros((M, 2*J))  # [varsigma_1, ..., varsigma_J, phi_1, ..., phi_J]
        
        epsilon = 1e-6
        
        for i, lam in enumerate(lambda_vals):
            # Derivatives w.r.t. varsigmas (parameters 0 to J-1)
            for j in range(J):
                # Finite difference for ∂g/∂varsigma_j
                varsigmas_plus = varsigmas.copy()
                varsigmas_plus[j] += epsilon
                varsigmas_minus = varsigmas.copy()
                varsigmas_minus[j] -= epsilon
                
                g_plus = self._theoretical_moment_function_varsigma_nls(
                    np.array([lam]), varsigmas_plus, phis, alpha
                )[0]
                g_minus = self._theoretical_moment_function_varsigma_nls(
                    np.array([lam]), varsigmas_minus, phis, alpha
                )[0]
                
                jacobian[i, j] = (g_plus - g_minus) / (2 * epsilon)
            
            # Derivatives w.r.t. phis (parameters J to 2J-1)
            for j in range(J):
                # Finite difference for ∂g/∂phi_j
                phis_plus = phis.copy()
                phis_plus[j] += epsilon
                phis_minus = phis.copy()
                phis_minus[j] -= epsilon
                
                # Ensure we stay in bounds
                phis_plus[j] = min(phis_plus[j], 0.99)
                phis_minus[j] = max(phis_minus[j], -0.99)
                
                g_plus = self._theoretical_moment_function_varsigma_nls(
                    np.array([lam]), varsigmas, phis_plus, alpha
                )[0]
                g_minus = self._theoretical_moment_function_varsigma_nls(
                    np.array([lam]), varsigmas, phis_minus, alpha
                )[0]
                
                jacobian[i, J + j] = (g_plus - g_minus) / (2 * epsilon)
        
        return jacobian


    def _compute_vcv_matrix_beta(self, X: np.ndarray, c: float, a: float, b: float,
                                alpha: float, beta: float) -> np.ndarray:
        """
        Computes variance-covariance matrix for continuous mixture (beta) parameters.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Time series data
        c : float
            Estimated c parameter
        a : float
            Estimated a parameter (beta distribution)
        b : float
            Estimated b parameter (beta distribution)  
        alpha : float
            Estimated alpha parameter (treated as fixed)
        beta : float
            Estimated beta parameter (treated as fixed)
            
        Returns:
        --------
        numpy.ndarray
            Variance-covariance matrix for [c, a, b]
        """
        n = len(X)
        n_params = 3  # c, a, b
        
        # Define lambda values for moment conditions
        lambda_vals = np.linspace(0.1, 1.5, 10)
        
        try:
            # Calculate empirical moment function
            empirical_g = []
            u0 = 0.01
            for lam in lambda_vals:
                psi = self.estimate_ecf(X, u0, lam * u0)
                log_abs_psi = np.log(np.abs(psi))
                g_lambda = log_abs_psi / (np.abs(u0)**alpha)
                empirical_g.append(g_lambda)
            
            empirical_g = np.array(empirical_g)
            
            # Compute Jacobian numerically
            theta = np.array([c, a, b])
            jacobian = np.zeros((len(lambda_vals), n_params))
            epsilon = 1e-5
            
            for i, lam in enumerate(lambda_vals):
                for j in range(n_params):
                    # Create perturbation
                    theta_plus = theta.copy()
                    theta_minus = theta.copy()
                    theta_plus[j] += epsilon
                    theta_minus[j] -= epsilon
                    
                    # Ensure positive parameters
                    theta_plus = np.maximum(theta_plus, 1e-6)
                    theta_minus = np.maximum(theta_minus, 1e-6)
                    
                    # Compute theoretical moments
                    g_plus = self.theoretical_moment_function_beta(
                        np.array([lam]), theta_plus, alpha
                    )[0]
                    g_minus = self.theoretical_moment_function_beta(
                        np.array([lam]), theta_minus, alpha
                    )[0]
                    
                    jacobian[i, j] = (g_plus - g_minus) / (2 * epsilon)
            
            # Compute residuals
            theoretical_g = self.theoretical_moment_function_beta(lambda_vals, theta, alpha)
            residuals = empirical_g - theoretical_g
            
            # Estimate residual variance
            sigma2 = np.mean(residuals**2)
            
            # Variance-covariance matrix
            JtJ = jacobian.T @ jacobian
            
            # Regularize if needed
            eigenvals = np.linalg.eigvals(JtJ)
            if np.min(eigenvals) <= 1e-12:
                JtJ += 1e-8 * np.eye(n_params)
            
            vcv_matrix = sigma2 * np.linalg.inv(JtJ) / n
            
            return vcv_matrix
            
        except Exception as e:
            print(f"Warning: VCV computation failed for beta mixture: {e}")
            # Return diagonal matrix with rough approximations
            variances = np.array([
                (0.1 * c + 0.01)**2,
                (0.2 * a + 0.05)**2,
                (0.2 * b + 0.05)**2
            ])
            return np.diag(variances)


    # --- Monte Carlo simulations ---


    def mcsingle(self, sim_idx: int, param_idx: int, params: Dict[str, Any], 
                T: int, seed_offset: int = 0, verbose: bool = False, method: str = 'nls') -> Dict[str, Any]:
        """
        Executes a single Monte Carlo simulation for a given parameter set.

        Parameters:
        -----------
        sim_idx : int
            Simulation index
        param_idx : int
            Parameter set index
        params : dict
            Model parameters
        T : int
            Sample size
        seed_offset : int
            Offset for random seed
        verbose : bool
            If True, prints error information
            
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        result = {}
        
        try:
            # Different seed for each simulation
            seed = (sim_idx * 100 + param_idx) * 10 + seed_offset
            np.random.seed(seed)
            
            # Generate process
            Y, _ = self.generate_mixture(T, params)
            
            # Estimate parameters
            start_time = time.time()
            
            # Prepare params_dict for estimation
            est_params_dict = {}
            if self.model_type == 'discrete':
                est_params_dict['J'] = len(params['phis'])
            
            estimated_params = self.sequential_estimation(Y, params_dict=est_params_dict, verbose=False, method = method)
            estimation_time = time.time() - start_time
            
            # Calculate errors
            errors = self.evaluate_estimation(params, estimated_params)
            
            # Build results dictionary
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'estimation_time': estimation_time,
                'method_used': method,
                **errors
            }
            
            # Add true and estimated parameters
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    result[f"{key}_true"] = value.tolist()
                else:
                    result[f"{key}_true"] = value
                    
            for key, value in estimated_params.items():
                if isinstance(value, np.ndarray):
                    result[f"{key}_est"] = value.tolist()
                else:
                    result[f"{key}_est"] = value
            
        except Exception as e:
            if verbose:
                print(f"Error in simulation {sim_idx}, parameter set {param_idx}: {e}")
            # Return result with error
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'method_used': method,
                'error': str(e)
            }
        
        return result


    def mcstudy(self, n_simulations: int = 50, T: int = 500, parameter_sets: Optional[List[Dict[str, Any]]] = None, 
                n_jobs: int = 4, verbose: bool = True, method: str = 'md') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a parallelized Monte Carlo study to evaluate model performance.
        Modified to handle varsigma parameterization for discrete model.
        
        Parameters:
        -----------
        n_simulations : int
            Number of simulations per parameter set
        T : int
            Sample size
        parameter_sets : list of dict or None
            List of parameter sets to test
        n_jobs : int
            Number of parallel processes (-1 to use all cores)
        verbose : bool
            If True, prints progress information
            
        Returns:
        --------
        tuple
            (results_df, summary_df) containing:
                results_df: Detailed results for each simulation
                summary_df: Summary statistics for each parameter set
        """
        if parameter_sets is None:
            # Generate default parameter sets according to model type
            if self.model_type == 'discrete':
                phis1 = np.array([0.3, 0.6, 0.9])
                # Define varsigmas instead of pis and c separately
                varsigmas1 = np.array([0.2, 0.3, 0.5])  # These sum to 1, so c=1
                
                phis2 = np.array([0.4, 0.7])
                varsigmas2 = np.array([0.48, 0.32])  # These sum to 0.8, so c=0.8
                
                parameter_sets = [
                    {
                        'alpha': 1.5,
                        'beta': 0.0,
                        'sigma': 1.0,
                        'phis': phis1,
                        'varsigmas': varsigmas1,
                        'J': len(phis1)
                    },
                    {
                        'alpha': 1.7,
                        'beta': 0.3,
                        'sigma': 0.9,
                        'phis': phis2,
                        'varsigmas': varsigmas2,
                        'J': len(phis2)
                    }
                ]
                
                # Calculate c and pis for reporting (though not directly used in simulation)
                for params in parameter_sets:
                    params['c'] = np.sum(params['varsigmas'])
                    params['pis'] = params['varsigmas'] / params['c']
                    
            else:  # continuous
                parameter_sets = [
                    {
                        'alpha': 1.5,
                        'beta': 0.0,
                        'c': 1.0,
                        'sigma': 1.0,
                        'a': 2.0,
                        'b': 5.0,
                        'n_components': 50
                    },
                    {
                        'alpha': 1.7,
                        'beta': 0.3,
                        'c': 0.8,
                        'sigma': 0.9,
                        'a': 3.0,
                        'b': 3.0,
                        'n_components': 50
                    }
                ]
        else:
            # Check and process given parameters
            if self.model_type == 'discrete':
                for param_set in parameter_sets:
                    if 'J' not in param_set and 'phis' in param_set:
                        param_set['J'] = len(param_set['phis'])
                    
                    # If varsigmas not provided but pis and c are, calculate varsigmas
                    if 'varsigmas' not in param_set and 'pis' in param_set and 'c' in param_set:
                        param_set['varsigmas'] = param_set['pis'] * param_set['c']
                    # If varsigmas provided but pis or c not, calculate them
                    elif 'varsigmas' in param_set:
                        if 'c' not in param_set:
                            param_set['c'] = np.sum(param_set['varsigmas'])
                        if 'pis' not in param_set:
                            param_set['pis'] = param_set['varsigmas'] / param_set['c']
        
        # Create task list
        tasks = [
            (sim_idx, param_idx, params, T, method)
            for sim_idx in range(n_simulations)
            for param_idx, params in enumerate(parameter_sets)
        ]
        
        # Configure CPU usage
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 4)
        
        # Execute tasks in parallel
        if verbose:
            print(f"Running {len(tasks)} simulations on {n_jobs} cores...")
            print(f"Model type: {self.model_type}")
            print(f"Estimation method: {method}")
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(self.mcsingle)(sim_idx, param_idx, params, T, method=method_val)
            for sim_idx, param_idx, params, T, method_val in tasks
        )
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Process summary statistics
        summary_rows = []
        
        # Define parameters to analyze based on model type
        if self.model_type == 'discrete':
            scalar_params = ['alpha', 'beta', 'c']
        else:
            scalar_params = ['alpha', 'beta', 'c', 'a', 'b']
        
        for param_set in sorted(results_df['param_set'].unique()):
            subset = results_df[results_df['param_set'] == param_set]
            
            # Dictionary to store results for this parameter set
            param_results = {'param_set': param_set}
            
            # Calculate statistics for scalar parameters
            for param in scalar_params:
                if f"{param}_true" in subset.columns and f"{param}_est" in subset.columns:
                    # Get true value and estimated values
                    true_value = subset[f"{param}_true"].iloc[0]
                    est_values = subset[f"{param}_est"].values
                    
                    # Skip if not successful simulations
                    valid_mask = ~pd.isna(est_values)
                    if not np.any(valid_mask):
                        continue
                    
                    valid_est = est_values[valid_mask]
                    
                    # Calculate metrics
                    mean_est = np.mean(valid_est)
                    bias = mean_est - true_value
                    mse = np.mean((valid_est - true_value) ** 2)
                    rmse = np.sqrt(mse)
                    
                    # Store metrics
                    param_results[f"{param}_true"] = true_value
                    param_results[f"{param}_mean"] = mean_est
                    param_results[f"{param}_bias"] = bias
                    param_results[f"{param}_rmse"] = rmse
            
            # For discrete model, handle vector parameters with label switching correction
            if self.model_type == 'discrete':
                vector_stats = self._compute_vector_statistics_with_label_correction(subset, param_set)
                param_results.update(vector_stats)

            # Average estimation time
            if 'estimation_time' in subset.columns:
                param_results['estimation_time_mean'] = subset['estimation_time'].mean()
                param_results['estimation_time_std'] = subset['estimation_time'].std()
            
            # Add success rate
            if 'error' in subset.columns:
                # Count simulations without errors
                param_results['success_rate'] = len(subset[subset['error'].isna()]) / len(subset)
            else:
                # If no error column, assume all simulations were successful
                param_results['success_rate'] = 1.0
            
            summary_rows.append(param_results)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        return results_df, summary_df


    def analyze_results(self, results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                    plot: bool = True, save_plot: bool = False, verbose: bool = True) -> pd.DataFrame:
        """
        Analyzes and visualizes Monte Carlo simulation results with Mean, Bias, and RMSE metrics.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Detailed results
        summary_df : pd.DataFrame
            Summary results with Mean, Bias, and RMSE
        plot : bool
            If True, generates plots
        save_plot : bool
            If True, saves plots
        verbose : bool
            If True, prints summary information
            
        Returns:
        --------
        pd.DataFrame
            Formatted summary statistics table with Mean, Bias, and RMSE
        """
        if verbose:
            print(f"\n=== Parameter Estimation Summary ({self.model_type}) ===")
        
        # Create formatted summary table from the already computed summary_df
        formatted_summary = pd.DataFrame()
        
        # Define parameters to display based on model type
        if self.model_type == 'discrete':
            scalar_params = ['alpha', 'beta', 'c']
        else:
            scalar_params = ['alpha', 'beta', 'c', 'a', 'b']
        
        for param_set in sorted(summary_df['param_set'].unique()):
            subset = summary_df[summary_df['param_set'] == param_set]
            
            # For scalar parameters
            for param in scalar_params:
                if f"{param}_true" in subset.columns:
                    row_name = f"Set {param_set+1} - {param}"
                    
                    # Extract metrics from summary_df
                    formatted_summary.loc[row_name, 'True'] = subset[f"{param}_true"].iloc[0]
                    formatted_summary.loc[row_name, 'Mean'] = subset[f"{param}_mean"].iloc[0]
                    formatted_summary.loc[row_name, 'Bias'] = subset[f"{param}_bias"].iloc[0]
                    formatted_summary.loc[row_name, 'RMSE'] = subset[f"{param}_rmse"].iloc[0]
            
            # For discrete model, handle vector parameters
            if self.model_type == 'discrete':
                # Handle phis
                j = 1
                while f"phi_{j}_true" in subset.columns:
                    row_name = f"Set {param_set+1} - phi_{j}"
                    formatted_summary.loc[row_name, 'True'] = subset[f"phi_{j}_true"].iloc[0]
                    formatted_summary.loc[row_name, 'Mean'] = subset[f"phi_{j}_mean"].iloc[0]
                    formatted_summary.loc[row_name, 'Bias'] = subset[f"phi_{j}_bias"].iloc[0]
                    formatted_summary.loc[row_name, 'RMSE'] = subset[f"phi_{j}_rmse"].iloc[0]
                    j += 1
                
                # Handle pis
                j = 1
                while f"pi_{j}_true" in subset.columns:
                    row_name = f"Set {param_set+1} - pi_{j}"
                    formatted_summary.loc[row_name, 'True'] = subset[f"pi_{j}_true"].iloc[0]
                    formatted_summary.loc[row_name, 'Mean'] = subset[f"pi_{j}_mean"].iloc[0]
                    formatted_summary.loc[row_name, 'Bias'] = subset[f"pi_{j}_bias"].iloc[0]
                    formatted_summary.loc[row_name, 'RMSE'] = subset[f"pi_{j}_rmse"].iloc[0]
                    j += 1
            
            # Add estimation time
            if 'estimation_time_mean' in subset.columns:
                row_time = f"Set {param_set+1} - Time"
                formatted_summary.loc[row_time, 'Mean'] = subset['estimation_time_mean'].iloc[0]
                if 'estimation_time_std' in subset.columns:
                    formatted_summary.loc[row_time, 'Std'] = subset['estimation_time_std'].iloc[0]
            
            # Add success rate
            if 'success_rate' in subset.columns:
                row_success = f"Set {param_set+1} - Success"
                formatted_summary.loc[row_success, 'Rate'] = subset['success_rate'].iloc[0]
        
        # Round values for better display
        formatted_summary = formatted_summary.round(6)
        
        if verbose:
            print(formatted_summary)
        
        # Plot results if requested
        if plot:
            self._plot_mc_results(results_df, save_plot, verbose=verbose)
        
        return formatted_summary


    def _plot_mc_results(self, results_df: pd.DataFrame, save_plot: bool = False, 
                        verbose: bool = True) -> None:
        """
        Visualizes Monte Carlo results with parameter distribution plots.
        Modified to show Mean, Bias, and RMSE information.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Detailed results
        save_plot : bool
            If True, saves plots
        verbose : bool
            If True, prints progress information
        """
        # Define Greek symbols for parameters
        greek_symbols = {
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "c": "c",
            "sigma": r"$\sigma$",
            "a": "a",
            "b": "b"
        }
        
        # Create a separate plot for each parameter set
        for param_set in sorted(results_df['param_set'].unique()):
            # Filter data for this parameter set
            subset = results_df[results_df['param_set'] == param_set]
            
            # Check if there are errors in this parameter set
            if 'error' in subset.columns and not subset['error'].isna().all():
                if verbose:
                    print(f"Skipping visualization for parameter set {param_set} due to errors")
                continue
            
            # Determine parameters to plot according to model type
            if self.model_type == 'discrete':
                param_names = ['alpha', 'beta', 'c']
            else:
                param_names = ['alpha', 'beta', 'c', 'a', 'b']
            
            # Create figure
            n_params = len(param_names)
            n_cols = min(3, n_params)
            n_rows = int(np.ceil(n_params / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            
            # Ensure axes is always a 2D array
            if n_params == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = np.array([axes])
            
            # Plot each scalar parameter
            plot_idx = 0
            for param_name in param_names:
                if f"{param_name}_true" in subset.columns and f"{param_name}_est" in subset.columns:
                    ax = axes[plot_idx // n_cols, plot_idx % n_cols]
                    
                    # Get data
                    param_true = subset[f"{param_name}_true"].iloc[0]
                    param_est = subset[f"{param_name}_est"].values
                    
                    # Calculate statistics
                    param_mean = np.mean(param_est)
                    param_bias = param_mean - param_true
                    param_rmse = np.sqrt(np.mean((param_est - param_true) ** 2))
                    
                    # Create histogram
                    ax.hist(param_est, bins=15, alpha=0.7, edgecolor='black')
                    
                    # Add true value
                    ax.axvline(param_true, color='red', linestyle='dashed', linewidth=2, label='True value')
                    
                    # Add estimated mean
                    ax.axvline(param_mean, color='green', linestyle='dotted', linewidth=2, label='Mean estimate')
                    
                    # Add labels with Bias and RMSE
                    symbol = greek_symbols.get(param_name, param_name)
                    title_text = f'{symbol}\n'
                    title_text += f'True: {param_true:.3f}, Mean: {param_mean:.3f}\n'
                    title_text += f'Bias: {param_bias:.3f}, RMSE: {param_rmse:.3f}'
                    ax.set_title(title_text, fontsize=10)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    plot_idx += 1
            
            # Remove unused subplots
            for i in range(plot_idx, n_rows * n_cols):
                fig.delaxes(axes[i // n_cols, i % n_cols])
            
            # Global title
            plt.suptitle(f'Parameter Set {param_set+1} - {self.model_type} Model', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save or display
            if save_plot:
                filename = f"mc_results_{self.model_type}_set{param_set+1}.png"
                plt.savefig(os.path.join(self.results_dir, filename), dpi=300)
                if verbose:
                    print(f"Figure saved to {os.path.join(self.results_dir, filename)}")
            
            plt.show()
            
            # For discrete model, also plot phis and pis
            if self.model_type == 'discrete':
                self._plot_discrete_params(subset, param_set, save_plot, verbose=verbose)


    def _plot_discrete_params(self, subset: pd.DataFrame, param_set: int, 
                            save_plot: bool = False, verbose: bool = True) -> None:
        """
        Plots vector parameters (phis and varsigmas) for the discrete model.
        Modified to handle varsigma parameterization.
        
        Parameters:
        -----------
        subset : pd.DataFrame
            Subset of results for one parameter set
        param_set : int
            Parameter set index
        save_plot : bool
            If True, saves plots
        verbose : bool
            If True, prints progress information
        """
        # Check if necessary columns exist
        if 'phis_true' not in subset.columns or 'phis_est' not in subset.columns:
            return
        
        # Get dimensions
        phis_true = np.array(subset['phis_true'].iloc[0])
        J = len(phis_true)
        
        # Prepare figure
        fig, axes = plt.subplots(1, 3 if 'varsigmas_est' in subset.columns else 2, figsize=(16, 5))
        
        # Plot phis
        ax1 = axes[0]
        all_phis_est = np.array(subset['phis_est'].tolist())
        
        # Calculate statistics for phis
        for j in range(J):
            phi_true = phis_true[j]
            phi_est_values = all_phis_est[:, j]
            
            # Calculate statistics
            phi_mean = np.mean(phi_est_values)
            phi_bias = phi_mean - phi_true
            phi_rmse = np.sqrt(np.mean((phi_est_values - phi_true) ** 2))
            
            # Create histogram
            ax1.hist(phi_est_values, bins=15, alpha=0.6, 
                    label=f'$\\phi_{j+1}$ (Mean: {phi_mean:.3f})')
            
            # Add true value
            ax1.axvline(phi_true, color=f'C{j}', linestyle='dashed', linewidth=2)
            
            # Add text with statistics
            y_pos = ax1.get_ylim()[1] * (0.9 - j * 0.15)
            ax1.text(0.02, y_pos, f'$\\phi_{j+1}$: True={phi_true:.3f}, Bias={phi_bias:.3f}, RMSE={phi_rmse:.3f}',
                    transform=ax1.transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax1.set_title('Distribution of estimated $\\phi$ values')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot varsigmas if available
        if 'varsigmas_true' in subset.columns and 'varsigmas_est' in subset.columns:
            ax2 = axes[1]
            varsigmas_true = np.array(subset['varsigmas_true'].iloc[0])
            all_varsigmas_est = np.array(subset['varsigmas_est'].tolist())
            
            # Calculate statistics for varsigmas
            for j in range(J):
                varsigma_true = varsigmas_true[j]
                varsigma_est_values = all_varsigmas_est[:, j]
                
                # Calculate statistics
                varsigma_mean = np.mean(varsigma_est_values)
                varsigma_bias = varsigma_mean - varsigma_true
                varsigma_rmse = np.sqrt(np.mean((varsigma_est_values - varsigma_true) ** 2))
                
                # Create histogram
                ax2.hist(varsigma_est_values, bins=15, alpha=0.6, 
                        label=f'$\\varsigma_{j+1}$ (Mean: {varsigma_mean:.3f})')
                
                # Add true value
                ax2.axvline(varsigma_true, color=f'C{j}', linestyle='dashed', linewidth=2)
                
                # Add text with statistics
                y_pos = ax2.get_ylim()[1] * (0.9 - j * 0.15)
                ax2.text(0.02, y_pos, f'$\\varsigma_{j+1}$: True={varsigma_true:.3f}, Bias={varsigma_bias:.3f}, RMSE={varsigma_rmse:.3f}',
                        transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax2.set_title('Distribution of estimated $\\varsigma$ values')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot pis for backward compatibility
        ax_idx = 2 if 'varsigmas_est' in subset.columns else 1
        if 'pis_true' in subset.columns and 'pis_est' in subset.columns:
            ax3 = axes[ax_idx]
            pis_true = np.array(subset['pis_true'].iloc[0])
            all_pis_est = np.array(subset['pis_est'].tolist())
            
            # Calculate statistics for pis
            for j in range(J):
                pi_true = pis_true[j]
                pi_est_values = all_pis_est[:, j]
                
                # Calculate statistics
                pi_mean = np.mean(pi_est_values)
                pi_bias = pi_mean - pi_true
                pi_rmse = np.sqrt(np.mean((pi_est_values - pi_true) ** 2))
                
                # Create histogram
                ax3.hist(pi_est_values, bins=15, alpha=0.6, 
                        label=f'$\\pi_{j+1}$ (Mean: {pi_mean:.3f})')
                
                # Add true value
                ax3.axvline(pi_true, color=f'C{j}', linestyle='dashed', linewidth=2)
                
                # Add text with statistics
                y_pos = ax3.get_ylim()[1] * (0.9 - j * 0.15)
                ax3.text(0.02, y_pos, f'$\\pi_{j+1}$: True={pi_true:.3f}, Bias={pi_bias:.3f}, RMSE={pi_rmse:.3f}',
                        transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax3.set_title('Distribution of estimated $\\pi$ values')
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'Vector Parameters - Set {param_set+1}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_plot:
            filename = f"mc_results_discrete_vectors_set{param_set+1}.png"
            plt.savefig(os.path.join(self.results_dir, filename), dpi=300)
            if verbose:
                print(f"Figure saved to {os.path.join(self.results_dir, filename)}")
        
        plt.show()


    def _find_best_component_permutation(self, estimated_params: Dict[str, Any], 
                                    true_params: Dict[str, Any]) -> List[int]:
        """
        Find the optimal permutation of components to minimize estimation error.
        
        This method solves the label switching problem by testing all possible
        permutations and selecting the one that minimizes the total error.
        
        Parameters:
        -----------
        estimated_params : dict
            Dictionary containing estimated parameters
        true_params : dict
            Dictionary containing true parameters
            
        Returns:
        --------
        list
            Indices of the optimal permutation
        """
        from itertools import permutations
        
        if 'phis' not in estimated_params or 'phis' not in true_params:
            J = estimated_params.get('J', 3)
            return list(range(J))
        
        J = len(true_params['phis'])
        true_phis = np.array(true_params['phis'])
        est_phis = np.array(estimated_params['phis'])
        
        # Prepare varsigmas for comparison
        true_varsigmas = None
        est_varsigmas = None
        
        if 'varsigmas' in true_params and 'varsigmas' in estimated_params:
            true_varsigmas = np.array(true_params['varsigmas'])
            est_varsigmas = np.array(estimated_params['varsigmas'])
        elif 'pis' in true_params and 'pis' in estimated_params:
            true_c = true_params.get('c', 1.0)
            est_c = estimated_params.get('c', 1.0)
            true_varsigmas = np.array(true_params['pis']) * true_c
            est_varsigmas = np.array(estimated_params['pis']) * est_c
        
        best_error = float('inf')
        best_permutation = list(range(J))
        
        for perm in permutations(range(J)):
            perm_list = list(perm)
            perm_phis = est_phis[perm_list]
            
            # Error on phis
            phi_error = np.sum((perm_phis - true_phis)**2)
            total_error = phi_error
            
            # Add error on varsigmas if available
            if true_varsigmas is not None and est_varsigmas is not None:
                perm_varsigmas = est_varsigmas[perm_list]
                varsigma_error = np.sum((perm_varsigmas - true_varsigmas)**2)
                total_error += varsigma_error
            
            if total_error < best_error:
                best_error = total_error
                best_permutation = perm_list
        
        return best_permutation


    def _apply_permutation_to_params(self, params: Dict[str, Any], 
                                permutation: List[int]) -> Dict[str, Any]:
        """
        Apply a permutation to reorder vector parameters.
        
        Parameters:
        -----------
        params : dict
            Parameter dictionary containing vector parameters
        permutation : list
            List of indices defining the permutation
            
        Returns:
        --------
        dict
            Dictionary with reordered parameters
        """
        reordered = params.copy()
        
        if 'phis' in params:
            reordered['phis'] = np.array(params['phis'])[permutation]
        
        if 'varsigmas' in params:
            reordered['varsigmas'] = np.array(params['varsigmas'])[permutation]
        
        if 'pis' in params:
            reordered['pis'] = np.array(params['pis'])[permutation]
        
        return reordered


    def _compute_vector_statistics_with_label_correction(self, subset: pd.DataFrame, 
                                                    param_set: int) -> Dict[str, Any]:
        """
        Compute statistics for vector parameters with label switching correction.
        
        This method handles the label switching problem by finding the optimal
        permutation for each simulation before computing aggregate statistics.
        
        Parameters:
        -----------
        subset : pd.DataFrame
            Subset of results for one parameter set
        param_set : int
            Parameter set index
            
        Returns:
        --------
        dict
            Dictionary containing corrected statistics
        """
        param_results = {}
        
        if 'phis_true' not in subset.columns or 'phis_est' not in subset.columns:
            return param_results
        
        phis_true = np.array(subset['phis_true'].iloc[0])
        varsigmas_true = (np.array(subset['varsigmas_true'].iloc[0]) 
                        if 'varsigmas_true' in subset.columns else None)
        pis_true = (np.array(subset['pis_true'].iloc[0]) 
                if 'pis_true' in subset.columns else None)
        c_true = subset['c_true'].iloc[0] if 'c_true' in subset.columns else 1.0
        
        # Build true parameters dictionary
        true_params = {
            'phis': phis_true,
            'c': c_true
        }
        if varsigmas_true is not None:
            true_params['varsigmas'] = varsigmas_true
        if pis_true is not None:
            true_params['pis'] = pis_true
        
        # Filter valid simulations
        valid_subset = subset[subset['phis_est'].notna()]
        
        if len(valid_subset) == 0:
            return param_results
        
        J = len(phis_true)
        
        # Reorder each simulation according to optimal permutation
        reordered_phis = []
        reordered_varsigmas = []
        reordered_pis = []
        
        for idx, row in valid_subset.iterrows():
            # Build estimated parameters for this simulation
            est_params = {
                'phis': np.array(row['phis_est']),
                'c': row.get('c_est', 1.0)
            }
            if 'varsigmas_est' in row and row['varsigmas_est'] is not None:
                est_params['varsigmas'] = np.array(row['varsigmas_est'])
            if 'pis_est' in row and row['pis_est'] is not None:
                est_params['pis'] = np.array(row['pis_est'])
            
            # Find optimal permutation
            optimal_perm = self._find_best_component_permutation(est_params, true_params)
            
            # Apply permutation
            reordered_est = self._apply_permutation_to_params(est_params, optimal_perm)
            
            # Store reordered results
            reordered_phis.append(reordered_est['phis'])
            if 'varsigmas' in reordered_est:
                reordered_varsigmas.append(reordered_est['varsigmas'])
            if 'pis' in reordered_est:
                reordered_pis.append(reordered_est['pis'])
        
        # Compute statistics on reordered data
        reordered_phis = np.array(reordered_phis)
        
        # Statistics for phis
        for j in range(J):
            true_value = phis_true[j]
            est_values = reordered_phis[:, j]
            
            mean_est = np.mean(est_values)
            bias = mean_est - true_value
            rmse = np.sqrt(np.mean((est_values - true_value) ** 2))
            
            param_results[f"phi_{j+1}_true"] = true_value
            param_results[f"phi_{j+1}_mean"] = mean_est
            param_results[f"phi_{j+1}_bias"] = bias
            param_results[f"phi_{j+1}_rmse"] = rmse
        
        # Statistics for varsigmas
        if reordered_varsigmas and varsigmas_true is not None:
            reordered_varsigmas = np.array(reordered_varsigmas)
            
            for j in range(J):
                true_value = varsigmas_true[j]
                est_values = reordered_varsigmas[:, j]
                
                mean_est = np.mean(est_values)
                bias = mean_est - true_value
                rmse = np.sqrt(np.mean((est_values - true_value) ** 2))
                
                param_results[f"varsigma_{j+1}_true"] = true_value
                param_results[f"varsigma_{j+1}_mean"] = mean_est
                param_results[f"varsigma_{j+1}_bias"] = bias
                param_results[f"varsigma_{j+1}_rmse"] = rmse
        
        # Statistics for pis (for backward compatibility)
        if reordered_pis and pis_true is not None:
            reordered_pis = np.array(reordered_pis)
            
            for j in range(J):
                true_value = pis_true[j]
                est_values = reordered_pis[:, j]
                
                mean_est = np.mean(est_values)
                bias = mean_est - true_value
                rmse = np.sqrt(np.mean((est_values - true_value) ** 2))
                
                param_results[f"pi_{j+1}_true"] = true_value
                param_results[f"pi_{j+1}_mean"] = mean_est
                param_results[f"pi_{j+1}_bias"] = bias
                param_results[f"pi_{j+1}_rmse"] = rmse
        
        return param_results


    # --- Test methods ---


    def smart_demo(self, n: int = 1000, params: Optional[Dict[str, Any]] = None, 
                    seed: int = 42, plot: bool = True, verbose: bool = False, method: str = 'md') -> Dict[str, Any]:
        """
        Generates an aggregated process and tests parameter estimation.
        Modified to handle varsigma parameterization for discrete model.
        
        Parameters:
        -----------
        n : int
            Length of the series
        params : dict or None
            Model parameters (uses default values if None)
        seed : int
            Seed for random generation
        plot : bool
            Indicates whether results should be visualized
        verbose : bool
            If True, prints progress information
            
        Returns:
        --------
        dict
            Complete test results
        """
        np.random.seed(seed)
        
        # Default parameters according to model type
        if params is None:
            if self.model_type == 'discrete':
                phis = np.array([0.3, 0.6, 0.9])
                # Use varsigmas instead of pis and c separately
                varsigmas = np.array([0.2, 0.3, 0.5])  # These sum to 1, so c=1
                
                params = {
                    'alpha': 1.5,
                    'beta': 0.0,
                    'sigma': 1.0,
                    'phis': phis,
                    'varsigmas': varsigmas,
                    'J': len(phis)  # Add J explicitly
                }
                
                # Calculate c and pis for reporting
                params['c'] = np.sum(varsigmas)
                params['pis'] = varsigmas / params['c']
                
            else:  # continuous
                params = {
                    'alpha': 1.5,
                    'beta': 0.0,
                    'c': 1.0,
                    'sigma': 1.0,
                    'a': 2.0,
                    'b': 5.0,
                    'n_components': 50
                }
        else:
            # If params is provided but without necessary fields, add them
            if self.model_type == 'discrete':
                if 'J' not in params and 'phis' in params:
                    params['J'] = len(params['phis'])
                    
                # Handle conversion between parameterizations
                if 'varsigmas' not in params and 'pis' in params and 'c' in params:
                    params['varsigmas'] = params['pis'] * params['c']
                elif 'varsigmas' in params:
                    if 'c' not in params:
                        params['c'] = np.sum(params['varsigmas'])
                    if 'pis' not in params:
                        params['pis'] = params['varsigmas'] / params['c']
        
        if verbose:
            print(f"Generating an aggregated {self.model_type} process")
            print(f"Parameters: {params}")
        
        # Generate aggregated process
        Y, components = self.generate_mixture(n, params)
        
        # Sequential estimation
        if verbose:
            print("\nSequential parameter estimation:")
        
        params_dict = None
        if self.model_type == 'discrete':
            params_dict = {'J': params['J']}
        
        estimated_params = self.sequential_estimation(Y, params_dict=params_dict, verbose=verbose, method = method)
        
        # Performance evaluation
        errors = self.evaluate_estimation(params, estimated_params)
        
        if verbose:
            print("\nEstimation errors:")
            for key, value in errors.items():
                if not isinstance(value, (list, np.ndarray)):
                    print(f"{key}: {value:.6f}")
        
        # Plot process and its components
        if plot:
            self._plot_test_results(Y, components, params, estimated_params)
        
        return {
            'Y': Y,
            'components': components,
            'true_params': params,
            'estimated_params': estimated_params,
            'errors': errors
        }

    
    def _plot_test_results(self, Y: np.ndarray, components: List[np.ndarray], 
                        true_params: Dict[str, Any], estimated_params: Dict[str, Any]) -> None:
        """
        Visualizes the results of an estimation test.

        Parameters:
        -----------
        Y : numpy.ndarray
            Aggregated process
        components : list
            Individual components
        true_params : dict
            True parameters
        estimated_params : dict
            Estimated parameters
        """
        plt.figure(figsize=(14, 8))
        
        # Aggregated process
        plt.subplot(2, 1, 1)
        plt.plot(Y, label='Aggregated process')
        
        # Title with parameters
        alpha = true_params['alpha']
        beta = true_params['beta']
        c = true_params['c']
        
        if self.model_type == 'discrete':
            title = f'Discrete alpha-stable aggregated process (α={alpha}, β={beta}, c={c})'
        else:
            a = true_params['a']
            b = true_params['b']
            title = f'Beta alpha-stable aggregated process (α={alpha}, β={beta}, c={c}, a={a}, b={b})'
        
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Individual components
        plt.subplot(2, 1, 2)
        
        # Limit the number of components displayed
        max_components_to_show = 5
        n_components_to_show = min(len(components), max_components_to_show)
        
        for j in range(n_components_to_show):
            if self.model_type == 'discrete':
                phi = true_params['phis'][j]
                pi = true_params['pis'][j]
                label = f'Composante {j+1} (φ={phi:.2f}, π={pi:.2f})'
            else:
                # For the continuous model, show only the index
                label = f'Composante {j+1}'
            
            plt.plot(components[j], alpha=0.7, label=label)
        
        if len(components) > max_components_to_show:
            plt.title(f'First {max_components_to_show} individual components (out of {len(components)})')
        else:
            plt.title('Individual components')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Parameter comparison plot
        if self.model_type == 'continuous':
            # For the continuous model, plot the beta distribution
            self._plot_beta_comparison(true_params, estimated_params)

    
    def _plot_beta_comparison(self, true_params: Dict[str, Any], 
                            estimated_params: Dict[str, Any]) -> None:
        """
        Compares true and estimated beta distributions for the continuous model.

        Parameters:
        -----------
        true_params : dict
            True parameters
        estimated_params : dict
            Estimated parameters
        """
        phi = np.linspace(0.01, 0.99, 500)
        
        # Distribution vraie
        a_true = true_params['a']
        b_true = true_params['b']
        density_true = self.beta_pdf(phi, a_true, b_true)
        
        # Estimated distribution
        a_est = estimated_params['a']
        b_est = estimated_params['b']
        density_est = self.beta_pdf(phi, a_est, b_est)
        
        plt.figure(figsize=(10, 6))
        plt.plot(phi, density_true, 'b-', linewidth=2, label=f'Beta vraie (a={a_true:.2f}, b={b_true:.2f})')
        plt.plot(phi, density_est, 'r--', linewidth=2, label=f'Estimated beta (a={a_est:.2f}, b={b_est:.2f})')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'Density $\pi(\phi)$')
        plt.title('Comparison of beta distributions')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def smart_simu(self, alphas: List[float] = [1.0, 1.5, 1.7], 
                    betas: List[float] = [0.0, 0.3], 
                    n_simulations: int = 5, T: int = 2000, 
                    n_jobs: int = 4, verbose: bool = True, method: str = 'md') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to run performance analysis on a parameter grid.
        
        Parameters:
        -----------
        alphas : list
            Alpha values to test
        betas : list
            Beta values to test
        n_simulations : int
            Number of simulations per parameter combination
        T : int
            Sample size
        n_jobs : int
            Number of parallel processes
        verbose : bool
            If True, prints progress information
            
        Returns:
        --------
        tuple
            (results_df, summary_df) DataFrames with results
        """
        if verbose:
            print(f"=== Performance analysis for {self.model_type} model ===")
            print(f"Parameter grid: {len(alphas)} alphas × {len(betas)} betas")
            print(f"Simulations per point: {n_simulations}")
            print(f"Sample size: {T}")
        
        # Create parameter sets
        parameter_sets = []
        
        for alpha in alphas:
            for beta in betas:
                if self.model_type == 'discrete':
                    phis = np.array([0.3, 0.6, 0.9])
                    pis = np.array([0.2, 0.3, 0.5])
                    param_set = {
                        'alpha': alpha,
                        'beta': beta,
                        'c': 1.0,
                        'sigma': 1.0,
                        'phis': phis,
                        'pis': pis,
                        'J': len(phis)  # Add J explicitly
                    }
                else:  # continuous
                    param_set = {
                        'alpha': alpha,
                        'beta': beta,
                        'c': 1.0,
                        'sigma': 1.0,
                        'a': 9.0,
                        'b': 3.0,
                        'n_components': 20
                    }
                
                parameter_sets.append(param_set)
        
        # Launch Monte Carlo study
        results_df, summary_df = self.mcstudy(
            n_simulations=n_simulations,
            T=T,
            parameter_sets=parameter_sets,
            n_jobs=n_jobs,
            verbose=verbose, 
            method = method
        )
        
        # Analyze and visualize results
        formatted_summary = self.analyze_results(results_df, summary_df, plot=True, save_plot=True, verbose=verbose)
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, f'mc_results_{self.model_type}_raw.csv'), index=False)
        summary_df.to_csv(os.path.join(self.results_dir, f'mc_results_{self.model_type}_summary.csv'), index=False)
        formatted_summary.to_csv(os.path.join(self.results_dir, f'mc_results_{self.model_type}_formatted.csv'))
        
        if verbose:
            print(f"\nResults saved in {self.results_dir}/")
        
        return results_df, summary_df