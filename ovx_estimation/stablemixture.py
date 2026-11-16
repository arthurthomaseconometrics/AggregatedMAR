import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, optimize
from joblib import Parallel, delayed
import multiprocessing
from typing import Tuple, Dict, Optional, List
from numpy.typing import ArrayLike
import time
from tqdm.auto import tqdm

class StableMixtureDeconvolution:
    """
    Implementation of an alpha-stable mixture model with multiple anticipative (non-causal) components.
    
    This class provides tools for modeling, estimating, and filtering aggregated alpha-stable processes
    of the form:
        Y_t = sum(pi_j * X_j,t)  for j=1,...,J
    
    where each X_j,t is a non-causal alpha-stable AR(1) process:
        X_j,t = rho_j * X_j,t+1 + epsilon_j,t
    
    with epsilon_j,t ~ S(alpha, beta, 1, 0) being i.i.d. alpha-stable innovations.
    
    The model parameters include:
        - rho_j: Autoregressive coefficients (0 < rho_j < 1) for each component
        - varsigma_j: Combined scale parameters (varsigma_j = sigma * pi_j)
        - alpha: Tail index parameter (0 < alpha <= 2)
        - beta: Asymmetry parameter (-1 <= beta <= 1)
    
    Derived parameters:
        - sigma: Overall scale parameter (sigma = sum(varsigma_j))
        - pi_j: Mixing weights (pi_j = varsigma_j / sigma) with sum(pi_j) = 1
    
    Main features:
    - Parameter estimation using minimum distance methods based on characteristic functions
    - Component deconvolution through MCMC-based filtering
    - Monte Carlo simulation tools for model evaluation
    - Statistical inference for both direct and derived parameters
    - Visualization capabilities for model diagnostics
    
    References:
    - de Truchis, Fries, Thomas (2025) "Local explosion modelling by non-causal process"
    - de Truchis, Fries, Thomas (2025) "Local explosion modelling by non-causal process"
    """

    def __init__(self, 
                    num_components: int = 2,
                    alpha_value: float = 1.0, 
                    beta_value: float = 0.0,
                    fix_alpha: bool = False,
                    fix_beta: bool = False,
                    verbose: bool = False):
            """
            Initializes the convolution model with options for alpha and beta parameters.

            Args:
                num_components: Number of latent AR(1) components (J)
                alpha_value: Alpha parameter value (common to all components)
                beta_value: Beta parameter value (common to all components)
                fix_alpha: If True, alpha parameter will be fixed to alpha_value during estimation
                fix_beta: If True, beta parameter will be fixed to beta_value during estimation
                verbose: If True, prints detailed information during estimation
            """
            self.num_components = num_components
            self.alpha_value = alpha_value
            self.beta_value = beta_value
            self.fix_alpha = fix_alpha
            self.fix_beta = fix_beta
            self.verbose = verbose
            
            # Define parameter names based on configuration
            self.params_names = []
            for j in range(num_components):
                self.params_names.append(f"rho_{j+1}")
            for j in range(num_components):
                self.params_names.append(f"varsigma_{j+1}")
            
            # Only add alpha and beta to parameter names if they are not fixed
            if not self.fix_alpha:
                self.params_names.append("alpha")
            if not self.fix_beta:
                self.params_names.append("beta")
            
            # Storage for estimated parameters
            self.params = None


    def generate(self, 
                T: int, 
                params: tuple, 
                burn_in: int = 200, 
                seed: Optional[int] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Generates a mixed process with J non-causal alpha-stable components.

        Args:
            T: Number of observations to generate
            params: Tuple of parameters (rho_1, ..., rho_J, varsigma_1, ..., varsigma_J, alpha, beta)
            burn_in: Number of observations to eliminate at the beginning to ensure stationarity
            seed: Seed for reproducibility
            
        Returns:
            Tuple containing (Y, [X_1, X_2, ..., X_J]) where:
                Y: Mixed process (Y_t = sum(pi_j * X_j,t) for j=1,...,J)
                X_j: J non-causal alpha-stable components
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Extract parameters
        params_dict = self._params_to_dict(params)
        
        # Get common parameters
        alpha = params_dict.get("alpha", self.alpha_value)
        beta = params_dict.get("beta", self.beta_value)
        
        # Calculate sigma (scale parameter)
        varsigmas = []
        for j in range(self.num_components):
            varsigmas.append(params_dict[f"varsigma_{j+1}"])
        sigma = sum(varsigmas)  # sigma is the sum of varsigmas since pi_j sum to 1
        
        # Calculate pi values (weights)
        pi_values = [varsigma / sigma for varsigma in varsigmas]
        
        # Generate J non-causal components
        X_components = []
        
        for j in range(self.num_components):
            rho_j = params_dict[f"rho_{j+1}"]
            pi_j = pi_values[j]
            
            # Initialize component
            X_j = np.zeros(T + burn_in)
            epsilon_j = stats.levy_stable.rvs(
                alpha=alpha, beta=beta, loc=0, scale=1,  # Unit scale since we apply pi_j * sigma later
                size=T + burn_in, random_state=seed + j if seed is not None else None
            )
            
            # Build the non-causal process (in reverse time)
            for t in range(T + burn_in - 2, -1, -1):
                X_j[t] = rho_j * X_j[t+1] + epsilon_j[t]
            
            # Adjust with pi_j and sigma
            X_j = pi_j * sigma * X_j
            
            X_components.append(X_j[burn_in:])
        
        # Compute the mixed process
        Y = np.sum(X_components, axis=0)
        
        return Y, X_components


    def char_function(self, u: float, v: float, params: tuple) -> complex:
            """
            Calculates the joint characteristic function of (Y_t, Y_{t-1})

            Args:
                u, v: Evaluation points
                params: Tuple of parameters (rho_1, ..., rho_J, varsigma_1, ..., varsigma_J, alpha, beta)
                    
            Returns:
                Value of the characteristic function at point (u, v)
            """
            params_dict = self._params_to_dict(params)
            
            # Get common parameters
            alpha = params_dict.get("alpha", self.alpha_value)
            beta = params_dict.get("beta", self.beta_value)
            
            # Calculate sigma (scale parameter)
            varsigmas = []
            for j in range(self.num_components):
                varsigmas.append(params_dict[f"varsigma_{j+1}"])
            sigma = sum(varsigmas)  # sigma is the sum of varsigmas
            
            # Calculate pi values (weights)
            pi_values = [varsigma / sigma for varsigma in varsigmas]
            
            # Initialize the exponent for the characteristic function
            exponent = 0
            
            # Accumulate contributions from each component
            for j in range(self.num_components):
                rho_j = params_dict[f"rho_{j+1}"]
                pi_j = pi_values[j]
                
                # Calculate the exponent for the j-th component
                if alpha != 1:
                    # General formula for alpha != 1
                    component_exponent = -pi_j**alpha * sigma**alpha * (
                        np.abs(u + rho_j*v)**alpha * (1 - 1j * beta * np.sign(u + rho_j*v) * np.tan(np.pi * alpha / 2)) / (1-np.abs(rho_j)**alpha) 
                        + np.abs(v)**alpha * (1 - 1j * beta * np.sign(v) * np.tan(np.pi * alpha / 2))
                    )
                else:
                    # Special case for alpha = 1 (Cauchy with beta)
                    component_exponent = -pi_j * sigma * (
                        np.abs(u + rho_j*v) * (1 + 1j * beta * np.sign(u + rho_j*v) * 2/np.pi * np.log(np.abs(u + rho_j*v) + 1e-10)) / (1-np.abs(rho_j)) 
                        + np.abs(v) * (1 + 1j * beta * np.sign(v) * 2/np.pi * np.log(np.abs(v) + 1e-10))
                    )
                
                exponent += component_exponent
            
            # Return the characteristic function
            cf = np.exp(exponent)
            
            return cf


    def objective_function(self, params: tuple, Y: np.ndarray, 
                            use_optimal_weights: bool = True,
                            preliminary_params: Optional[tuple] = None) -> float:
            """
            Objective function based on the minimum distance between empirical and theoretical
            characteristic functions.

            Args:
                params: Tuple of parameters to evaluate
                Y: Observed time series
                use_optimal_weights: If True, uses optimal weights based on variance
                preliminary_params: Parameters estimated in the preliminary step (for weights calculation)
                    
            Returns:
                Value of the objective function (to minimize)
            """
            # Check parameter validity
            params_dict = self._params_to_dict(params)
            
            # Get alpha and beta
            alpha = params_dict.get("alpha", self.alpha_value)
            beta = params_dict.get("beta", self.beta_value)
            
            # Check alpha and beta validity
            if alpha <= 0 or alpha > 2:
                return 1e10  # Penalty for invalid parameters
            if beta < -1 or beta > 1:
                return 1e10  # Penalty for invalid parameters
            
            # Check rho values (they must be in (0,1) for anticipative processes)
            for j in range(self.num_components):
                rho_j = params_dict.get(f"rho_{j+1}", 0)
                if rho_j <= 0 or rho_j >= 1:
                    return 1e10  # Penalty for invalid parameters
            
            # Check varsigma values (they must be positive)
            for j in range(self.num_components):
                varsigma_j = params_dict.get(f"varsigma_{j+1}", 0)
                if varsigma_j <= 0:
                    return 1e10  # Penalty for invalid parameters
            
            # Define grid for (u, v)
            u_grid = np.linspace(-1.5, 1.5, 15) # np.linspace(-0.5, 0.5, 10)  # Explore the vicinity of the origin
            v_grid = np.linspace(-1.5, 1.5, 15) # np.linspace(-0.5, 0.5, 10)  # Explore the vicinity of the origin
            
            # Calculate optimal weights if requested
            if use_optimal_weights:
                # Use preliminary parameters for weight calculation if provided
                params_for_weights = preliminary_params if preliminary_params is not None else params
                weights = self.compute_optimal_weights(Y, params_for_weights, u_grid, v_grid)
            else:
                # Use uniform weights by default
                weights = {(u, v): 1.0 for u in u_grid for v in v_grid}
            
            # Compute objective function
            diff_sum = 0
            count = 0
            
            for u in u_grid:
                for v in v_grid:
                    # Theoretical characteristic function
                    cf_theo = self.char_function(u, v, params)
                    
                    # Empirical characteristic function
                    cf_emp = np.mean(np.exp(1j * (u * Y[1:] + v * Y[:-1])))
                    
                    # Squared difference weighted
                    diff_sum += weights[(u, v)] * np.abs(cf_emp - cf_theo)**2
                    count += 1
            
            return diff_sum / count


    def generate_initial_guess(self) -> List[float]:
            """
            Generates an initial guess for parameter estimation with the most persistent
            component (highest rho) placed first for consistency.
            
            Returns:
                List of initial values for the parameters
            """
            # Initialize arrays for parameters
            rho_values = []
            varsigma_values = []
            
            # Generate initial values for each component
            for j in range(self.num_components):
                # Spread the rho values between 0.8 and 0.2 in decreasing order
                # This ensures the most persistent component is first
                rho_j = 0.8 - 0.6 * j / (self.num_components - 1) if self.num_components > 1 else 0.5
                rho_values.append(rho_j)
                
                # Set initial varsigma values (approximately equal for all components)
                varsigma_values.append(1.0 / self.num_components)
            
            # Combine all initial values
            initial_guess = rho_values + varsigma_values
            
            # Add alpha and beta only if they are not fixed
            if not self.fix_alpha:
                initial_guess.append(self.alpha_value)
            
            if not self.fix_beta:
                initial_guess.append(self.beta_value)
            
            return initial_guess


    def _params_to_dict(self, params: tuple) -> Dict[str, float]:
            """
            Converts a tuple of parameters to a dictionary for easier access.
            Includes derived parameters sigma and pi values.
            Also handles fixed alpha and beta parameters.

            Args:
                params: Tuple of parameters
                    
            Returns:
                Dictionary of parameters including derived values
            """
            params_dict = {}
            params_idx = 0
            
            # Extract parameters for each component
            for j in range(1, self.num_components + 1):
                if params_idx < len(params):
                    params_dict[f"rho_{j}"] = params[params_idx]
                    params_idx += 1
            
            # Extract varsigma parameters
            varsigmas = []
            for j in range(1, self.num_components + 1):
                if params_idx < len(params):
                    params_dict[f"varsigma_{j}"] = params[params_idx]
                    varsigmas.append(params[params_idx])
                    params_idx += 1
            
            # Calculate and add sigma and pi values
            if varsigmas:
                sigma = sum(varsigmas)
                params_dict["sigma"] = sigma
                for j in range(1, self.num_components + 1):
                    if j-1 < len(varsigmas):
                        params_dict[f"pi_{j}"] = varsigmas[j-1] / sigma
            
            # Extract alpha (or use fixed value)
            if not self.fix_alpha:
                if params_idx < len(params):
                    params_dict["alpha"] = params[params_idx]
                    params_idx += 1
                else:
                    params_dict["alpha"] = self.alpha_value
            else:
                params_dict["alpha"] = self.alpha_value
                
            # Extract beta (or use fixed value)
            if not self.fix_beta:
                if params_idx < len(params):
                    params_dict["beta"] = params[params_idx]
                else:
                    params_dict["beta"] = self.beta_value
            else:
                params_dict["beta"] = self.beta_value
            
            return params_dict


    def estimate(self, Y: np.ndarray, initial_guess: Optional[List[float]] = None, 
                use_optimal_weights: bool = True, step1: bool = True) -> tuple:
            """
            Estimates the convolution model parameters using the minimum distance method.

            Args:
                Y: Observed time series
                initial_guess: Initial values for the optimization algorithm
                use_optimal_weights: If True, uses optimal weights for the minimum distance estimator
                    
            Returns:
                Tuple containing the estimated parameters
            """
            # Define bounds based on parameters to estimate
            bounds = []
            
            # Bounds for rho parameters (0, 1) for anticipative processes
            for _ in range(self.num_components):
                bounds.append((0.01, 0.99))
            
            # Bounds for varsigma parameters (positive values)
            for _ in range(self.num_components):
                bounds.append((0.005, 5))
            
            # Bounds for alpha (0, 2) if not fixed
            if not self.fix_alpha:
                bounds.append((0.1, 1.99))
            
            # Bounds for beta (-1, 1) if not fixed
            if not self.fix_beta:
                bounds.append((-0.99, 0.99))
            
            # Generate initial guess if not provided
            if initial_guess is None:
                initial_guess = self.generate_initial_guess()
                
            # STEP 1: Preliminary estimation with uniform weights            
            if step1:
                if self.verbose:
                    print("Step 1: Estimation with uniform weights...")
                    print(f"Initial guess: {initial_guess}")
                
                result_step1 = optimize.minimize(
                    lambda params: self.objective_function(params, Y, use_optimal_weights=False),
                    initial_guess,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100}  # Fewer iterations for preliminary step
                )
                
                if not result_step1.success:
                    if self.verbose:
                        print(f"Warning: Preliminary optimization did not converge: {result_step1.message}")  
                        print("Using parameters from initial values.")  
                    preliminary_params = initial_guess
                else:
                    preliminary_params = tuple(result_step1.x)     
            else:
                    preliminary_params = initial_guess
                    
            # STEP 2: Compute optimal weights based on preliminary estimation if requested
            if self.verbose and use_optimal_weights:
                print("\nEstimated parameters:")
                est_params_dict = self._params_to_dict(preliminary_params)
                for name, value in est_params_dict.items():
                    print(f"  {name} = {value:.4f}")
                print("\nCalculating optimal weights based on the preliminary estimation...")
            
            # Run optimization with optimal weights if requested
            if self.verbose and use_optimal_weights:
                pretty_params = [float(x) for x in preliminary_params]
                print("Starting parameter estimation with optimal weights...")
                print(f"New initial guess: {pretty_params}")
            
            if use_optimal_weights:
                result = optimize.minimize(
                    lambda params: self.objective_function(params, Y, use_optimal_weights=use_optimal_weights, preliminary_params=preliminary_params),
                    preliminary_params,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 200}
                )
                
                if not result.success:
                    if self.verbose:
                        print(f"Warning: Optimal weights optimization did not converge: {result.message}")
                        print("Using parameters from the preliminary step.")
                    self.params = preliminary_params
                    if result_step1.success:
                        final_params = result_step1.x
                    else:
                        final_params = preliminary_params
                else:
                    # Store final estimated parameters
                    final_params = result.x
                    self.params = tuple(final_params)
            else:
                final_params = preliminary_params
                self.params = preliminary_params
                    
            if self.verbose:
                print("Estimation complete.")
                est_params_dict = self._params_to_dict(self.params)
                for name, value in est_params_dict.items():
                    print(f"  {name} = {value:.4f}")
            
            # Calculate sigma and pi values from varsigma
            est_params_dict = self._params_to_dict(self.params)
            varsigmas = [est_params_dict[f"varsigma_{j}"] for j in range(1, self.num_components + 1)]
            sigma = sum(varsigmas)
            pi_values = [varsigma / sigma for varsigma in varsigmas]
            
            if self.verbose:
                print("\nDerived parameters:")
                print(f"  sigma = {sigma:.4f}")
                for j in range(1, self.num_components + 1):
                    print(f"  pi_{j} = {pi_values[j-1]:.4f}")
            
            return self.params


    def inference(self, Y: np.ndarray, params: Optional[tuple] = None, 
                    alpha: float = 0.05) -> pd.DataFrame:
            """
            Calculates inference statistics for the estimated parameters, including
            derived parameters like sigma and pi values.
            
            Args:
                Y: Observed time series
                params: Estimated parameters (uses self.params if None)
                alpha: Significance level (default 0.05 for 95% confidence intervals)
                    
            Returns:
                DataFrame containing estimates, standard errors, t-statistics,
                p-values and confidence intervals
            """
            if params is None:
                if self.params is None:
                    raise ValueError("Parameters must be estimated before calculating inference")
                params = self.params
                
            n = len(Y)
            
            # Define objective function for Hessian calculation
            def obj_func(theta):
                # Convert parameter vector to tuple
                theta_tuple = tuple(theta)
                
                # Calculate objective function
                return self.objective_function(theta_tuple, Y, use_optimal_weights=False)
            
            # Convert params to array for scipy
            params_array = np.array(params)
            
            # Calculate Hessian numerically with a robust approach
            try:
                # Use variable epsilons based on parameters
                epsilons = []
                param_idx = 0
                
                # Epsilon for rho parameters
                for j in range(self.num_components):
                    epsilons.append(max(0.001, min(0.01, params_array[param_idx] * 0.05)))
                    param_idx += 1
                    
                # Epsilon for varsigma parameters
                for j in range(self.num_components):
                    epsilons.append(max(0.001, params_array[param_idx] * 0.05))
                    param_idx += 1
                    
                # Epsilon for alpha (if not fixed)
                if not self.fix_alpha:
                    if param_idx < len(params_array):
                        epsilons.append(max(0.005, min(0.02, params_array[param_idx] * 0.02)))
                        param_idx += 1
                        
                # Epsilon for beta (if not fixed)
                if not self.fix_beta:
                    if param_idx < len(params_array):
                        epsilons.append(max(0.005, min(0.01, abs(params_array[param_idx]) * 0.05)))
                
                # Calculate Hessian with a robust approach
                n_params = len(params_array)
                hessian = np.zeros((n_params, n_params))
                grad = np.zeros(n_params)
                
                # Calculate gradient with central finite differences
                f0 = obj_func(params_array)
                for i in range(n_params):
                    ei = np.zeros(n_params)
                    ei[i] = epsilons[i]
                    
                    # Forward and backward evaluation
                    fi_plus = obj_func(params_array + ei)
                    fi_minus = obj_func(params_array - ei)
                    
                    # Central finite difference
                    grad[i] = (fi_plus - fi_minus) / (2 * epsilons[i])
                
                # Calculate Hessian with second-order finite differences
                for i in range(n_params):
                    ei = np.zeros(n_params)
                    ei[i] = epsilons[i]
                    
                    for j in range(i, n_params):
                        ej = np.zeros(n_params)
                        ej[j] = epsilons[j]
                        
                        # For diagonal terms
                        if i == j:
                            # 3-point formula
                            fi_plus = obj_func(params_array + ei)
                            fi_minus = obj_func(params_array - ei)
                            hessian[i, i] = (fi_plus - 2*f0 + fi_minus) / (epsilons[i] * epsilons[i])
                        else:
                            # More stable cross-derivative formula
                            fij_plus_plus = obj_func(params_array + ei + ej)
                            fij_plus_minus = obj_func(params_array + ei - ej)
                            fij_minus_plus = obj_func(params_array - ei + ej)
                            fij_minus_minus = obj_func(params_array - ei - ej)
                            
                            hessian[i, j] = (fij_plus_plus - fij_plus_minus - fij_minus_plus + fij_minus_minus) / (4 * epsilons[i] * epsilons[j])
                            hessian[j, i] = hessian[i, j]
                
                # Stabilize Hessian if needed
                # Check if Hessian is numerically positive definite
                try:
                    np.linalg.cholesky(hessian)
                except np.linalg.LinAlgError:
                    # If not PD, add small value to diagonal
                    min_eig = np.min(np.linalg.eigvals(hessian))
                    if min_eig < 0:
                        hessian = hessian - 1.2 * min_eig * np.eye(n_params)
                
                # Calculate variance-covariance matrix
                try:
                    vcv_matrix = np.linalg.inv(hessian) / n
                except np.linalg.LinAlgError:
                    # For singular matrix, use regularized pseudo-inverse
                    hessian_reg = hessian + 1e-8 * np.eye(n_params)
                    vcv_matrix = np.linalg.pinv(hessian_reg) / n
                    print("Warning: Hessian matrix near singularity, using regularized matrix.")
            
            except Exception as e:
                print(f"Error calculating Hessian: {e}")
                print("Using an alternative approximation for VCV matrix...")
                
                # Approach based on parameter sensitivity
                n_params = len(params_array)
                vcv_matrix = np.zeros((n_params, n_params))
                
                # Individual parameter variability
                param_idx = 0
                param_names = []
                
                # Add parameters in order
                for j in range(self.num_components):
                    param_names.append(f"rho_{j+1}")
                for j in range(self.num_components):
                    param_names.append(f"varsigma_{j+1}")
                if not self.fix_alpha:
                    param_names.append("alpha")
                if not self.fix_beta:
                    param_names.append("beta")
                
                for i in range(n_params):
                    # Base perturbation to test stability
                    perturb_scale = 0.05 * (1.0 if params_array[i] == 0 else abs(params_array[i]))
                    
                    # Adjust based on parameter type
                    if i < len(param_names) and param_names[i] == "alpha":
                        perturb_scale = min(0.1, perturb_scale)
                    elif i < len(param_names) and param_names[i] == "beta":
                        perturb_scale = min(0.1, perturb_scale)
                    
                    # Calculate diagonal variance
                    vcv_matrix[i, i] = perturb_scale**2
                
                print("Approximate VCV matrix created. Confidence intervals are approximate.")
            
            # Extract standard errors (square root of diagonal elements)
            std_errors = np.sqrt(np.diag(vcv_matrix))
            
            # Check for NaN/Inf standard errors and replace them
            param_names = []
            param_idx = 0
            
            # Define parameter names
            for j in range(self.num_components):
                param_names.append(f"rho_{j+1}")
            for j in range(self.num_components):
                param_names.append(f"varsigma_{j+1}")
            if not self.fix_alpha:
                param_names.append("alpha")
            if not self.fix_beta:
                param_names.append("beta")
            
            for i in range(len(std_errors)):
                if np.isnan(std_errors[i]) or np.isinf(std_errors[i]):
                    param_name = param_names[i] if i < len(param_names) else f"param_{i}"
                    print(f"Warning: Standard error not calculable for {param_name}.")
                    
                    # Approximate value based on parameter
                    if param_name == "alpha":
                        std_errors[i] = 0.2  # Typical value for alpha
                    elif param_name == "beta":
                        std_errors[i] = 0.3  # Typical value for beta
                    else:
                        std_errors[i] = abs(params_array[i]) * 0.2  # 20% of parameter
            
            # Calculate t-statistics
            t_stats = params_array / std_errors
            
            # Calculate p-values (normal approximation)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
            
            # Calculate confidence intervals
            z_critical = stats.norm.ppf(1 - alpha/2)
            lower_ci = params_array - z_critical * std_errors
            upper_ci = params_array + z_critical * std_errors
            
            # Create a DataFrame for base parameter results
            results = pd.DataFrame({
                'Parameter': param_names,
                'Estimate': params_array,
                'Std. Error': std_errors,
                't-statistic': t_stats,
                'p-value': p_values,
                f'CI {100*(1-alpha)}% Lower': lower_ci,
                f'CI {100*(1-alpha)}% Upper': upper_ci
            })
            
            # Extract parameter dictionary
            params_dict = self._params_to_dict(params)
            
            # Calculate derived parameters: sigma and pi values
            varsigmas = []
            for j in range(self.num_components):
                varsigmas.append(params_dict[f"varsigma_{j+1}"])
            
            # Calculate sigma
            sigma = sum(varsigmas)
            
            # Calculate pi values
            pi_values = [varsigma / sigma for varsigma in varsigmas]
            
            # Calculate standard errors for derived parameters using delta method
            # For sigma = sum(varsigma_j)
            sigma_var = 0.0
            varsigma_indices = [self.num_components + i for i in range(self.num_components)]
            
            for i in range(self.num_components):
                for j in range(self.num_components):
                    i_idx = varsigma_indices[i]  # Index for varsigma_i in params_array
                    j_idx = varsigma_indices[j]  # Index for varsigma_j in params_array
                    if i_idx < len(params_array) and j_idx < len(params_array):
                        sigma_var += vcv_matrix[i_idx, j_idx]  # dsigma/dvarsigma_i = 1 for all i
            
            sigma_stderr = np.sqrt(max(0, sigma_var))
            
            # For pi_j = varsigma_j / sigma
            pi_stddev = []
            for j in range(self.num_components):
                pi_j_var = 0.0
                j_idx = varsigma_indices[j]  # Index for varsigma_j
                
                if j_idx < len(params_array):
                    for k in range(self.num_components):
                        k_idx = varsigma_indices[k]  # Index for varsigma_k
                        
                        if k_idx < len(params_array):
                            # Derivative of pi_j with respect to varsigma_j and varsigma_k
                            if j == k:
                                # d(pi_j)/d(varsigma_j) = (sigma - varsigma_j) / sigma^2
                                deriv_j = (sigma - varsigmas[j]) / (sigma**2)
                            else:
                                # d(pi_j)/d(varsigma_k) = -varsigma_j / sigma^2
                                deriv_j = -varsigmas[j] / (sigma**2)
                                
                            deriv_k = (sigma - varsigmas[k]) / (sigma**2) if j == k else -varsigmas[j] / (sigma**2)
                            
                            pi_j_var += deriv_j * deriv_k * vcv_matrix[j_idx, k_idx]
                
                pi_stddev.append(np.sqrt(max(0, pi_j_var)))
            
            # Calculate t-statistics, p-values, and confidence intervals for derived parameters
            # For sigma
            sigma_t = sigma / sigma_stderr if sigma_stderr > 0 else np.nan
            sigma_p = 2 * (1 - stats.norm.cdf(abs(sigma_t))) if not np.isnan(sigma_t) else np.nan
            sigma_lower = sigma - z_critical * sigma_stderr
            sigma_upper = sigma + z_critical * sigma_stderr
            
            # Add sigma to results DataFrame
            sigma_row = pd.DataFrame({
                'Parameter': ['sigma'],
                'Estimate': [sigma],
                'Std. Error': [sigma_stderr],
                't-statistic': [sigma_t],
                'p-value': [sigma_p],
                f'CI {100*(1-alpha)}% Lower': [sigma_lower],
                f'CI {100*(1-alpha)}% Upper': [sigma_upper]
            })
            
            results = pd.concat([results, sigma_row], ignore_index=True)
            
            # For pi values
            for j in range(self.num_components):
                pi_j = pi_values[j]
                pi_j_stderr = pi_stddev[j]
                pi_j_t = pi_j / pi_j_stderr if pi_j_stderr > 0 else np.nan
                pi_j_p = 2 * (1 - stats.norm.cdf(abs(pi_j_t))) if not np.isnan(pi_j_t) else np.nan
                pi_j_lower = pi_j - z_critical * pi_j_stderr
                pi_j_upper = pi_j + z_critical * pi_j_stderr
                
                pi_row = pd.DataFrame({
                    'Parameter': [f'pi_{j+1}'],
                    'Estimate': [pi_j],
                    'Std. Error': [pi_j_stderr],
                    't-statistic': [pi_j_t],
                    'p-value': [pi_j_p],
                    f'CI {100*(1-alpha)}% Lower': [pi_j_lower],
                    f'CI {100*(1-alpha)}% Upper': [pi_j_upper]
                })
                
                results = pd.concat([results, pi_row], ignore_index=True)
            
            # Add alpha and beta to results if they are fixed
            if self.fix_alpha:
                fixed_alpha_row = pd.DataFrame({
                    'Parameter': ['alpha (fixed)'],
                    'Estimate': [self.alpha_value],
                    'Std. Error': [np.nan],
                    't-statistic': [np.nan],
                    'p-value': [np.nan],
                    f'CI {100*(1-alpha)}% Lower': [np.nan],
                    f'CI {100*(1-alpha)}% Upper': [np.nan]
                })
                results = pd.concat([results, fixed_alpha_row], ignore_index=True)
                
            if self.fix_beta:
                fixed_beta_row = pd.DataFrame({
                    'Parameter': ['beta (fixed)'],
                    'Estimate': [self.beta_value],
                    'Std. Error': [np.nan],
                    't-statistic': [np.nan],
                    'p-value': [np.nan],
                    f'CI {100*(1-alpha)}% Lower': [np.nan],
                    f'CI {100*(1-alpha)}% Upper': [np.nan]
                })
                results = pd.concat([results, fixed_beta_row], ignore_index=True)
            
            return results


    def mcmc_filter(self, Y: np.ndarray, params: Optional[tuple] = None,
                            show_progress: bool = True, n_particles: int = 2000,
                            window_size: int = 5, n_jobs: int = 4, 
                            seed: Optional[int] = 42) -> Tuple[List[np.ndarray]]:
        """
        MCMC filter that accounts for the varsigma_j parameters.
        
        Main features:
        1. Marginal distribution with the model alpha (not proposal_alpha)
        2. Scaling by varsigma_j
        3. Adaptive weighting based on component scales
        4. Component-specific MCMC proposal variances
        5. Residual redistribution proportional to varsigma_j
        
        Args:
            Y: Observed time series
            params: Model parameters (uses self.params if None)
            show_progress: If True, shows a progress bar
            n_particles: Number of particles to use for each time point
            window_size: Size of the sliding window for filtering
            n_jobs: Number of parallel processes
            seed: Seed for reproducibility
                
        Returns:
            List of J filtered components [X_1, X_2, ..., X_J]
        """
        
        # Check parameters
        if params is None:
            if self.params is None:
                raise ValueError("Parameters must be estimated before filtering")
            params = self.params
        
        T = len(Y)
        params_dict = self._params_to_dict(params)
        
        # Extract parameters
        alpha = params_dict.get("alpha", self.alpha_value)
        beta = params_dict.get("beta", self.beta_value)
        
        # Extract varsigma values directly (not recalculating sigma and pi)
        varsigmas = []
        rho_values = []
        for j in range(self.num_components):
            varsigmas.append(params_dict[f"varsigma_{j+1}"])
            rho_values.append(params_dict[f"rho_{j+1}"])
        
        # Calculate theoretical scales for each component
        component_scales = []
        for j in range(self.num_components):
            if alpha != 1:
                # Theoretical scale of X_j,t: varsigma_j / (1-|rho_j|^alpha)^(1/alpha)
                theoretical_scale = varsigmas[j] / ((1 - abs(rho_values[j])**alpha)**(1/alpha))
            else:
                # For alpha = 1 (Cauchy case)
                theoretical_scale = varsigmas[j] / (1 - abs(rho_values[j]))
            component_scales.append(theoretical_scale)
        
        if show_progress:
            print(f"Component scales: {[f'{scale:.4f}' for scale in component_scales]}")
            print(f"Varsigma values: {[f'{vs:.4f}' for vs in varsigmas]}")
        
        # Initialize filtered components
        X_filtered = [np.zeros(T) for _ in range(self.num_components)]
        
        # Half window size
        half_window = window_size // 2
        
        # Use the provided seed to initialize random offset
        if seed is not None:
            np.random.seed(seed)
            seed_offset = seed
        else:
            seed_offset = np.random.randint(0, 10000)
        
        def process_timepoint(t_idx, seed_offset):
            """
            Processing of one time point, with scaling by varsigma_j.
            """
            
            # Set non-interactive backend for this worker process
            os.environ['MPLBACKEND'] = 'Agg'
            matplotlib.use('Agg', force=True)
            plt.ioff()
            
            try:
                # Check if the point is in the valid range for window filtering
                if t_idx < half_window or t_idx >= T - half_window:
                    return t_idx, [None] * self.num_components
                
                # Set random seed for reproducibility
                np.random.seed(t_idx * 1000 + seed_offset)
                
                # Extract the window of observations
                y_window = Y[t_idx - half_window:t_idx + half_window + 1]
                
                # Initialize particles for each component in the window
                X_particles = [np.zeros((n_particles, window_size)) for _ in range(self.num_components)]
                weights = np.zeros(n_particles)
                
                # Generate particles from the marginal distribution
                for i in range(n_particles):
                    for j in range(self.num_components):
                        rho_j = rho_values[j]
                        varsigma_j = varsigmas[j]
                        
                        # Generate component center from the marginal distribution
                        if alpha != 1:
                            # Use TRUE alpha, not proposal_alpha
                            marginal_scale = varsigma_j / ((1 - abs(rho_j)**alpha)**(1/alpha))
                            x_center = stats.levy_stable.rvs(
                                alpha=alpha, beta=beta,  # model alpha, not proposal_alpha
                                scale=marginal_scale,
                                loc=0
                            )
                        else:
                            # Special case for alpha = 1
                            marginal_scale = varsigma_j / (1 - abs(rho_j))
                            x_center = stats.levy_stable.rvs(
                                alpha=1.0, beta=beta,
                                scale=marginal_scale,
                                loc=0
                            )
                        
                        # Place at the center of the window
                        X_particles[j][i, half_window] = x_center
                        
                        # Propagate with the innovation scale varsigma_j
                        # Propagate forward
                        for k in range(half_window+1, window_size):
                            innovation = stats.levy_stable.rvs(
                                alpha=alpha, beta=beta, 
                                scale=varsigma_j,  # scale varsigma_j
                                loc=0
                            )
                            X_particles[j][i, k] = rho_j * X_particles[j][i, k-1] + innovation
                        
                        # Propagate backward
                        for k in range(half_window-1, -1, -1):
                            innovation = stats.levy_stable.rvs(
                                alpha=alpha, beta=beta, 
                                scale=varsigma_j,  # scale varsigma_j
                                loc=0
                            )
                            X_particles[j][i, k] = rho_j * X_particles[j][i, k+1] + innovation
                    
                    # Calculate the combined process for this particle
                    combined = np.zeros(window_size)
                    for j in range(self.num_components):
                        combined += X_particles[j][i]
                    
                    # Adaptive particle weighting based on component scales
                    residuals = y_window - combined
                    
                    # Calculate error score
                    if alpha >= 2.0:
                        error_score = np.mean(residuals**2)
                    else:
                        power = min(alpha, 1.5)
                        error_score = np.mean(np.abs(residuals)**power)
                    
                    # Adaptive concentration parameter based on total scale
                    total_scale = sum(varsigmas)
                    adaptive_gamma = 3.0 / max(total_scale, 0.1)  # Inverse scaling
                    
                    # Convert to weight
                    weights[i] = np.exp(-error_score * adaptive_gamma)
                
                # MCMC refinement with component-specific proposal variances
                for mcmc_iter in range(2):  # 2 iterations of MCMC
                    for i in range(n_particles):
                        for j in range(self.num_components):
                            # Store current state
                            current_component = X_particles[j][i].copy()
                            
                            # Component-specific proposal variance
                            # Base proposal variance on varsigma_j
                            proposal_std = max(varsigmas[j] * 0.1, 0.01)  # Adaptive scaling
                            
                            # Propose perturbation to center
                            delta = np.random.normal(0, proposal_std)
                            X_particles[j][i, half_window] += delta
                            
                            # Propagate perturbation
                            rho_j = rho_values[j]
                            
                            # Forward propagation
                            for k in range(half_window+1, window_size):
                                # Keep same innovations but update states
                                old_innovation = current_component[k] - rho_j * current_component[k-1]
                                X_particles[j][i, k] = rho_j * X_particles[j][i, k-1] + old_innovation
                            
                            # Backward propagation
                            for k in range(half_window-1, -1, -1):
                                old_innovation = current_component[k] - rho_j * current_component[k+1]
                                X_particles[j][i, k] = rho_j * X_particles[j][i, k+1] + old_innovation
                            
                            # Calculate acceptance probability
                            new_combined = np.zeros(window_size)
                            old_combined = np.zeros(window_size)
                            
                            for k in range(self.num_components):
                                if k == j:
                                    new_combined += X_particles[k][i]
                                    old_combined += current_component
                                else:
                                    new_combined += X_particles[k][i]
                                    old_combined += X_particles[k][i]
                            
                            new_residuals = y_window - new_combined
                            old_residuals = y_window - old_combined
                            
                            if alpha >= 2.0:
                                new_error = np.mean(new_residuals**2)
                                old_error = np.mean(old_residuals**2)
                            else:
                                power = min(alpha, 1.5)
                                new_error = np.mean(np.abs(new_residuals)**power)
                                old_error = np.mean(np.abs(old_residuals)**power)
                            
                            # Accept or reject
                            if new_error < old_error or np.random.random() < np.exp(adaptive_gamma * (old_error - new_error)):
                                # Accept: keep new state
                                pass
                            else:
                                # Reject: restore old state
                                X_particles[j][i] = current_component
                
                # Normalize weights
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(n_particles) / n_particles
                
                # Calculate filtered values as weighted average for each component
                x_filtered = []
                for j in range(self.num_components):
                    x_est = np.sum(weights * X_particles[j][:, half_window])
                    x_filtered.append(x_est)
                
                return t_idx, x_filtered
                
            finally:
                # Clean up matplotlib objects
                plt.close('all')
                plt.clf()
                plt.cla()
        
        # Process time points in parallel (same as original)
        timepoints = list(range(half_window, T - half_window))
        
        # Determine the number of processes to use
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 4)
        
        if show_progress:
            print(f"Parallel filtering with {n_jobs} processes, {n_particles} particles per time point")
            print(f"Processing {T - 2*half_window} time points...")
        
        # Process time points in parallel
        results = []
        if show_progress:
            with Parallel(n_jobs=n_jobs, backend="loky", prefer="processes") as parallel:
                results = parallel(
                    delayed(process_timepoint)(t, seed_offset) 
                    for t in tqdm(timepoints, desc="Filtering", ncols=80)
                )
        else:
            with Parallel(n_jobs=n_jobs, backend="loky", prefer="processes") as parallel:
                results = parallel(
                    delayed(process_timepoint)(t, seed_offset) for t in timepoints
                )
        
        # Process results
        for t_idx, x_filtered_values in results:
            if None not in x_filtered_values:
                for j in range(self.num_components):
                    X_filtered[j][t_idx] = x_filtered_values[j]
        
        # Process endpoints (same as original)
        if show_progress:
            print("Processing series endpoints...")
        
        for t in range(half_window):
            for j in range(self.num_components):
                X_filtered[j][t] = X_filtered[j][half_window] * (rho_values[j] ** (half_window - t))
        
        for t in range(T - half_window, T):
            for j in range(self.num_components):
                X_filtered[j][t] = X_filtered[j][T - half_window - 1] * (rho_values[j] ** (t - (T - half_window - 1)))
        
        # Optimal residual redistribution based on varsigma_j
        if show_progress:
            print("Optimal final adjustment using varsigma weights...")
        
        # Calculate the combined process
        combined = np.zeros_like(Y)
        for j in range(self.num_components):
            combined += X_filtered[j]
        
        # Calculate residuals
        residual = Y - combined
        
        # Redistribute residuals proportionally to varsigma_j (not magnitudes)
        total_varsigma = sum(varsigmas)
        for j in range(self.num_components):
            weight_j = varsigmas[j] / total_varsigma
            X_filtered[j] += weight_j * residual
        
        if show_progress:
            print("Filtering completed.")
            for j in range(self.num_components):
                component_var = np.var(X_filtered[j])
                print(f"Component {j+1} variance: {component_var:.4f} (varsigma_{j+1}={varsigmas[j]:.4f})")
        
        return X_filtered


    def evaluate_filtering(self, X_true: List[np.ndarray], X_filtered: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluates the quality of filtering for non-causal alpha-stable components.

        Args:
            X_true: List of true alpha-stable components
            X_filtered: List of filtered alpha-stable components
                
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Check if the number of components match
        if len(X_true) != len(X_filtered):
            raise ValueError(f"Number of components mismatch: {len(X_true)} vs {len(X_filtered)}")
        
        num_components = len(X_true)
        
        # Calculate metrics for each component
        for j in range(num_components):
            # Correlation between true and filtered components
            metrics[f"corr_X{j+1}"] = np.corrcoef(X_true[j], X_filtered[j])[0,1]
            
            # Mean Squared Error (MSE)
            metrics[f"mse_X{j+1}"] = np.mean((X_true[j] - X_filtered[j])**2)
            
            # Mean Absolute Error (MAE)
            metrics[f"mae_X{j+1}"] = np.mean(np.abs(X_true[j] - X_filtered[j]))
            
            # Root Mean Squared Error (RMSE)
            metrics[f"rmse_X{j+1}"] = np.sqrt(metrics[f"mse_X{j+1}"])
            
            # Mean Absolute Percentage Error (MAPE) - calculated carefully to avoid division by zero
            # Calculate only for non-zero true values
            non_zero = np.abs(X_true[j]) > 1e-10
            
            if np.sum(non_zero) > 0:
                metrics[f"mape_X{j+1}"] = np.mean(np.abs((X_true[j][non_zero] - X_filtered[j][non_zero]) / X_true[j][non_zero])) * 100
            else:
                metrics[f"mape_X{j+1}"] = np.nan
            
            # Alpha-stable specific metrics - tail behavior
            q95_true = np.percentile(np.abs(X_true[j]), 95)
            
            extremes_true = np.abs(X_true[j]) > q95_true
            extremes_filtered = np.abs(X_filtered[j]) > q95_true
            
            # Check how many extreme events were correctly identified
            if np.sum(extremes_true) > 0:
                metrics[f"extreme_recall_X{j+1}"] = np.sum(extremes_true & extremes_filtered) / np.sum(extremes_true)
            else:
                metrics[f"extreme_recall_X{j+1}"] = np.nan
        
        # Calculate aggregate metrics
        combined_true = np.zeros_like(X_true[0])
        combined_filtered = np.zeros_like(X_filtered[0])
        
        for j in range(num_components):
            combined_true += X_true[j]
            combined_filtered += X_filtered[j]
        
        # Correlation of the combined process
        metrics["corr_combined"] = np.corrcoef(combined_true, combined_filtered)[0,1]
        
        # MSE of the combined process
        metrics["mse_combined"] = np.mean((combined_true - combined_filtered)**2)
        
        # RMSE of the combined process
        metrics["rmse_combined"] = np.sqrt(metrics["mse_combined"])
        
        return metrics


    def plot_components(self, Y: np.ndarray, X_true: List[np.ndarray], X_filtered: List[np.ndarray], 
                    params: Optional[tuple] = None, est_params: Optional[tuple] = None) -> None:
        """
        Plots the true mixed process, its components, and optionally estimated parameters.

        Args:
            Y: Observed mixed process
            X_true: List of true non-causal components
            X_filtered: List of filtered non-causal components
            params: True parameters (optional)
            est_params: Estimated parameters (optional)
        """
        num_components = len(X_true)
        plt.figure(figsize=(12, 9 + 3 * num_components))
        
        # Plot mixed process
        plt.subplot(num_components + 2, 1, 1)
        plt.plot(Y, 'k-', linewidth=1.0, alpha=0.7)
        plt.title('Mixed Process Y')
        plt.grid(True)
        
        # Plot the combined filtered process
        combined_filtered = np.zeros_like(Y)
        for j in range(num_components):
            combined_filtered += X_filtered[j]
            
        plt.subplot(num_components + 2, 1, 2)
        plt.plot(Y, 'k-', label='Original', alpha=0.6)
        plt.plot(combined_filtered, 'g-', label='Filtered')
        plt.title('Original vs. Combined Filtered Process')
        plt.grid(True)
        plt.legend()
        
        # Plot each component
        colors = ['b', 'r', 'g', 'm', 'c', 'y']  # Color list for components
        
        for j in range(num_components):
            plt.subplot(num_components + 2, 1, j + 3)
            plt.plot(X_true[j], colors[j % len(colors)] + '-', label=f'True X{j+1}')
            plt.plot(X_filtered[j], colors[j % len(colors)] + '--', label=f'Filtered X{j+1}')
            
            # Get correlation
            corr = np.corrcoef(X_true[j], X_filtered[j])[0,1]
            
            # Get parameters
            if params is not None:
                params_dict = self._params_to_dict(params)
                rho_j = params_dict[f"rho_{j+1}"]
                
                # Calculate sigma and pi values
                varsigmas = [params_dict[f"varsigma_{k+1}"] for k in range(self.num_components)]
                sigma = sum(varsigmas)
                pi_j = varsigmas[j] / sigma
                
                plt.title(f'Component X{j+1} (ρ{j+1}={rho_j:.2f}, π{j+1}={pi_j:.2f}, Correlation: {corr:.4f})')
            else:
                plt.title(f'Component X{j+1} (Correlation: {corr:.4f})')
                
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # If parameters are provided, print them
        if params is not None or est_params is not None:
            print("Parameters:")
            
            if params is not None:
                params_dict = self._params_to_dict(params)
                print("\nTrue Parameters:")
                
                # Extract varsigma values
                varsigmas = [params_dict[f"varsigma_{j+1}"] for j in range(self.num_components)]
                sigma = sum(varsigmas)
                pi_values = [varsigma / sigma for varsigma in varsigmas]
                
                for j in range(self.num_components):
                    print(f"  rho_{j+1} = {params_dict[f'rho_{j+1}']:.4f}")
                for j in range(self.num_components):
                    print(f"  varsigma_{j+1} = {params_dict[f'varsigma_{j+1}']:.4f}")
                print(f"  alpha = {params_dict['alpha']:.4f}")
                print(f"  beta = {params_dict['beta']:.4f}")
                print(f"  sigma = {sigma:.4f}")
                for j in range(self.num_components):
                    print(f"  pi_{j+1} = {pi_values[j]:.4f}")
            
            if est_params is not None:
                est_params_dict = self._params_to_dict(est_params)
                print("\nEstimated Parameters:")
                
                # Extract varsigma values
                varsigmas = [est_params_dict[f"varsigma_{j+1}"] for j in range(self.num_components)]
                sigma = sum(varsigmas)
                pi_values = [varsigma / sigma for varsigma in varsigmas]
                
                for j in range(self.num_components):
                    print(f"  rho_{j+1} = {est_params_dict[f'rho_{j+1}']:.4f}")
                for j in range(self.num_components):
                    print(f"  varsigma_{j+1} = {est_params_dict[f'varsigma_{j+1}']:.4f}")
                print(f"  alpha = {est_params_dict['alpha']:.4f}")
                print(f"  beta = {est_params_dict['beta']:.4f}")
                print(f"  sigma = {sigma:.4f}")
                for j in range(self.num_components):
                    print(f"  pi_{j+1} = {pi_values[j]:.4f}")
                    
                if params is not None:
                    print("\nParameter Errors:")
                    params_dict = self._params_to_dict(params)
                    for j in range(self.num_components):
                        name = f"rho_{j+1}"
                        error = est_params_dict[name] - params_dict[name]
                        print(f"  {name} error = {error:.4f}")
                    for j in range(self.num_components):
                        name = f"varsigma_{j+1}"
                        error = est_params_dict[name] - params_dict[name]
                        print(f"  {name} error = {error:.4f}")
                    error = est_params_dict["alpha"] - params_dict["alpha"]
                    print(f"  alpha error = {error:.4f}")
                    error = est_params_dict["beta"] - params_dict["beta"]
                    print(f"  beta error = {error:.4f}")
        
        plt.show()


    def compute_optimal_weights(self, Y: np.ndarray, 
                            params: Optional[tuple] = None,
                            u_grid: Optional[ArrayLike] = None, 
                            v_grid: Optional[ArrayLike] = None) -> Dict[Tuple[float, float], float]:
        """
        Calculates optimal weights for the minimum distance estimator
        based on the variance of the empirical characteristic function.

        Args:
            Y: observed time series
            params: parameters estimated in the preliminary step (uses self.params if None)
            u_grid: grid of u points
            v_grid: grid of v points
                
        Returns:
            Dictionary of optimal weights indexed by tuples (u, v)
        """
        if params is None:
            if self.params is None:
                # If no parameters are provided, use uniform weights
                if u_grid is None:
                    u_grid = np.linspace(-1.5, 1.5, 15)
                if v_grid is None:
                    v_grid = np.linspace(-1.5, 1.5, 15)
                return {(u, v): 1.0 for u in u_grid for v in v_grid}
            params = self.params
            
        n = len(Y)
        
        # Define bandwidth parameter for the HAC estimator
        q = int(np.ceil(n**(1/4)))
        
        # Define grid points for (u, v)
        if u_grid is None:
            u_grid = np.linspace(-1.5, 1.5, 15)
            
        if v_grid is None:
            v_grid = np.linspace(-1.5, 1.5, 15)
        
        # Dictionary to store weights
        weights = {}
        
        for u in u_grid:
            for v in v_grid:
                # Calculate empirical characteristic function
                cf_emp = np.mean(np.exp(1j * (u*Y[1:] + v*Y[:-1])))
                
                # Variance component (V_n)
                cf_values = np.exp(1j * (u*Y[1:] + v*Y[:-1]))
                V_n = np.mean(np.abs(cf_values - cf_emp)**2)
                
                # Covariance component (R_n) - HAC method with Bartlett kernel
                R_n = 0
                for h in range(1, min(q+1, n//4)):
                    # Bartlett kernel function: K(x) = 1 - |x| for |x| <= 1
                    kernel_weight = 1 - h/q
                    
                    # Calculate autocovariance at lag h
                    gamma_h = 0
                    for t_idx in range(n-h-1):
                        gamma_h += (cf_values[t_idx] - cf_emp) * np.conj(cf_values[t_idx+h] - cf_emp)
                    gamma_h /= (n-h-1)
                    
                    # Add to R_n
                    R_n += 2 * kernel_weight * np.real(gamma_h)
                
                # Calculate gradient by finite differences
                epsilon = 1e-5
                grad = np.zeros(len(params), dtype=complex)
                
                for i in range(len(params)):
                    params_plus = list(params)
                    params_plus[i] += epsilon
                    params_minus = list(params)
                    params_minus[i] -= epsilon
                    
                    cf_plus = self.char_function(u, v, tuple(params_plus))
                    cf_minus = self.char_function(u, v, tuple(params_minus))
                    
                    grad[i] = (cf_plus - cf_minus) / (2 * epsilon)
                
                # Calculate theoretical variance based on gradients
                grad_norm_squared = np.sum(np.abs(grad)**2)
                
                # Increase weights near origin for better alpha identification
                origin_weight = 1.0
                if max(abs(u), abs(v)) < 0.1:
                    origin_weight = 5.0
                
                # Avoid extreme values
                if grad_norm_squared < 1e-10:
                    var_theo = max(V_n + R_n, 1e-10)
                else:
                    var_theo = max(grad_norm_squared * (V_n + R_n), 1e-10)
                
                # Optimal weight (inversely proportional to variance, adjusted for origin)
                weights[(u, v)] = origin_weight / var_theo
        
        # Avoid extreme weights by normalizing
        values = np.array(list(weights.values()))
        max_weight = np.max(values)
        min_weight = np.min(values)
        
        if max_weight / min_weight > 1e6:
            # Limit max/min ratio to 1e6
            for key in weights:
                weights[key] = min(weights[key], min_weight * 1e6)
        
        # Normalize weights to sum to the number of points
        total = sum(weights.values())
        n_points = len(weights)
        norm_factor = n_points / total
        
        for key in weights:
            weights[key] *= norm_factor
        
        return weights


    def mcsingle(self, sim_idx: int, param_idx: int, params: tuple, T: int) -> Dict:
        """
        Runs a single Monte Carlo simulation for a given parameter set
        
        Args:
            sim_idx: Simulation index
            param_idx: Parameter set index
            params: Model parameters
            T: Sample size
                    
        Returns:
            Dictionary containing simulation results
        """
        result = {}
        
        try:
            # Different seed for each simulation
            seed = sim_idx * 100 + param_idx
            
            # Generate the process
            Y, X_true = self.generate(T, params, seed=seed)
            
            # Estimate parameters
            start_time = time.time()
            est_params = self.estimate(Y)
            estimation_time = time.time() - start_time
            
            # Build parameters dictionary
            est_params_dict = self._params_to_dict(est_params)
            params_dict = self._params_to_dict(params)
            
            # Initialize results dictionary
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'estimation_time': estimation_time,
            }
            
            # Add true and estimated parameters
            for param_name in self.params_names:
                # Get values
                true_value = params_dict.get(param_name, None)
                est_value = est_params_dict.get(param_name, None)
                
                if true_value is not None and est_value is not None:
                    result[f"{param_name}_true"] = true_value
                    result[f"{param_name}_est"] = est_value
                    result[f"{param_name}_error"] = est_value - true_value
                    result[f"{param_name}_rel_error"] = abs(est_value - true_value) / max(abs(true_value), 1e-10)
            
            # Calculate derived parameters
            varsigmas_true = []
            varsigmas_est = []
            
            for j in range(self.num_components):
                varsigmas_true.append(params_dict[f"varsigma_{j+1}"])
                varsigmas_est.append(est_params_dict[f"varsigma_{j+1}"])
            
            sigma_true = sum(varsigmas_true)
            sigma_est = sum(varsigmas_est)
            
            pi_true = [v / sigma_true for v in varsigmas_true]
            pi_est = [v / sigma_est for v in varsigmas_est]
            
            # Add derived parameters to results
            result["sigma_true"] = sigma_true
            result["sigma_est"] = sigma_est
            result["sigma_error"] = sigma_est - sigma_true
            result["sigma_rel_error"] = abs(sigma_est - sigma_true) / max(abs(sigma_true), 1e-10)
            
            for j in range(self.num_components):
                result[f"pi_{j+1}_true"] = pi_true[j]
                result[f"pi_{j+1}_est"] = pi_est[j]
                result[f"pi_{j+1}_error"] = pi_est[j] - pi_true[j]
                result[f"pi_{j+1}_rel_error"] = abs(pi_est[j] - pi_true[j]) / max(abs(pi_true[j]), 1e-10)
            
        except Exception as e:
            print(f"Error in simulation {sim_idx}, parameter set {param_idx}: {e}")
            # Return empty result with simulation IDs
            result = {
                'simulation': sim_idx,
                'param_set': param_idx,
                'error': str(e)
            }
        
        return result


    def mcstudy(self, n_simulations: int = 50, T: int = 500, 
                parameter_sets: Optional[List[tuple]] = None, 
                n_jobs: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs a parallelized Monte Carlo simulation to evaluate model performance.

        Args:
            n_simulations: Number of simulations per parameter set
            T: Sample size
            parameter_sets: List of parameter sets to test
            n_jobs: Number of parallel processes to use (-1 to use all cores)
                
        Returns:
            Tuple (results_df, summary_df) containing:
                results_df: Detailed results for each simulation
                summary_df: Summary statistics for each parameter set
        """
        if parameter_sets is None:
            # Generate default parameter sets based on model configuration
            parameter_sets = []
            
            # Parameter set 1: Components with distinct characteristics
            params1 = []
            # rho values for each component
            for j in range(self.num_components):
                rho_j = 0.3 + (0.5 * j / (self.num_components - 1) if self.num_components > 1 else 0)
                params1.append(rho_j)
            
            # varsigma values for each component (approximately equal)
            for j in range(self.num_components):
                params1.append(1.0 / self.num_components)
            
            # alpha and beta
            params1.extend([self.alpha_value, self.beta_value])
            parameter_sets.append(tuple(params1))
            
            # Parameter set 2: Different configuration
            params2 = []
            # rho values - reversed order from params1
            for j in range(self.num_components):
                rho_j = 0.8 - (0.5 * j / (self.num_components - 1) if self.num_components > 1 else 0)
                params2.append(rho_j)
            
            # varsigma values - skewed distribution
            total = 0
            for j in range(self.num_components):
                total += (j + 1)
            
            for j in range(self.num_components):
                params2.append((j + 1) / total)
            
            # alpha and beta - slightly different
            params2.extend([min(self.alpha_value + 0.2, 1.95), max(self.beta_value + 0.1, -0.9)])
            parameter_sets.append(tuple(params2))
        
        # Create task list
        tasks = [(sim_idx, param_idx, params, T) 
                for sim_idx in range(n_simulations) 
                for param_idx, params in enumerate(parameter_sets)]
        
        # Configure CPU usage
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 4)
        
        # Execute tasks in parallel
        print(f"Launching {len(tasks)} simulations on {n_jobs} cores...")
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.mcsingle)(sim_idx, param_idx, params, T)
            for sim_idx, param_idx, params, T in tasks
        )
        
        # Combine results into a DataFrame
        results_df = pd.DataFrame(results)
        
        # Process summary statistics
        summary_rows = []
        
        for param_set in sorted(results_df['param_set'].unique()):
            subset = results_df[results_df['param_set'] == param_set]
            
            # Dictionary to store results for this parameter set
            param_results = {'param_set': param_set}
            
            # Get all parameter names including derived parameters
            all_param_names = []
            for name in self.params_names:
                all_param_names.append(name)
            
            # Add sigma and pi_j parameters
            all_param_names.append("sigma")
            for j in range(1, self.num_components + 1):
                all_param_names.append(f"pi_{j}")
            
            # Process each parameter
            for param_name in all_param_names:
                true_col = f"{param_name}_true"
                est_col = f"{param_name}_est"
                error_col = f"{param_name}_error"
                rel_error_col = f"{param_name}_rel_error"
                
                if true_col in subset.columns:
                    # True value (constant for this parameter set)
                    param_results[f"{param_name}_true"] = subset[true_col].iloc[0]
                    
                    # Mean and std of estimates
                    if est_col in subset.columns:
                        param_results[f"{param_name}_mean"] = subset[est_col].mean()
                        param_results[f"{param_name}_std"] = subset[est_col].std()
                    
                    # Mean bias
                    if error_col in subset.columns:
                        param_results[f"{param_name}_bias"] = subset[error_col].mean()
                    
                    # RMSE
                    if est_col in subset.columns:
                        true_value = subset[true_col].iloc[0]
                        rmse = np.sqrt(np.mean((subset[est_col] - true_value)**2))
                        param_results[f"{param_name}_rmse"] = rmse
                    
                    # Mean relative error
                    if rel_error_col in subset.columns:
                        param_results[f"{param_name}_rel_error"] = subset[rel_error_col].mean()
            
            # Mean estimation time
            if 'estimation_time' in subset.columns:
                param_results['estimation_time'] = subset['estimation_time'].mean()
            
            summary_rows.append(param_results)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        return results_df, summary_df
    
    
    def analyze_results(self, results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                    plot: bool = True, save_plot: bool = False) -> pd.DataFrame:
        """
        Analyzes and visualizes the results of the Monte Carlo simulation
        
        Args:
            results_df: DataFrame of detailed results
            summary_df: DataFrame of summary results
            plot: If True, generates plots
            save_plot: If True, saves plots to files
                    
        Returns:
            DataFrame with formatted summary statistics
        """
        # Print summary of parameters
        model_type = f"{self.num_components}-component model"
        model_type += f" with alpha={self.alpha_value:.2f}, beta={self.beta_value:.2f}"
        
        print(f"\n=== Summary of parameter estimation ({model_type}) ===")
        
        # Create formatted summary table
        formatted_summary = pd.DataFrame()
        
        for param_set in summary_df['param_set'].unique():
            subset = summary_df[summary_df['param_set'] == param_set]
            
            # Process all parameters including derived parameters
            all_param_names = []
            for name in self.params_names:
                all_param_names.append(name)
            
            # Add sigma and pi_j parameters
            all_param_names.append("sigma")
            for j in range(1, self.num_components + 1):
                all_param_names.append(f"pi_{j}")
            
            for param_name in all_param_names:
                true_col = f"{param_name}_true"
                mean_col = f"{param_name}_mean"
                std_col = f"{param_name}_std"
                bias_col = f"{param_name}_bias"
                rmse_col = f"{param_name}_rmse"
                
                if true_col in subset.columns and mean_col in subset.columns:
                    row_name = f"Set {param_set+1} - {param_name}"
                    formatted_summary.loc[row_name, 'True'] = subset[true_col].iloc[0]
                    formatted_summary.loc[row_name, 'Mean Est.'] = subset[mean_col].iloc[0]
                    
                    if std_col in subset.columns:
                        formatted_summary.loc[row_name, 'Std Dev'] = subset[std_col].iloc[0]
                        
                    if bias_col in subset.columns:
                        formatted_summary.loc[row_name, 'Bias'] = subset[bias_col].iloc[0]
                        
                    if rmse_col in subset.columns:
                        formatted_summary.loc[row_name, 'RMSE'] = subset[rmse_col].iloc[0]
            
            # Add average estimation time
            if 'estimation_time' in subset.columns:
                row_name = f"Set {param_set+1} - Time"
                formatted_summary.loc[row_name, 'Mean Est.'] = subset['estimation_time'].iloc[0]
        
        print(formatted_summary)
        
        # Plot results if requested
        if plot:
            self._plot_mc_results(results_df, save_plot)
        
        return formatted_summary


    def _plot_mc_results(self, results_df: pd.DataFrame, save_plot: bool = False) -> None:
        """
        Visualizes the Monte Carlo results with parameter distribution plots
        
        Args:
            results_df: DataFrame of detailed results
            save_plot: If True, saves plots to files
        """
        # Define Greek symbols for parameters
        greek_symbols = {
            "rho": r"$\rho$",
            "varsigma": r"$\varsigma$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "sigma": r"$\sigma$",
            "pi": r"$\pi$"
        }
        
        # Create a separate plot for each parameter set
        for param_set in sorted(results_df['param_set'].unique()):
            # Filter data for this parameter set
            subset = results_df[results_df['param_set'] == param_set]
            
            # Skip if there are errors in this parameter set
            if 'error' in subset.columns and not subset['error'].isna().all():
                print(f"Skipping visualization for parameter set {param_set} due to errors")
                continue
            
            # Get all parameter names including derived parameters
            param_names = []
            for name in self.params_names:
                if f"{name}_true" in subset.columns and f"{name}_est" in subset.columns:
                    param_names.append(name)
            
            # Add sigma
            if "sigma_true" in subset.columns and "sigma_est" in subset.columns:
                param_names.append("sigma")
            
            # Add pi parameters
            for j in range(1, self.num_components + 1):
                if f"pi_{j}_true" in subset.columns and f"pi_{j}_est" in subset.columns:
                    param_names.append(f"pi_{j}")
            
            if not param_names:
                continue
            
            # Determine subplot layout - try to make it more square-like
            n_params = len(param_names)
            n_cols = min(4, n_params)
            n_rows = int(np.ceil(n_params / n_cols))
            
            # Create figure
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            
            # Ensure axes is always a 2D array
            if n_params == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = np.array([axes])
            
            # Extract true parameters for the plot title
            true_params = {}
            for name in param_names:
                true_params[name] = subset[f"{name}_true"].iloc[0]
            
            # Plot each parameter
            for i, param_name in enumerate(param_names):
                ax = axes[i // n_cols, i % n_cols]
                
                # Get data
                param_true = subset[f"{param_name}_true"].iloc[0]
                param_est = subset[f"{param_name}_est"].values
                
                # Compute statistics
                param_mean = np.mean(param_est)
                param_std = np.std(param_est)
                
                # Create histogram
                ax.hist(param_est, bins=min(10, len(param_est) // 5), alpha=0.7)
                
                # Add true value as vertical line
                ax.axvline(param_true, color='red', linestyle='dashed', linewidth=2, label='True value')
                
                # Add mean estimated value as vertical line
                ax.axvline(param_mean, color='green', linestyle='dotted', linewidth=2, label='Mean estimate')
                
                # Add labels
                # Extract the base parameter name (e.g., "rho_1" -> "rho")
                base_param = param_name.split('_')[0] if '_' in param_name else param_name
                # Get the Greek symbol
                symbol = greek_symbols.get(base_param, param_name)
                # Add the subscript if needed
                if '_' in param_name:
                    suffix = param_name.split('_')[1]
                    symbol = f"{symbol}_{{{suffix}}}"
                
                ax.set_title(f'{symbol} (true: {param_true:.2f}, est: {param_mean:.2f}±{param_std:.2f})')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
                # Add grid and legend
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Remove unused subplots
            for i in range(n_params, n_rows * n_cols):
                fig.delaxes(axes[i // n_cols, i % n_cols])
            
            # Set title for the whole figure
            plt.suptitle(f'Parameter Set {param_set+1} - Monte Carlo Results', fontsize=14)
            
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
            
            # Save or show plot
            if save_plot:
                filename = f"mc_results_J{self.num_components}_set{param_set+1}.png"
                plt.savefig(filename, dpi=300)
                print(f"Saved figure to {filename}")
            else:
                plt.show()


    def smart_demo(self, params: tuple, T: int = 200, seed: int = 42, 
                    show_progress: bool = True, n_particles: int = 1000) -> None:
        """
        Generates and visualizes a sample of alpha-stable mixture filtering
        with metric evaluation and method comparison.

        Args:
            params: Model parameters
            T: Sample size
            seed: Seed for reproducibility
            show_progress: Show progress bars during filtering
            n_particles: Number of particles for filtering
        """
        # Generate the process
        Y, X_true = self.generate(T, params, seed=seed)
        
        # Estimate parameters
        initial_guess = self.generate_initial_guess()
        est_params = self.estimate(Y, initial_guess=initial_guess)
        
        # Display parameters
        params_dict = self._params_to_dict(params)
        est_params_dict = self._params_to_dict(est_params)
        
        print("True Parameters:")
        # Extract varsigma values
        varsigmas_true = [params_dict[f"varsigma_{j+1}"] for j in range(self.num_components)]
        sigma_true = sum(varsigmas_true)
        pi_values_true = [varsigma / sigma_true for varsigma in varsigmas_true]
        
        for j in range(self.num_components):
            print(f"  rho_{j+1} = {params_dict[f'rho_{j+1}']:.4f}")
        for j in range(self.num_components):
            print(f"  varsigma_{j+1} = {params_dict[f'varsigma_{j+1}']:.4f}")
        print(f"  alpha = {params_dict['alpha']:.4f}")
        print(f"  beta = {params_dict['beta']:.4f}")
        print(f"  sigma = {sigma_true:.4f}")
        for j in range(self.num_components):
            print(f"  pi_{j+1} = {pi_values_true[j]:.4f}")
                
        print("\nEstimated Parameters:")
        # Extract varsigma values
        varsigmas_est = [est_params_dict[f"varsigma_{j+1}"] for j in range(self.num_components)]
        sigma_est = sum(varsigmas_est)
        pi_values_est = [varsigma / sigma_est for varsigma in varsigmas_est]
        
        for j in range(self.num_components):
            print(f"  rho_{j+1} = {est_params_dict[f'rho_{j+1}']:.4f}")
        for j in range(self.num_components):
            print(f"  varsigma_{j+1} = {est_params_dict[f'varsigma_{j+1}']:.4f}")
        print(f"  alpha = {est_params_dict['alpha']:.4f}")
        print(f"  beta = {est_params_dict['beta']:.4f}")
        print(f"  sigma = {sigma_est:.4f}")
        for j in range(self.num_components):
            print(f"  pi_{j+1} = {pi_values_est[j]:.4f}")
        
        # Start filtering
        print("\nStarting filtering...")
        start_time = time.time()
        
        X_filtered = self.mcmc_filter(
                            Y, params,
                            n_particles=n_particles,
                            window_size=5,
                            n_jobs=4,
                            show_progress=show_progress
                        )
        
        end_time = time.time()
        print(f"Filtering completed in {end_time - start_time:.2f} seconds.")

        
        # Evaluate filtering performance
        metrics = self.evaluate_filtering(X_true, X_filtered)
        
        print("\nFiltering Performance:")
        for key, value in metrics.items():
            if not np.isnan(value):
                print(f"  {key} = {value:.4f}")
        
        # Visualize results
        self.plot_components(Y, X_true, X_filtered, params, est_params)
        
        # Find an extreme event for zoom
        # Combine the true components to find the maximum
        combined_true = np.zeros_like(Y)
        for j in range(self.num_components):
            combined_true += np.abs(X_true[j])
        
        # Find the peak index
        peak_idx = np.argmax(combined_true)
        
        # Create a window around the extreme event
        start_idx = max(0, peak_idx - 20)
        end_idx = min(T, peak_idx + 20)
        
        # Plot zoomed view
        plt.figure(figsize=(15, 12))
        
        # Combined process - zoom
        plt.subplot(self.num_components + 1, 1, 1)
        plt.plot(range(start_idx, end_idx), Y[start_idx:end_idx], 'k-')
        plt.title('Mixed Process - Zoom on Extreme Event')
        plt.grid(True)
        
        # Each component - zoom
        colors = ['b', 'r', 'g', 'm', 'c', 'y']  # Color list for components
        
        for j in range(self.num_components):
            plt.subplot(self.num_components + 1, 1, j + 2)
            plt.plot(range(start_idx, end_idx), X_true[j][start_idx:end_idx], colors[j % len(colors)] + '-', label='True')
            plt.plot(range(start_idx, end_idx), X_filtered[j][start_idx:end_idx], colors[j % len(colors)] + '--', label='Filtered')
            
            plt.title(f'Component X{j+1} - Zoom (ρ{j+1}={params_dict[f"rho_{j+1}"]:.2f})')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()