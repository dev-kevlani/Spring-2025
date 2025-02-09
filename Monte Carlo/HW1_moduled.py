"""
Financial Engineering Simulation Project
Contains Problems 1-6 with detailed documentation
"""

# -------------------------------
# Import Section
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom

# -------------------------------
# Problem 1: Binomial Model Analysis
# -------------------------------
class BinomialModelAnalyzer:
    """
    Class for analyzing convergence properties of binomial option pricing model
    
    Attributes:
        r (float): Risk-free rate
        sigma (float): Volatility parameter
        S0 (float): Initial stock price
        mu_values (list): List of drift parameters to analyze
        config (dict): Configuration parameters for different analyses
    """
    
    def __init__(self):
        """Initialize parameters for binomial model analysis"""
        self.r = 0.05
        self.sigma = 0.4
        self.S0 = 100
        self.mu_values = [-0.2, 0, 0.2, self.r - (self.sigma**2)/2]
        self.config = {
            'N_max_part1': 10000,    # Maximum N for pN convergence analysis
            'N_part3': 100000,       # Time steps for log-return analysis
            'samples_part3': 10**6,  # Monte Carlo samples for distribution analysis
            'K': 100,                # Strike price for option pricing
            'T': 1                   # Time horizon for option pricing
        }

    def compute_risk_neutral_prob(self, N, mu):
        """
        Compute risk-neutral probability for binomial model
        
        Args:
            N (int): Number of time steps
            mu (float): Drift parameter
            
        Returns:
            float: Risk-neutral probability pN
        """
        # Calculate up/down factors using CRR parametrization
        u_N = np.exp(mu/N + self.sigma * np.sqrt(1/N))  # Up factor
        d_N = 1 / u_N                                   # Down factor
        pN = (np.exp(self.r/N) - d_N)/(u_N - d_N)       # Risk-neutral probability
        return pN

    def analyze_pN_convergence(self):
        """
        Analyze convergence of risk-neutral probabilities and their variance
        
        Returns:
            dict: Dictionary containing convergence results for all mu values
        """
        N_values = np.arange(1, self.config['N_max_part1']+1)
        results = {mu: {'pN': [], 'pN_var': []} for mu in self.mu_values}

        # Calculate pN and pN(1-pN) for each mu and N
        for mu in self.mu_values:
            for N in N_values:
                pN = self.compute_risk_neutral_prob(N, mu)
                results[mu]['pN'].append(pN)
                results[mu]['pN_var'].append(pN*(1-pN))

        # Plot results in 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Convergence of Risk-Neutral Probabilities', y=1.02)
        for ax, mu in zip(axes.flatten(), self.mu_values):
            ax.plot(N_values, results[mu]['pN'], label=r'$p_N$')
            ax.plot(N_values, results[mu]['pN_var'], label=r'$p_N(1-p_N)$')
            ax.set_title(fr'$\mu = {mu}$', pad=12)
            ax.set_xlabel('Number of Time Steps (N)')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return results

    def analyze_convergence_rates(self, target_mu=0):
        """
        Analyze convergence rates using log-log regression
        
        Args:
            target_mu (float): Specific mu value to analyze
            
        Returns:
            tuple: Estimated convergence rates (k1, k2)
        """
        N_values = np.arange(1, self.config['N_max_part1']+1)
        pN_values = [self.compute_risk_neutral_prob(N, target_mu) for N in N_values]
        pN_var_values = [p*(1-p) for p in pN_values]

        # Filter valid values where 0 < pN < 1
        valid_mask = (np.array(pN_values) > 0) & (np.array(pN_values) < 1)
        valid_N = N_values[valid_mask]
        
        # Calculate logarithmic deviations from theoretical limits
        log_N = np.log(1/valid_N)
        log_pN_dev = np.log(np.abs(np.array(pN_values)[valid_mask] - 0.5))
        log_pNvar_dev = np.log(np.abs(np.array(pN_var_values)[valid_mask] - 0.25))

        # Perform linear regression to estimate convergence rates
        k1, _ = np.polyfit(log_N, log_pN_dev, 1)
        k2, _ = np.polyfit(log_N, log_pNvar_dev, 1)
        
        # Plot convergence diagnostics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(log_N, log_pN_dev, s=10, alpha=0.7)
        ax1.set_title(fr'$p_N$ Convergence ($k_1$ = {k1:.2f})')
        ax1.set_xlabel(r'$\log(1/N)$')
        ax1.set_ylabel(r'$\log|p_N - 0.5|$')
        
        ax2.scatter(log_N, log_pNvar_dev, s=10, alpha=0.7, color='orange')
        ax2.set_title(fr'$p_N(1-p_N)$ Convergence ($k_2$ = {k2:.2f})')
        ax2.set_xlabel(r'$\log(1/N)$')
        ax2.set_ylabel(r'$\log|p_N(1-p_N) - 0.25|$')
        plt.tight_layout()
        plt.show()
        
        return k1, k2

    def analyze_log_return_distribution(self):
        """
        Analyze convergence of log-return distribution to normal
        
        Returns:
            tuple: (empirical returns, theoretical parameters)
        """
        N = self.config['N_part3']
        samples = self.config['samples_part3']
        mu = self.r - (self.sigma**2)/2  # Theoretical drift
        
        # Generate binomial model parameters
        u = np.exp(mu/N + self.sigma/np.sqrt(N))
        d = 1/u
        p = self.compute_risk_neutral_prob(N, mu)
        
        # Simulate terminal log returns
        up_moves = binom.rvs(N, p, size=samples)
        log_returns = up_moves*np.log(u) + (N - up_moves)*np.log(d)
        
        # Compare with theoretical normal distribution
        plt.figure(figsize=(10, 6))
        plt.hist(log_returns, bins=100, density=True, alpha=0.6, 
                label='Empirical Distribution')
        x = np.linspace(mu-3*self.sigma, mu+3*self.sigma, 100)
        plt.plot(x, norm.pdf(x, mu, self.sigma), 'r--', lw=2, 
                label='Theoretical Normal')
        plt.title('Log-Return Distribution Convergence')
        plt.xlabel('Log Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return log_returns, (mu, self.sigma)

    def compare_option_pricing(self):
        """
        Compare binomial model option prices with Black-Scholes
        
        Returns:
            dict: Pricing results for different mu values
        """
        def black_scholes_price(S0, K, T, r, sigma):
            """Compute Black-Scholes call price"""
            d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        results = {}
        bs_price = black_scholes_price(self.S0, self.config['K'], 
                                      self.config['T'], self.r, self.sigma)
        
        for mu in self.mu_values:
            # Binomial model parameters
            u = np.exp(mu/self.config['N_part3'] + self.sigma/np.sqrt(self.config['N_part3']))
            d = 1/u
            p = self.compute_risk_neutral_prob(self.config['N_part3'], mu)
            
            # Simulate terminal stock prices
            up_moves = binom.rvs(self.config['N_part3'], p, size=self.config['samples_part3'])
            S_T = self.S0 * (u**up_moves) * (d**(self.config['N_part3'] - up_moves))
            payoff = np.maximum(S_T - self.config['K'], 0)
            mc_price = np.exp(-self.r*self.config['T'])*np.mean(payoff)
            
            results[mu] = {
                'binomial_price': mc_price,
                'black_scholes': bs_price,
                'std_error': np.std(payoff)/np.sqrt(self.config['samples_part3'])
            }
        
        # Plot convergence for one mu value
        cumulative_means = np.cumsum(payoff)/np.arange(1, len(payoff)+1)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_means, label='Monte Carlo Estimate')
        plt.axhline(bs_price, color='r', linestyle='--', label='Black-Scholes')
        plt.title('Option Price Convergence')
        plt.xlabel('Number of Simulations')
        plt.ylabel('Call Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return results

# -------------------------------
# Problem 2: Monte Carlo Integration
# -------------------------------
def analyze_confidence_intervals(N=1000, samples=10000):
    """
    Analyze coverage of 95% CIs for π/4 estimation
    
    Args:
        N (int): Number of points per Monte Carlo estimate
        samples (int): Number of confidence intervals to generate
        
    Returns:
        float: Proportion of CIs containing true value
    """
    true_value = np.pi/4
    coverage = 0
    
    for _ in range(samples):
        # Generate random points in unit square
        X, Y = np.random.rand(2, N)
        
        # Calculate proportion in unit circle
        in_circle = (X**2 + Y**2 <= 1).mean()
        
        # Compute 95% confidence interval
        std_err = np.sqrt(in_circle*(1 - in_circle)/N)
        lower = in_circle - 1.96*std_err
        upper = in_circle + 1.96*std_err
        
        coverage += (lower <= true_value <= upper)
    
    plt.figure(figsize=(8, 5))
    plt.bar(['Coverage', 'Miss'], [coverage/samples, 1 - coverage/samples])
    plt.title('95% Confidence Interval Coverage Probability')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    return coverage/samples

# -------------------------------
# Problem 3: Pareto Convergence
# -------------------------------
def analyze_pareto_convergence(N=10**6, reps=10, alphas=[0.5, 1.5, 3], b=1):
    """
    Analyze convergence of moments for Pareto distribution
    
    Args:
        N (int): Number of samples per repetition
        reps (int): Number of independent repetitions
        alphas (list): List of shape parameters
        b (float): Scale parameter
        
    Returns:
        dict: Contains convergence paths for all parameters
    """
    results = {a: {'means': [], 'stds': []} for a in alphas}
    
    for _ in range(reps):
        for alpha in alphas:
            # Generate Pareto samples using inverse transform
            u = np.random.uniform(size=N)
            samples = b/(1 - u)**(1/alpha)
            
            # Calculate cumulative statistics
            cumulative_mean = np.cumsum(samples)/np.arange(1, N+1)
            cumulative_std = np.sqrt(
                (np.cumsum(samples**2) - cumulative_mean**2*np.arange(1, N+1)) / 
                np.arange(0, N)
            )
            
            results[alpha]['means'].append(cumulative_mean)
            results[alpha]['stds'].append(cumulative_std)
    
    # Plot convergence paths
    fig, axes = plt.subplots(2, len(alphas), figsize=(18, 10))
    for i, alpha in enumerate(alphas):
        # Plot means
        ax = axes[0,i]
        for trace in results[alpha]['means']:
            ax.plot(trace, alpha=0.5, lw=1)
        ax.set_title(fr'α = {alpha} - Sample Means')
        ax.grid(True, alpha=0.3)
        
        # Plot standard deviations
        ax = axes[1,i]
        for trace in results[alpha]['stds']:
            ax.plot(trace, alpha=0.5, lw=1)
        ax.set_title(fr'α = {alpha} - Sample Standard Deviations')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return results

# -------------------------------
# Problem 4: St. Petersburg Paradox
# -------------------------------
def analyze_st_petersburg(N=10**6, reps=100):
    """
    Analyze St. Petersburg paradox through simulation
    
    Args:
        N (int): Number of games per estimate
        reps (int): Number of independent estimates
        
    Returns:
        list: List of expected value estimates
    """
    estimates = []
    for _ in range(reps):
        # Simulate number of heads before first tail
        trials = np.random.geometric(p=0.5, size=N)
        # Calculate payouts: 2^(number of heads)
        payouts = 2**(trials - 1)
        estimates.append(payouts.mean())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(estimates, 'o-', alpha=0.7)
    plt.axhline(20, color='r', linestyle='--', 
               label='$20 Guaranteed')
    plt.title('St. Petersburg Paradox: Expected Value Estimates')
    plt.xlabel('Simulation Batch')
    plt.ylabel('Expected Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return estimates

# -------------------------------
# Problem 6: Card Matching Problem
# -------------------------------
def analyze_card_matching(max_power=6):
    """
    Analyze expectation and variance of card matches
    
    Args:
        max_power (int): Maximum power of 10 for sample sizes
        
    Returns:
        dict: Contains expectation and variance estimates
    """
    N_values = [10**k for k in range(1, max_power+1)]
    results = {'Expectation': [], 'Variance': []}
    
    for N in N_values:
        # Simulate N shuffles and count matches
        matches = np.zeros(N)
        for i in range(N):
            # Generate random permutation and count matches
            deck = np.random.permutation(100)
            matches[i] = (deck == np.arange(100)).sum()
        
        # Store statistics
        results['Expectation'].append(matches.mean())
        results['Variance'].append(matches.var(ddof=1))
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogx(N_values, results['Expectation'], 'bo-')
    ax1.axhline(1, color='r', linestyle='--')
    ax1.set_title('Expectation Convergence')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Expected Matches')
    
    ax2.semilogx(N_values, results['Variance'], 'go-')
    ax2.axhline(1, color='r', linestyle='--')
    ax2.set_title('Variance Convergence')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Variance')
    
    plt.tight_layout()
    return results

# -------------------------------
# Main Execution
# -------------------------------
# Initialize analyzer for Problem 1
bma = BinomialModelAnalyzer()

# Problem 1 Analyses
p1_results = bma.analyze_pN_convergence()
k1, k2 = bma.analyze_convergence_rates()
log_returns, (mu, sigma) = bma.analyze_log_return_distribution()
option_results = bma.compare_option_pricing()

print(option_results)

# Problem 2 Analysis
# ci_coverage = analyze_confidence_intervals()
# print(f"\nProblem 2: {ci_coverage:.2%} of CIs contained π/4")

# # Problem 3 Analysis
# pareto_results = analyze_pareto_convergence()

# # Problem 4 Analysis
# st_pete_results = analyze_st_petersburg()

# # Problem 6 Analysis
# card_results = analyze_card_matching()

# plt.show()