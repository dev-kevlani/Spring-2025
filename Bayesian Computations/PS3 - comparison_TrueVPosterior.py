import random
import numpy as np
import scipy.stats as stats
from scipy.stats import beta, gamma, norm, invgamma, pareto
import matplotlib.pyplot as plt

class Comparing_Posterior_Distributions:

    def __init__(self, N_experiments):
        self.N_experiments = N_experiments
        self.u = np.random.uniform(0, 1, self.N_experiments)
    
    def posteriorBeta(self, n, successes, a_prior, b_prior):
        np.random.seed(42)

        a_post = a_prior + successes
        b_post = b_prior + n - successes

        u_beta = np.random.uniform(0, 1, self.N_experiments)
        posterior_samples = beta.ppf(u_beta, a_post, b_post)

        plt.figure()
        x_beta = np.linspace(0, 1, 1000)
        plt.hist(posterior_samples, bins=50, density=True, label='Posterior Samples')
        plt.plot(x_beta, beta.pdf(x_beta, a_post, b_post), label='True Beta Density')
        plt.title('Beta Posterior')
        plt.xlabel('p')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def posterior_gamma(self, n, sum_x, a_prior, b_prior):
        np.random.seed(42)

        a_post = a_prior + sum_x
        b_post = b_prior + n

        posterior_samples = gamma.ppf(self.u, a = a_post, scale = b_post)
        x_gamma = np.linspace(0, gamma.ppf(0.99, a=a_post, scale=b_post), 100)
        plt.figure(figsize=(8, 5))
        plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, label='Sampled Gamma')
        plt.plot(x_gamma, gamma.pdf(x_gamma, a=a_post, scale=b_post), 'r-', lw=2, label='True Gamma PDF')
        plt.title('Gamma Posterior')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def posterior_normal(self, n, x_bar, sigma2, mu0, sigma02):
        np.random.seed(42)

        sigma2_n = 1 / (1/sigma02 + n/sigma2)
        mu_n = sigma2_n * (mu0/sigma02 + n*x_bar/sigma2)

        posterior_samples = norm.ppf(self.u, loc=mu_n, scale=np.sqrt(sigma2_n))
        x = np.linspace(min(posterior_samples), max(posterior_samples), 200)

        plt.figure(figsize=(8, 5))
        plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, label='Sampled Normal')

        y = stats.norm.pdf(x, mu_n, np.sqrt(sigma2_n))
        
        plt.plot(x, y, 'r-', label='True Density')
        plt.title('Normal Posterior')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def posterior_inverseGamma(self, n, sum_squared_dev, alpha, beta):
        np.random.seed(42)

        alpha_post = alpha + n/2
        beta_post = beta + sum_squared_dev/2

        posterior_samples = invgamma.ppf(self.u, alpha_post, scale=beta_post)
        x_invgamma = np.linspace(-5, 5, 100)
        x_invgamma = np.linspace(invgamma.ppf(0.01, alpha_post, scale=beta_post),
                         invgamma.ppf(0.99, alpha_post, scale=beta_post), 1000)
        plt.figure(figsize=(8, 5))
        plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, label='Sampled Inverse Gamma')
        plt.plot(x_invgamma, invgamma.pdf(x_invgamma, alpha_post, scale=beta_post), 'r-', lw=2, label='True Inverse Gamma PDF')
        plt.title('Inverse Gamma Posterior')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def posterior_pareto(self, n, max_x, xm, alpha):
        np.random.seed(42)

        xm_post = max(xm, max_x)
        alpha_post = alpha + n

        posterior_samples = xm_post / (1-self.u)**(1/alpha_post)
        plt.figure(figsize=(8, 5))
        plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, label='Sampled Pareto')
        x = np.linspace(xm_post, np.percentile(posterior_samples, 99), 200)
        y = alpha_post * xm_post**alpha_post / x**(alpha_post+1)
        plt.plot(x, y, 'r-', lw=2, label='True Pareto PDF')
        plt.title('Pareto Posterior')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
    

if __name__ == '__main__':
    distributions = Comparing_Posterior_Distributions(N_experiments=100000)
    distributions.posteriorBeta(n = 20, successes = 12, a_prior = 2, b_prior = 3)
    distributions.posterior_gamma(n = 50, sum_x = 0.9, a_prior = 0.5, b_prior = 2)
    distributions.posterior_normal(n = 20, x_bar = 5, sigma2 = 1, mu0=0, sigma02=4)
    distributions.posterior_inverseGamma(n = 20, sum_squared_dev = 50, alpha = 3, beta = 2)
    distributions.posterior_pareto(n = 20, max_x=10, xm=2, alpha=2)
