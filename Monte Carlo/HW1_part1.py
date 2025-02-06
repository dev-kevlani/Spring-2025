import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#                                       Problem 1 - Part 1
import numpy as np
import matplotlib.pyplot as plt

# Given parameters
r = 0.05
sigma = 0.4
s = 100
mu_values = [-0.2, 0, 0.2, r - sigma**2 / 2]
N_max = 10000
N_values = np.arange(1, N_max + 1, 1)

# Function to compute p_N
def compute_p_N(N, mu):
    u_N = np.exp(mu / N + sigma * np.sqrt(1 / N))
    d_N = 1 / u_N
    return (np.exp(r / N) - d_N) / (u_N - d_N)

results = {}
for mu in mu_values:
    p_N_values = np.array([compute_p_N(N, mu) for N in N_values])
    pN_1_minus_pN = p_N_values * (1 - p_N_values)
    results[mu] = (p_N_values, pN_1_minus_pN)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, mu in zip(axes.flatten(), mu_values):
    p_N_values, pN_1_minus_pN = results[mu]
    ax.plot(N_values, p_N_values, label=r'$p_N$', color='b')
    ax.plot(N_values, pN_1_minus_pN, label=r'$p_N(1 - p_N)$', color='r')
    ax.set_title(fr'$\mu = {mu}$')
    ax.set_xlabel('N')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# Compute numerical limits for large N
numerical_limits = {mu: {"lim p_N": results[mu][0][-1], "lim p_N(1 - p_N)": results[mu][1][-1]} for mu in mu_values}
print(numerical_limits)

                                    #Problem 1 - Part 2
def estimating_convergence(N_values, p_N_values, pN_1_minus_pN_values):
    """Estimation of k1 and k2 using log-log regression."""

    p_lim = round(p_N_values[-1], 2)
    pN_1_minus_pN_lim = round(pN_1_minus_pN_values[-1], 2)

    # Computing deviations
    # Computing deviations
    deviation_pN = np.abs(p_N_values - p_lim)
    deviation_pN_1_minus_pN = np.abs(pN_1_minus_pN - pN_1_minus_pN_lim)

    # Removing zeros to avoid log issues
    mask = (deviation_pN > 0) & (deviation_pN_1_minus_pN > 0)
    N_filtered = N_values[mask]
    deviation_pN = deviation_pN[mask]
    deviation_pN_1_minus_pN = deviation_pN_1_minus_pN[mask]

    log_1_over_N = np.log(1 / N_filtered)
    log_deviation_pN = np.log(deviation_pN)
    log_deviation_pN_1_minus_pN = np.log(deviation_pN_1_minus_pN)

    # Linear regression to estimate k1 and k2
    k1, c1_log = np.polyfit(log_1_over_N, log_deviation_pN, 1)
    k2, c2_log = np.polyfit(log_1_over_N, log_deviation_pN_1_minus_pN, 1)

    # Conversion of c1, c2 from log scale
    c1 = np.exp(c1_log)
    c2 = np.exp(c2_log)

    return k1, c1, k2, c2, log_1_over_N, log_deviation_pN, log_deviation_pN_1_minus_pN

p_N_values = np.array([compute_p_N(N, 0) for N in N_values])
pN_1_minus_pN = p_N_values * (1 - p_N_values)
p_lim = 0.5
pN_1_minus_pN_lim = 0.25

# Estimate k1, k2, and plot log-log graphs
k1, c1, k2, c2, log_1_over_N, log_deviation_pN, log_deviation_pN_1_minus_pN = estimating_convergence(
    N_values, p_N_values, pN_1_minus_pN)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(log_1_over_N, log_deviation_pN, 'bo', label=r'$\log |p_N - p_\infty|$')
plt.xlabel(r'$\log(1/N)$')
plt.ylabel(r'$\log |p_N - p_\infty|$')
plt.title(f'Log-Log Plot for $p_N$ (Estimated $k_1$ = {k1:.2f})')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(log_1_over_N, log_deviation_pN_1_minus_pN, 'ro', label=r'$\log |p_N(1 - p_N) - (p_\infty(1 - p_\infty))|$')
plt.xlabel(r'$\log(1/N)$')
plt.ylabel(r'$\log |p_N(1 - p_N) - (p_\infty(1 - p_\infty))|$')
plt.title(f'Log-Log Plot for $p_N(1 - p_N)$ (Estimated $k_2$ = {k2:.2f})')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("Estimated k1 = ", k1 , "\n Estimated k2 = ", k2)

#                                   Problem 1 - Part 3
N = 100000 #taking small N because of computing issues
samples = 1000000

r = 0.05
theo_sigma = 0.4
theo_mu = r - (sigma**2)/2

S0_N = 100

un = np.exp(mu/N + sigma/np.sqrt(N))
dn = 1 / un
pn = (np.exp(r/N) - dn)/(un - dn)
def ln_SN_over_S0(N, samples):

    up_moves = np.random.binomial(N, pn, size=samples)
    log_SN_over_S0 = up_moves*np.log(un) + (N-up_moves)*np.log(dn)

    return log_SN_over_S0

X_N_samples = ln_SN_over_S0(N, samples)
mean_XN = np.mean(X_N_samples)
std_XN = np.std(X_N_samples)

print("Theoritical Mean = ", theo_mu)
print("Empirical Mean = ", mean_XN)
print("Theoritical std = ", theo_sigma)
print("Empirical std = ", std_XN)

plt.hist(X_N_samples, bins=100, density=True, alpha=0.6, 
         label=f'Empirical (μ = {mu:.4f})')
x = np.linspace(theo_mu - 3*theo_sigma, theo_mu + 3*theo_sigma, 1000)
plt.plot(x, 1/(theo_sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x - theo_mu)/theo_sigma)**2),
         'r--', label='Theoretical N(-0.03, 0.4²)')
plt.xlabel('ln(S_N/S₀)')
plt.ylabel('Density')
plt.legend()
plt.show()
                                 # Problem 1 - Part 5
N = 10000 #taking small N because of computing issues
samples = 1000000

r = 0.05
theo_sigma = 0.4
mu = [-0.2, 0, 0.2, r - (sigma**2)/2]

S0_N = 100

K = 100
T = 1
def compute_limiting_call_price(N, samples, K, un, dn, pn):

    up_moves = np.random.binomial(N, pn, size=samples)
    log_ST = np.log(S0_N) + up_moves*np.log(un) + (N-up_moves)*np.log(dn)
    S_T = np.exp(log_ST)
    payoff = np.maximum(S_T-K, 0)
    disc_payoff = np.exp(-r*T)*payoff

    return disc_payoff

def compute_bsm_price(K):
    d1 = (np.log(S0_N/K) + (r+ 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S0_N*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

results = {}

for mean in mu:
    results[mean] = {}

    un = np.exp(mean/N + sigma/np.sqrt(N))
    dn = 1 / un
    pn = (np.exp(r/N) - dn)/(un - dn)

    call_payoffs = compute_limiting_call_price(N, samples, K, un, dn, pn)
    call_price = np.mean(call_payoffs)
    std_error = np.std(call_payoffs) / np.sqrt(samples)

    print(f"Empirical Call Price for mu = {mean:.3f} is {call_price:.3f}")
    print(f"Empirical std error for mu = {mean:.3f} is {std_error:.3f}")

    results[mean] = {
        'call_price': call_price,
        'std_error': std_error
    }

bsm_price = compute_bsm_price(K)
print("Black Scholes Price = ", bsm_price)

plt.figure(figsize=(6,4))
plt.plot(np.arange(1, samples+1), np.cumsum(call_payoffs)/np.arange(1, samples+1), color='blue')
plt.axhline(bsm_price, linestyle='--', color='red', label='Black-Scholes Price')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated Call Price')
plt.title('Monte Carlo Convergence (μ = -0.03)')
plt.legend()
plt.show()


# xxxxxx--------------END OF QUESTION 1---------------xxxxxxx