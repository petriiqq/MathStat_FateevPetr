import numpy as np
import scipy.stats as st

np.random.seed(123)

SAMPLE_SIZE = 100
CONFIDENCE  = 0.95
TRUE_THETA  = 22
N_BOOTSTRAP = 1000

sample = st.uniform(loc=TRUE_THETA, scale=TRUE_THETA).rvs(size=SAMPLE_SIZE)

x_mean  = np.mean(sample)
x_max   = np.max(sample)
x_var   = np.sum((sample - x_mean) ** 2) / SAMPLE_SIZE

est_mom = (2 / 3) * x_mean
est_mle = (SAMPLE_SIZE + 1) * x_max / (2 * SAMPLE_SIZE + 1)


def exact_interval(x_max, n, beta):
    g1    = ((1 + beta) / 2) ** (1 / n)
    g2    = ((1 - beta) / 2) ** (1 / n)
    left  = x_max / (1 + g1)
    right = x_max / (1 + g2)
    return left, right, right - left


def asymptotic_interval(est_mom, x_var, n, beta):
    z     = st.norm.ppf((1 + beta) / 2)
    sigma = (2 / 3) * np.sqrt(x_var / n)
    left  = est_mom - z * sigma
    right = est_mom + z * sigma
    return left, right, right - left


def bootstrap_mom_interval(sample, est_mom, B, beta):
    n      = len(sample)
    deltas = sorted([
        (2/3) * np.mean(np.random.choice(sample, size=n, replace=True)) - est_mom
        for _ in range(B)
    ])
    idx_lo = int((1 - beta) / 2 * B - 1)
    idx_hi = int((1 + beta) / 2 * B - 1)
    left   = est_mom - deltas[idx_hi]
    right  = est_mom - deltas[idx_lo]
    return left, right, right - left


def bootstrap_mle_interval(sample, est_mle, B, beta):
    n      = len(sample)
    deltas = sorted([
        (n + 1) * np.max(np.random.choice(sample, size=n, replace=True)) / (2*n + 1) - est_mle
        for _ in range(B)
    ])
    idx_lo = int((1 - beta) / 2 * B)
    idx_hi = int((1 + beta) / 2 * B)
    left   = est_mle - deltas[idx_hi]
    right  = est_mle - deltas[idx_lo]
    return left, right, right - left


exact_l,  exact_r,  exact_len  = exact_interval(x_max, SAMPLE_SIZE, CONFIDENCE)
asym_l,   asym_r,   asym_len   = asymptotic_interval(est_mom, x_var, SAMPLE_SIZE, CONFIDENCE)
bs_mom_l, bs_mom_r, bs_mom_len = bootstrap_mom_interval(sample, est_mom, N_BOOTSTRAP, CONFIDENCE)
bs_mle_l, bs_mle_r, bs_mle_len = bootstrap_mle_interval(sample, est_mle, N_BOOTSTRAP, CONFIDENCE)

results = [
    ("Точный",          exact_l,  exact_r,  exact_len),
    ("Асимптотический", asym_l,   asym_r,   asym_len),
    ("Бутстрап ОММ",    bs_mom_l, bs_mom_r, bs_mom_len),
    ("Бутстрап ОМП",    bs_mle_l, bs_mle_r, bs_mle_len),
]

print(f"{'Метод':<20} {'Левая граница':>15} {'Правая граница':>15} {'Длина':>10}")
print("-" * 65)
for name, l, r, length in results:
    print(f"{name:<20} {l:>15.6f} {r:>15.6f} {length:>10.6f}")

print("\nРейтинг по длине интервала:")
for rank, (name, _, _, length) in enumerate(sorted(results, key=lambda x: x[3]), 1):
    print(f"  {rank}) {name} (l = {round(length, 3)})")