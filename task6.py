import numpy as np
import scipy.stats as st

np.random.seed(123)

N          = 100
CONFIDENCE = 0.95
TH_TRUE    = 5
B_NP       = 1000
B_P        = 50000


def inv_pareto(u, th):
    return (1 - u) ** (1 / (1 - th))

def pareto_median(th):
    return 2 ** (1 / (th - 1))

def mle_theta(sample):
    return 1 + 1 / np.mean(np.log(sample))


sample = inv_pareto(st.uniform(0, 1).rvs(size=N), th=TH_TRUE)

th_hat  = mle_theta(sample)
z_lo    = st.norm.ppf((1 - CONFIDENCE) / 2)
z_hi    = st.norm.ppf((1 + CONFIDENCE) / 2)


def asymptotic_theta(th_hat, sample, z_lo, z_hi):
    scale  = np.sum(np.log(sample)) / np.sqrt(len(sample))
    left   = th_hat - z_hi / scale
    right  = th_hat - z_lo / scale
    return left, right, right - left

def asymptotic_median(th_hat, sample, z_lo, z_hi):
    m_hat  = pareto_median(th_hat)
    factor = m_hat * np.log(2) / (np.sqrt(len(sample)) * (th_hat - 1))
    left   = m_hat - factor * z_hi
    right  = m_hat - factor * z_lo
    return left, right, right - left

def bootstrap_np(sample, th_hat, B, conf):
    n      = len(sample)
    deltas = sorted([mle_theta(np.random.choice(sample, size=n, replace=True)) - th_hat
                     for _ in range(B)])
    idx_lo = int((1 - conf) / 2 * B - 1)
    idx_hi = int((1 + conf) / 2 * B - 1)
    left   = th_hat - deltas[idx_hi]
    right  = th_hat - deltas[idx_lo]
    return left, right, right - left

def bootstrap_p(th_hat, n, B, conf):
    deltas = sorted([mle_theta(inv_pareto(st.uniform(0, 1).rvs(size=n), th=th_hat)) - th_hat
                     for _ in range(B)])
    idx_lo = int((1 - conf) / 2 * B - 1)
    idx_hi = int((1 + conf) / 2 * B - 1)
    left   = th_hat - deltas[idx_hi]
    right  = th_hat - deltas[idx_lo]
    return left, right, right - left


asym_th_l,  asym_th_r,  asym_th_len  = asymptotic_theta(th_hat, sample, z_lo, z_hi)
asym_med_l, asym_med_r, asym_med_len = asymptotic_median(th_hat, sample, z_lo, z_hi)
np_l,       np_r,       np_len       = bootstrap_np(sample, th_hat, B_NP, CONFIDENCE)
p_l,        p_r,        p_len        = bootstrap_p(th_hat, N, B_P, CONFIDENCE)

print(f"Истинная медиана: {pareto_median(TH_TRUE):.6f},  оценка: {pareto_median(th_hat):.6f}")
print(f"Асимптотический ДИ для медианы: {asym_med_l:.6f} < m < {asym_med_r:.6f}  (len = {asym_med_len:.6f})\n")

print(f"Истинная theta: {TH_TRUE},  MLE theta: {th_hat:.6f}")

results = [
    ("Асимптотический",        asym_th_l, asym_th_r, asym_th_len),
    ("Бутстрап непараметрич.", np_l,      np_r,      np_len),
    ("Бутстрап параметрич.",   p_l,       p_r,       p_len),
]

print(f"\n{'Метод':<26} {'Левая':>12} {'Правая':>12} {'Длина':>10}")
print("-" * 64)
for name, l, r, length in results:
    print(f"{name:<26} {l:>12.6f} {r:>12.6f} {length:>10.6f}")

print("\nРейтинг по длине интервала (theta):")
for rank, (name, _, _, length) in enumerate(sorted(results, key=lambda x: x[3]), 1):
    print(f"  {rank}) {name} (len = {round(length, 3)})")