"""E1. Exact moments of a probabilistic TT vs Monte Carlo, for different element distributions with the SAME
mean/std.  Grid-TT at one multi-index == product of d random matrices with independent elements.

Checks
 (a) E f, Var f (exact recursion) vs MC for Normal / Uniform / Student-t5 / Laplace / two-point  -> z-scores
 (b) shape of f: skewness, excess kurtosis, Gaussian 95% interval coverage, error of Gaussian 5%-quantile,
     Cantelli (distribution-free) bound  -- as functions of d and coefficient of variation (cv)
 (c) numerical stability: naive E[f^2]-E[f]^2 vs stable recursion (float32/float64) for small cv / large d
"""
import numpy as np, json, time, sys
from scipy.stats import skew, kurtosis
sys.path.insert(0, '.')
from pttbo.tt import PTT, OneHotBasis
from pttbo.plot import plt, PAL

OUT = 'results/e1_moments.json'
rng = np.random.default_rng(0)


def draw(dist, mu, sd, size, rng):
    if dist == 'normal':
        z = rng.standard_normal(size)
    elif dist == 'uniform':
        z = rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    elif dist == 't5':
        z = rng.standard_t(5, size) / np.sqrt(5 / 3)
    elif dist == 'laplace':
        z = rng.laplace(0, 1 / np.sqrt(2), size)
    elif dist == 'twopoint':
        z = rng.choice([-1.0, 1.0], size)
    return mu + sd * z


def make_means(d, r, rng):
    """Mean cores: identity + random perturbation (keeps E f = O(1) for all d)."""
    rr = [1] + [r] * (d - 1) + [1]
    Ms = []
    for k in range(d):
        M = 0.3 * rng.standard_normal((rr[k], rr[k + 1])) / np.sqrt(r)
        M += np.eye(rr[k], rr[k + 1])
        Ms.append(M)
    return Ms


def exact(Ms, Ss, linearized=False):
    # grid TT with 1 grid point: basis one-hot of size 1
    B = [OneHotBasis(1)] * len(Ms)
    ptt = PTT([M[:, None, :] for M in Ms], [S.reshape(-1) ** 2 for S in Ss], B)
    m, v = ptt.moments(np.zeros((1, len(Ms)), int), linearized=linearized)
    return m[0], v[0]


def mc(Ms, Ss, dist, S, rng, batch=50000):
    out = []
    for s in range(0, S, batch):
        n = min(batch, S - s)
        l = np.ones((n, 1))
        for M, Sd in zip(Ms, Ss):
            G = draw(dist, M[None], Sd[None], (n,) + M.shape, rng)
            l = np.einsum('na,nab->nb', l, G)
        out.append(l[:, 0])
    return np.concatenate(out)


res = {'a': [], 'c': []}
dists = ['normal', 'uniform', 't5', 'laplace', 'twopoint']
t0 = time.time()
S = 200000
for d in [2, 5, 10, 20, 40]:
    for r in [2, 4]:
        for cv in [0.1, 0.3, 0.6]:
            Ms = make_means(d, r, rng)
            Ss = [cv * np.sqrt(np.mean(M ** 2)) * np.ones_like(M) for M in Ms]
            m_ex, v_ex = exact(Ms, Ss)
            _, v_lin = exact(Ms, Ss, linearized=True)
            for dist in dists:
                f = mc(Ms, Ss, dist, S, rng)
                fm, fv = f.mean(), f.var()
                se_m = np.sqrt(fv / S)
                # SE of variance estimate: sqrt((mu4 - var^2)/S)
                mu4 = np.mean((f - fm) ** 4)
                se_v = np.sqrt(max(mu4 - fv ** 2, 1e-300) / S)
                sd = np.sqrt(v_ex)
                cover = np.mean(np.abs(f - m_ex) <= 1.96 * sd)
                q05 = np.quantile(f, 0.05)
                q05_gauss = m_ex - 1.645 * sd
                lower_cantelli = m_ex - np.sqrt(19) * sd  # P(f < mu - k sd) <= 1/(1+k^2) = 0.05
                res['a'].append(dict(d=d, r=r, cv=cv, dist=dist, mean_exact=m_ex, mean_mc=fm, z_mean=(fm - m_ex) / se_m,
                                     var_exact=v_ex, var_mc=fv, z_var=(fv - v_ex) / se_v, var_lin=v_lin,
                                     skew=float(skew(f)), exkurt=float(kurtosis(f)), cover95_gauss=cover,
                                     q05_true=q05, q05_gauss=q05_gauss, q05_err_rel=(q05_gauss - q05) / sd,
                                     p_below_cantelli=float(np.mean(f < lower_cantelli)),
                                     p_below_gauss05=float(np.mean(f < q05_gauss))))
    print('d', d, 'done', round(time.time() - t0, 1), flush=True)

# (c) numerical stability
for d in [10, 50, 100, 200]:
    for cv in [1e-2, 1e-3, 1e-4]:
        r = 3
        Ms = make_means(d, r, np.random.default_rng(d))
        Ss = [cv * np.sqrt(np.mean(M ** 2)) * np.ones_like(M) for M in Ms]
        # reference: stable recursion in longdouble via exact second moments
        B = [OneHotBasis(1)] * d
        res_row = dict(d=d, cv=cv)
        for dt in [np.float32, np.float64, np.longdouble]:
            # naive: E[f^2] via Kronecker transfer, minus (E f)^2
            m = np.ones((1,), dtype=dt); Q = np.ones((1,), dtype=dt)
            p = np.ones((1, 1), dtype=dt); Sst = np.zeros((1, 1), dtype=dt)
            for M, Sd in zip(Ms, Ss):
                M = M.astype(dt); V = (Sd ** 2).astype(dt)
                T = np.kron(M, M) + np.diag(V.reshape(-1)).reshape(M.shape + M.shape).transpose(0, 2, 1, 3).reshape(M.shape[0] ** 2, M.shape[1] ** 2)
                m = m @ M; Q = Q @ T
                # stable
                Qf = Sst + np.outer(p, p)
                Snew = M.T @ Sst @ M
                Snew += np.diag(np.einsum('aa,ab->b', Qf, V))
                Sst = Snew; p = p @ M
            res_row[f'naive_{np.dtype(dt).name}'] = float(Q[0] - m[0] ** 2)
            res_row[f'stable_{np.dtype(dt).name}'] = float(Sst[0, 0])
        res['c'].append(res_row)
json.dump(res, open(OUT, 'w'), indent=1, default=float)

# ---------------------------------------------------------------- figures
import pandas as pd
A = pd.DataFrame(res['a'])
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for i, dist in enumerate(dists):
    sub = A[A.dist == dist]
    ax[0].scatter(sub.z_mean, sub.z_var, s=12, color=PAL[i], label=dist, alpha=0.8)
ax[0].axhline(0, color='#999', lw=0.8); ax[0].axvline(0, color='#999', lw=0.8)
ax[0].set_xlabel('z-score of mean (MC vs exact)'); ax[0].set_ylabel('z-score of variance')
ax[0].set_title('(a) exact moments = MC for all distributions'); ax[0].legend(fontsize=8)
for i, dist in enumerate(dists):
    sub = A[(A.dist == dist) & (A.r == 4) & (A.cv == 0.3)].sort_values('d')
    ax[1].plot(sub.d, sub.exkurt, marker='o', color=PAL[i], label=dist)
    sub = A[(A.dist == dist) & (A.r == 4) & (A.cv == 0.6)].sort_values('d')
    ax[2].plot(sub.d, sub.cover95_gauss, marker='o', color=PAL[i], label=dist)
ax[1].set_xscale('log'); ax[1].set_xlabel('d'); ax[1].set_ylabel('excess kurtosis of f'); ax[1].set_title('(b) tails, r=4, cv=0.3')
ax[2].axhline(0.95, color='#999', lw=0.8, ls='--'); ax[2].set_xscale('log'); ax[2].set_xlabel('d')
ax[2].set_ylabel('coverage of mean +- 1.96 sd'); ax[2].set_title('(c) Gaussian interval, r=4, cv=0.6')
for j, cv in enumerate([0.1, 0.3, 0.6]):
    sub = A[(A.dist == 'normal') & (A.r == 4) & (A.cv == cv)].sort_values('d')
    ax[3].plot(sub.d, sub.var_lin / sub.var_exact, marker='o', color=PAL[j], label=f'cv={cv}')
ax[3].set_xscale('log'); ax[3].set_xlabel('d'); ax[3].set_ylabel('Var_linearised / Var_exact')
ax[3].set_title('(d) delta-method underestimates variance'); ax[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/e1_moments.png'); print('saved')
