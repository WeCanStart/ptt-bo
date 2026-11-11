"""E2. What prior over functions does a probabilistic TT induce?
 (a) exact prior correlation  corr(f(x0), f(x0 + t*e))  for zero-mean prior (product kernel), identity-mean
     prior ('anova'), their GP limits, and the one-hot grid TT (delta kernel)
 (b) non-Gaussianity of the prior predictive vs rank r and d (MC): GP limit only as r -> inf
 (c) prior samples (2D slices)
"""
import numpy as np, json, sys
sys.path.insert(0, '.')
from scipy.stats import kurtosis
from pttbo.tt import PTT, FourierBasis, OneHotBasis
from pttbo.infer import PTTRegressor
from pttbo.plot import plt, PAL

res = {}
d = 5
B = [FourierBasis(6, 0.25) for _ in range(d)]


def prior_ptt(prior, r, bases):
    m = PTTRegressor(bases, rank=r, prior=prior)
    return PTT([W.copy() for W in m.W0], [v.copy() for v in m.v0], bases), m


x0 = np.full((1, d), 0.5)
ts = np.linspace(-0.5, 0.5, 41)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
res['a'] = {}
for j, (direction, lab) in enumerate([(np.eye(d)[0], 'one coordinate'), (np.ones(d), 'all coordinates')]):
    X = np.clip(x0 + ts[:, None] * direction[None], 0, 1)
    for i, (prior, r) in enumerate([('zero', 1), ('zero', 4), ('zero', 16), ('anova', 4), ('anova', 16)]):
        ptt, _ = prior_ptt(prior, r, B)
        C = ptt.cov_matrix(x0, X)[0]
        v0 = ptt.cov_matrix(x0, x0)[0, 0]
        vx = np.array([ptt.cov_matrix(X[k:k + 1], X[k:k + 1])[0, 0] for k in range(len(X))])
        corr = C / np.sqrt(v0 * vx)
        res['a'][f'{lab}|{prior}|r{r}'] = corr.tolist()
        ax[j].plot(ts, corr, color=PAL[i], label=f'{prior}, r={r}')
    # GP limit of the zero prior: product kernel
    k1 = B[0].kernel1d(np.array([0.5]), np.clip(0.5 + ts, 0, 1))[0] / B[0].kernel1d(np.array([0.5]), np.array([0.5]))[0, 0]
    gp = k1 if j == 0 else k1 ** d
    ax[j].plot(ts, gp, color='k', ls='--', lw=1.2, label='GP limit (product kernel)')
    ax[j].set_title(f'(a) prior corr., shift along {lab}'); ax[j].set_xlabel('t')
ax[0].set_ylabel('corr(f(x0), f(x0+t e))'); ax[0].legend(fontsize=7)

# one-hot grid TT: delta kernel for zero prior, non-trivial for anova
Bg = [OneHotBasis(8) for _ in range(d)]
for prior in ['zero', 'anova']:
    ptt, _ = prior_ptt(prior, 4, Bg)
    xa = np.zeros((1, d), int)
    Xs = np.array([[0] * d, [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    C = ptt.cov_matrix(xa, Xs)[0]
    res['a'][f'grid|{prior}'] = (C / C[0]).tolist()
print('grid-TT prior corr for hamming distance 0,1,2,5:', res['a']['grid|zero'], res['a']['grid|anova'])

# (b) kurtosis of prior predictive vs rank
rng = np.random.default_rng(0)
res['b'] = []
for i, dd in enumerate([2, 5, 10, 20]):
    ks = []
    for r in [1, 2, 4, 8, 16, 32, 64]:
        S = 100000
        rr = [1] + [r] * (dd - 1) + [1]
        vals = []
        bs = max(200, 400000 // (r * r))
        for s in range(0, S, bs):
            n = min(bs, S - s)
            l = np.ones((n, 1))
            for k in range(dd):
                G = rng.standard_normal((n, rr[k], rr[k + 1])) / (rr[k] * rr[k + 1]) ** 0.25
                l = np.einsum('na,nab->nb', l, G)
            vals.append(l[:, 0])
        ku = float(kurtosis(np.concatenate(vals)))
        ks.append(ku)
        res['b'].append(dict(d=dd, r=r, prior='zero', exkurt=ku))
    ax[2].plot([1, 2, 4, 8, 16, 32, 64], ks, marker='o', color=PAL[i], label=f'd={dd}')
ax[2].set_xscale('log', base=2); ax[2].set_yscale('symlog'); ax[2].set_xlabel('rank r')
ax[2].set_ylabel('excess kurtosis of prior f(x)'); ax[2].set_title('(b) zero-mean prior: Gaussian only as r->inf')
ax[2].legend(fontsize=8)
# anova prior kurtosis vs d
ks = []
for dd in [2, 5, 10, 20, 40]:
    m = PTTRegressor([OneHotBasis(1)] * dd, rank=4, prior='anova')
    S = 100000
    l = np.ones((S, 1))
    for k in range(dd):
        W = m.W0[k][:, 0, :]
        G = W[None] + rng.standard_normal((S,) + W.shape) * np.sqrt(m.v0[k]).reshape(W.shape)[None]
        l = np.einsum('na,nab->nb', l, G)
    ks.append(float(kurtosis(l[:, 0])))
    res['b'].append(dict(d=dd, r=4, prior='anova', exkurt=ks[-1], var=float(l[:, 0].var())))
print('anova prior exkurt vs d', ks)

# (c) prior samples, 2D slice
g = np.linspace(0, 1, 60)
G1, G2 = np.meshgrid(g, g)
X = np.full((g.size ** 2, d), 0.5); X[:, 0] = G1.ravel(); X[:, 1] = G2.ravel()
fig2, ax2 = plt.subplots(1, 5, figsize=(16, 3.2))
for i, (prior, r) in enumerate([('zero', 1), ('zero', 4), ('zero', 64), ('anova', 4), ('anova', 64)]):
    m = PTTRegressor(B, rank=r, prior=prior)
    ptt = PTT([W.copy() for W in m.W0], [v.copy() for v in m.v0], B)
    cores = ptt.sample_cores(np.random.default_rng(3 + i), 1)[0]
    f = PTT.eval_cores(cores, ptt.features(X)).reshape(G1.shape)
    ax2[i].imshow(f, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r'); ax2[i].grid(False)
    ax2[i].set_title(f'{prior} prior, r={r}'); ax2[i].set_xlabel('u1'); ax2[i].set_ylabel('u2')
fig2.suptitle('E2(c): prior samples of a 5D PTT (slice u3..u5 = 0.5)', y=1.02)
fig2.savefig('figures/e2_prior_samples.png')
ax[3].axis('off')
ax[3].text(0, 0.95, 'Grid (one-hot) TT, r=4, corr with x=0\nfor Hamming distance 0,1,2,5:', fontsize=9, va='top')
ax[3].text(0, 0.65, 'zero prior : ' + ', '.join(f'{v:.2f}' for v in res['a']['grid|zero']), fontsize=9, va='top')
ax[3].text(0, 0.50, 'anova prior: ' + ', '.join(f'{v:.2f}' for v in res['a']['grid|anova']), fontsize=9, va='top')
ax[3].text(0, 0.30, 'anova prior excess kurtosis vs d=2..40:\n' + ', '.join(f'{v:.2f}' for v in ks), fontsize=9, va='top')
fig.tight_layout(); fig.savefig('figures/e2_prior.png')
json.dump(res, open('results/e2_prior.json', 'w'), indent=1)
print('done')
