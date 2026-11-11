"""E12. How do fit time and accuracy scale with the number of observations n?  GP (O(n^3)) vs PTT (O(n)).
Rosenbrock10, uniform designs. GP with hyper-parameter optimisation only up to n=3000; GP with fixed
hyper-parameters (one Cholesky) up to n=6000; PTT MAP with fixed beta (no CV, 500 L-BFGS iterations) and
PTT + joint Laplace up to n=20000."""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
from pttbo.tt import FourierBasis
from pttbo.infer import PTTRegressor
from pttbo.gp import GPRegressor, ARDGP
from pttbo.functions import get_problem
from pttbo.plot import plt, PAL

prob = get_problem('rosenbrock', 10)
rng = np.random.default_rng(0)
Ut = rng.random((2000, 10)); yt = prob(Ut); sd = yt.std()
out = []


def rec(method, n, t, mu):
    out.append(dict(method=method, n=n, time=t, nrmse=float(np.sqrt(np.mean((mu - yt) ** 2)) / sd)))
    print(method, n, f'{t:.1f}s', f'{out[-1]["nrmse"]:.3f}', flush=True)
    json.dump(out, open('results/e12_scaling.json', 'w'), indent=1)


for n in [100, 300, 1000, 3000, 6000, 20000]:
    U = np.random.default_rng(n).random((n, 10)); y = prob(U)
    if n <= 3000:
        t = time.time(); g = GPRegressor(n_restarts=0).fit(U, y); mu, _ = g.predict(Ut); rec('gp_sklearn_mle', n, time.time() - t, mu)
        t = time.time(); g2 = ARDGP(n_restarts=1).fit(U, y); mu, _ = g2.predict(Ut); rec('gp_prior_map', n, time.time() - t, mu)
        p_fixed = g2.p
    if n <= 6000:
        g3 = ARDGP(); g3.p = p_fixed
        t = time.time(); g3.fit(U, y, optimize=False); mu, _ = g3.predict(Ut); rec('gp_fixed_hyper', n, time.time() - t, mu)
    B = [FourierBasis(5, 0.25) for _ in range(10)]
    t = time.time()
    m = PTTRegressor(B, rank=4, method='map', beta=30.0, maxiter=500).fit(U, y); mu, _ = m.predict(Ut)
    rec('ptt_map_fixed_beta', n, time.time() - t, mu)
    t = time.time()
    m2 = PTTRegressor(B, rank=4, method='lafull', beta=30.0, maxiter=500).fit(U, y); mu, _ = m2.predict(Ut)
    rec('ptt_lafull_fixed_beta', n, time.time() - t, mu)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
names = {'gp_sklearn_mle': 'GP, MLE (sklearn)', 'gp_prior_map': 'GP + priors, MAP', 'gp_fixed_hyper': 'GP, fixed hyper-params',
         'ptt_map_fixed_beta': 'PTT MAP (fixed beta)', 'ptt_lafull_fixed_beta': 'PTT + joint Laplace'}
for i, (k, lab) in enumerate(names.items()):
    rows = [r for r in out if r['method'] == k]
    if not rows:
        continue
    ax[0].plot([r['n'] for r in rows], [r['time'] for r in rows], marker='o', color=PAL[i], label=lab)
    ax[1].plot([r['n'] for r in rows], [r['nrmse'] for r in rows], marker='o', color=PAL[i], label=lab)
for a in ax:
    a.set_xscale('log'); a.set_xlabel('n (observations)')
ax[0].set_yscale('log'); ax[0].set_ylabel('fit + predict time, s (1 thread)'); ax[0].legend(fontsize=8)
ax[1].set_ylabel('normalised RMSE'); ax[1].set_title('Rosenbrock10')
plt.tight_layout(); plt.savefig('figures/e12_scaling.png'); print('saved')
