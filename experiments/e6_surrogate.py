"""E6. Offline surrogate quality: PTT (several inference schemes, two priors) vs GP Matern-5/2 ARD.
Random uniform training designs of size n, 2000 uniform test points.
Metrics: normalised RMSE, RMSE on the best 10% test points, Gaussian NLL (with noise), 95% coverage, time.
"""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
from scipy.stats import norm
from pttbo.tt import FourierBasis
from pttbo.infer import PTTRegressor
from pttbo.gp import GPRegressor
from pttbo.functions import get_problem, smooth_tt_problem

part = int(sys.argv[1]) if len(sys.argv) > 1 else 0
nparts = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PROBS = [('branin', lambda: get_problem('branin'), [20, 50, 150]),
         ('hartmann6', lambda: get_problem('hartmann6'), [50, 150, 400]),
         ('hartmann6_rot', lambda: get_problem('hartmann6', rotate=True), [50, 150, 400]),
         ('rosenbrock10', lambda: get_problem('rosenbrock', 10), [50, 150, 400]),
         ('rosenbrock10_perm', lambda: get_problem('rosenbrock', 10, permute=True), [50, 150, 400]),
         ('styblinski10', lambda: get_problem('styblinski', 10), [50, 150, 400]),
         ('styblinski10_rot', lambda: get_problem('styblinski', 10, rotate=True), [50, 150, 400]),
         ('levy10', lambda: get_problem('levy', 10), [50, 150, 400]),
         ('ackley10', lambda: get_problem('ackley', 10), [50, 150, 400]),
         ('smoothTT20_r3', lambda: smooth_tt_problem(20, 3), [50, 150, 400])]
jobs = [(p, n, s) for p in PROBS for n in p[2] for s in range(2)]
jobs = jobs[part::nparts]
out = []
fn = f'results/e6_part{part}.json'
t00 = time.time()
for (pname, pf, _), n, seed in jobs:
    prob = pf()
    rng = np.random.default_rng(7 + seed)
    U = rng.random((n, prob.d)); y = prob(U)
    Ut = rng.random((2000, prob.d)); yt = prob(Ut)
    top = yt <= np.quantile(yt, 0.1)
    sdy = yt.std()

    def score(name, mu, sd, ns, t, extra=None):
        s = np.sqrt(sd ** 2 + ns ** 2) + 1e-12
        out.append(dict(problem=pname, d=prob.d, n=n, seed=seed, method=name, time=t,
                        nrmse=float(np.sqrt(np.mean((mu - yt) ** 2)) / sdy),
                        nrmse_top10=float(np.sqrt(np.mean((mu[top] - yt[top]) ** 2)) / sdy),
                        nll=float(np.mean(-norm.logpdf(yt, mu, s)) + np.log(sdy)),
                        cov95=float(np.mean(np.abs(mu - yt) < 1.96 * s)), **(extra or {})))
    t = time.time(); g = GPRegressor(seed=seed).fit(U, y); mu, sd = g.predict(Ut); score('gp_matern', mu, sd, g.noise_std, time.time() - t)
    nf = 6 if prob.d <= 3 else 5
    for prior in ['anova', 'zero']:
        B = [FourierBasis(nf, 0.25) for _ in range(prob.d)]
        t = time.time(); base = PTTRegressor(B, rank=4, method='map', prior=prior, seed=seed).fit(U, y); tb = time.time() - t
        mu, sd = base.predict(Ut); score(f'tt_{prior}_map', mu, sd, base.noise_std, tb, dict(beta=base.beta_cv))
        meths = ['lafull', 'vb'] if prior == 'zero' else ['lafull', 'lablock', 'ladiag', 'la1', 'vb', 'ens']
        for meth in meths:
            if meth == 'ens' and (n > 150 or prob.d > 10):
                continue
            t = time.time()
            m = PTTRegressor(B, rank=4, method=meth, prior=prior, seed=seed, beta=base.beta_cv, n_sweeps=8).fit(U, y, warm=base)
            mu, sd = m.predict(Ut)
            score(f'tt_{prior}_{meth}', mu, sd, m.noise_std, tb + time.time() - t)
    print(pname, n, seed, 'done', round(time.time() - t00), flush=True)
    json.dump(out, open(fn, 'w'), indent=1)
print('finished')
