"""E6b. Extra surrogate baselines/ablations on the SAME designs as E6 (same seeds):
  - GP Matern-5/2 ARD with dimension-scaled lengthscale priors (BoTorch-style MAP)   -> 'gp_prior'
  - first-order additive GP (sum of 1-D Matern-5/2)                                  -> 'gp_additive'
  - PTT (anova prior, MAP mean) with TT-rank r in {1, 2, 8}  (r=4 is in E6)          -> 'tt_anova_map_r{r}'
"""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
from scipy.stats import norm
from pttbo.tt import FourierBasis
from pttbo.infer import PTTRegressor
from pttbo.gp import ARDGP, AdditiveGP
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
jobs = [(p, n, s) for p in PROBS for n in p[2] for s in range(2)][part::nparts]
out = []
t00 = time.time()
for (pname, pf, _), n, seed in jobs:
    prob = pf()
    rng = np.random.default_rng(7 + seed)
    U = rng.random((n, prob.d)); y = prob(U)
    Ut = rng.random((2000, prob.d)); yt = prob(Ut)
    top = yt <= np.quantile(yt, 0.1); sdy = yt.std()

    def score(name, mu, sd, ns, t):
        s = np.sqrt(sd ** 2 + ns ** 2) + 1e-12
        out.append(dict(problem=pname, d=prob.d, n=n, seed=seed, method=name, time=t,
                        nrmse=float(np.sqrt(np.mean((mu - yt) ** 2)) / sdy),
                        nrmse_top10=float(np.sqrt(np.mean((mu[top] - yt[top]) ** 2)) / sdy),
                        nll=float(np.mean(-norm.logpdf(yt, mu, s)) + np.log(sdy)),
                        cov95=float(np.mean(np.abs(mu - yt) < 1.96 * s))))
    for name, M in [('gp_prior', ARDGP), ('gp_additive', AdditiveGP)]:
        t = time.time(); g = M(seed=seed).fit(U, y); mu, sd = g.predict(Ut); score(name, mu, sd, g.noise_std, time.time() - t)
    nf = 6 if prob.d <= 3 else 5
    for r in [1, 2, 8]:
        if r == 8 and prob.d > 10:
            continue
        B = [FourierBasis(nf, 0.25) for _ in range(prob.d)]
        t = time.time(); m = PTTRegressor(B, rank=r, method='map', prior='anova', seed=seed).fit(U, y)
        mu, sd = m.predict(Ut); score(f'tt_anova_map_r{r}', mu, sd, m.noise_std, time.time() - t)
    print(pname, n, seed, round(time.time() - t00), flush=True)
    json.dump(out, open(f'results/e6b_part{part}.json', 'w'), indent=1)
print('finished')
