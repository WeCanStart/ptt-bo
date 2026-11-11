"""E4b. Same question as E4 (how close are approximate posteriors to a Gibbs gold standard?) at higher
dimension: Hartmann6 (d=6), Rosenbrock8 (d=8), smooth TT 8D (rank 2). Longer chains, R-hat reported.
usage: python experiments/e4b_posterior_hd.py PART NPARTS"""
import numpy as np, json, sys, time, os
sys.path.insert(0, '.')
from scipy.stats import norm
from pttbo.tt import FourierBasis
from pttbo.infer import PTTRegressor, gibbs, predict_samples
from pttbo.functions import get_problem, smooth_tt_problem

part, nparts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
RANK, NF = 3, 3
PROBS = [('hartmann6', lambda: get_problem('hartmann6')), ('rosenbrock8', lambda: get_problem('rosenbrock', 8)),
         ('smoothTT8', lambda: smooth_tt_problem(8, 2, nf=3))]
METHODS = ['la1', 'ladiag', 'lablock', 'lablock_lin', 'lafull', 'vb', 'vbdiag', 'rms']
jobs = [(p, n, s, pr) for p in PROBS for n in [30, 80] for s in range(2) for pr in ['anova', 'zero']][part::nparts]
fn = f'results/e4b_part{part}.json'
res = json.load(open(fn)) if os.path.exists(fn) else []
done = {(r['problem'], r['n'], r['seed'], r['prior']) for r in res}
t0 = time.time()
for (pname, fac), n, seed, prior in jobs:
    if (pname, n, seed, prior) in done:
        continue
    prob = fac()
    rng = np.random.default_rng(300 + 10 * seed + n)
    U = rng.random((n, prob.d)); y = prob(U)
    Ut = rng.random((300, prob.d)); yt = prob(Ut)
    B = [FourierBasis(NF, 0.25) for _ in range(prob.d)]
    base = PTTRegressor(B, rank=RANK, method='map', prior=prior, seed=seed).fit(U, y)
    beta = base.beta_cv
    z = (y - base.ym) / base.ys; zt = (yt - base.ym) / base.ys
    S = []
    for c in range(4):
        init = base.ptt.means if c == 0 else base._init_cores(np.random.default_rng(c), scale=1.0)
        S.append(gibbs(base, U, z, beta, 1.0, n_samples=400, burn=800, thin=2, init=[W.copy() for W in init],
                       rng=np.random.default_rng(50 + c)))
    F = np.stack([predict_samples(base, s, Ut) for s in S])
    Wv = F.var(1, ddof=1).mean(0); Bv = F.mean(1).var(0, ddof=1) * F.shape[1]
    rhat = np.sqrt(((F.shape[1] - 1) / F.shape[1] * Wv + Bv / F.shape[1]) / Wv)
    Fa = F.reshape(-1, F.shape[2]); gm, gs = Fa.mean(0), Fa.std(0)
    sn = 1 / np.sqrt(beta)

    def score(mu, sd):
        s_ = np.sqrt(sd ** 2 + sn ** 2)
        return dict(mean_dev=float(np.sqrt(np.mean((mu - gm) ** 2)) / np.mean(gs)),
                    log_sd_ratio=float(np.median(np.log(sd / gs))),
                    rms_log_sd_ratio=float(np.sqrt(np.mean(np.log(sd / gs) ** 2))),
                    cov95_truth=float(np.mean(np.abs(mu - zt) < 1.96 * s_)))
    common = dict(problem=pname, d=prob.d, n=n, seed=seed, prior=prior, beta=beta,
                  rhat=float(np.median(rhat)), rhat90=float(np.quantile(rhat, 0.9)))
    res.append(dict(method='gibbs', **common, **score(gm, gs)))
    for meth in METHODS:
        mm = 'lablock' if meth == 'lablock_lin' else meth
        m = PTTRegressor(B, rank=RANK, method=mm, prior=prior, seed=seed, beta=beta, n_sweeps=15).fit(U, y, warm=base)
        mu, sd = m.predict_std(Ut, linearized=(meth == 'lablock_lin'))
        res.append(dict(method=meth, **common, **score(mu, np.maximum(sd, 1e-9))))
    json.dump(res, open(fn, 'w'), indent=1)
    print(pname, n, seed, prior, 'rhat', round(common['rhat'], 3), round(time.time() - t0), flush=True)
print('finished')
