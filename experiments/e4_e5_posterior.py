"""E4: accuracy of approximate posteriors (Laplace variants, VB, ensemble) vs a Gibbs gold standard,
     same model / basis / rank / FIXED hyper-parameters.
E5: effect of the prior family (Normal / Student-t5 / Uniform with identical mean & variance) on the
     posterior predictive (Gibbs for all three).
Small problems so that Gibbs is cheap and reliable: Branin (2D), Hartmann3 (3D), smooth TT (4D, rank 2).
"""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
from scipy.stats import norm
from pttbo.tt import FourierBasis, PTT
from pttbo.infer import PTTRegressor, gibbs, predict_samples
from pttbo.functions import get_problem, smooth_tt_problem

which = sys.argv[1] if len(sys.argv) > 1 else 'all'
E4_METHODS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['la1', 'lablock', 'ladiag', 'lafull', 'vb', 'vbdiag', 'ens', 'lablock_lin']
TAG = sys.argv[3] if len(sys.argv) > 3 else which
RANK, NF = 3, 4
problems = [('branin', get_problem('branin')), ('hartmann3', get_problem('hartmann3')),
            ('smoothTT4', smooth_tt_problem(4, 2, nf=3))]
res4, res5 = [], []
t0 = time.time()
for pname, prob in problems:
    for n in [12, 40]:
        for seed in range(2):
            rng = np.random.default_rng(100 * seed + n)
            U = rng.random((n, prob.d)); y = prob(U)
            Ut = rng.random((400, prob.d)); yt = prob(Ut)
            for prior in ['zero', 'anova']:
                B = [FourierBasis(NF, 0.25) for _ in range(prob.d)]
                base = PTTRegressor(B, rank=RANK, method='map', prior=prior, seed=seed).fit(U, y)
                beta, alpha = base.beta_cv, 1.0
                z = (y - base.ym) / base.ys
                zt = (yt - base.ym) / base.ys
                # ---- gold standard: 4 Gibbs chains, normal prior
                chains = {}
                for pr in (['normal', 't', 'uniform'] if which in ('all', 'e5') else ['normal']):
                    nb, ns = (400, 300) if pr != 'uniform' else (300, 200)
                    S = []
                    for c in range(4 if pr == 'normal' else 2):
                        init = base.ptt.means if c == 0 else base._init_cores(np.random.default_rng(c), scale=1.0)
                        S.append(gibbs(base, U, z, beta, alpha, n_samples=ns, burn=nb, thin=2, prior=pr,
                                       init=[W.copy() for W in init], rng=np.random.default_rng(10 + c)))
                    F = np.stack([predict_samples(base, s, Ut) for s in S])  # chains x samples x points
                    # split-free R-hat on predictions
                    W_ = F.var(1, ddof=1).mean(0); Bv = F.mean(1).var(0, ddof=1) * F.shape[1]
                    rhat = np.sqrt(((F.shape[1] - 1) / F.shape[1] * W_ + Bv / F.shape[1]) / W_)
                    Fa = F.reshape(-1, F.shape[2])
                    chains[pr] = dict(mean=Fa.mean(0), sd=Fa.std(0), rhat=float(np.median(rhat)),
                                      rhat90=float(np.quantile(rhat, 0.9)), q05=np.quantile(Fa, 0.05, 0),
                                      q95=np.quantile(Fa, 0.95, 0))
                g = chains['normal']
                sn = 1 / np.sqrt(beta)

                def scores(mu, sd):
                    s = np.sqrt(sd ** 2 + sn ** 2)
                    return dict(rmse_truth=float(np.sqrt(np.mean((mu - zt) ** 2))),
                                cov95_truth=float(np.mean(np.abs(mu - zt) < 1.96 * s)),
                                nll_truth=float(np.mean(-norm.logpdf(zt, mu, s))),
                                mean_dev=float(np.sqrt(np.mean((mu - g['mean']) ** 2)) / np.mean(g['sd'])),
                                log_sd_ratio=float(np.median(np.log(sd / g['sd']))),
                                rms_log_sd_ratio=float(np.sqrt(np.mean(np.log(sd / g['sd']) ** 2))))
                common = dict(problem=pname, n=n, seed=seed, prior=prior, beta=beta, rhat=g['rhat'], rhat90=g['rhat90'])
                if which in ('all', 'e4'):
                    res4.append(dict(method='gibbs', **common, **scores(g['mean'], g['sd'])))
                    for meth in E4_METHODS:
                        mm = 'lablock' if meth == 'lablock_lin' else meth
                        m = PTTRegressor(B, rank=RANK, method=mm, prior=prior, seed=seed, beta=beta, n_sweeps=15).fit(U, y, warm=base)
                        mu, sd = m.predict_std(Ut, linearized=(meth == 'lablock_lin'))
                        res4.append(dict(method=meth, **common, **scores(mu, np.maximum(sd, 1e-9))))
                if which in ('all', 'e5'):
                    for pr in ['normal', 't', 'uniform']:
                        c = chains[pr]
                        res5.append(dict(**common, family=pr, rhat_f=c['rhat'], **scores(c['mean'], c['sd']),
                                         q_width_ratio=float(np.median((c['q95'] - c['q05']) / (g['q95'] - g['q05'])))))
                print(pname, n, seed, prior, 'rhat', round(g['rhat'], 3), round(time.time() - t0), flush=True)
                json.dump(dict(e4=res4, e5=res5), open(f'results/e4_e5_{TAG}.json', 'w'), indent=1)
print('done')
