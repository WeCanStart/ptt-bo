"""Additional BO studies (resumable; one JSON per run in results/<study>/).

  e9  : confirmation study, 10 seeds. New "interaction" problems (Dixon-Price, Powell, Zakharov) + controls
        (rotations / permutations), BoTorch baselines (LogEI, TuRBO-1), additive GP, CMA-ES.
  e10 : PTT-EI ablations (rank, basis lengthscale, prior) on 3 problems, seeds 0-3.
  e11 : noisy observations (noise sd = 10% of std f), 5 seeds; regret measured on the noiseless f.
        (superseded by e11b: with heavy-tailed f this noise level swamps the region of interest)
  e13 : why does PTT-EI stall after ~40 evaluations?  2x2 design on the E9 problems/seeds:
        {global PTT, trust-region PTT} x {raw y, power-warped y}; the same output warping is given to the GP
        baselines (GP+priors, BoTorch LogEI, TuRBO-1) for fairness. Unwarped baselines are taken from results/e9.
  e11b: noisy observations, redesigned: noise sd = 1% of the median gap f(U) - f* over the cube (sd is 0.1-0.7x the
        typical final regret), 4 problems, 10 seeds, noise-aware variants (plug-in EI for GP/PTT, qLogNEI, TuRBO
        with a noise-inferring likelihood). Stored: true f and noisy y of every evaluation. TR-PTT added after E13.
  e14 : HELD-OUT validation of TR-PTT (protocol fixed before running, see README): 9 problems never used while
        developing TR-PTT (Levy, Ackley, Griewank, Michalewicz, Rastrigin in d=10, budget 150; Rosenbrock, Dixon-Price,
        Levy in d=20, budget 200), 10 seeds; TR-PTT (rank 4, rank 2, +warp) vs TuRBO(+warp), LogEI+warp, global PTT.

usage: python experiments/e9_bo_extra.py STUDY PART NPARTS [methods,comma,separated]
"""
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pttbo.bo import run_bo
from pttbo.functions import get_problem

study, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
P = lambda name, d=None, **kw: (lambda: get_problem(name, d, **kw))

E9_PROBS = [('rosenbrock10', P('rosenbrock', 10), 100), ('rosenbrock10_perm', P('rosenbrock', 10, permute=True), 100),
            ('rosenbrock10_rot', P('rosenbrock', 10, rotate=True), 100),
            ('dixonprice10', P('dixonprice', 10), 100), ('dixonprice10_rot', P('dixonprice', 10, rotate=True), 100),
            ('powell12', P('powell', 12), 100), ('zakharov10', P('zakharov', 10), 100),
            ('zakharov10_rot', P('zakharov', 10, rotate=True), 100),
            ('styblinski10', P('styblinski', 10), 100), ('styblinski10_rot', P('styblinski', 10, rotate=True), 100),
            ('hartmann6', P('hartmann6'), 80)]
NOISE = None
if study == 'e9':
    PROBS = E9_PROBS
    SEEDS = range(10)
    METHODS = ['random', 'cmaes', 'gpp_ei', 'gpa_ei', 'tt_eilf', 'bt_logei', 'bt_turbo']
elif study == 'e10':
    PROBS = [('rosenbrock10', P('rosenbrock', 10), 100), ('zakharov10', P('zakharov', 10), 100),
             ('hartmann6', P('hartmann6'), 80)]
    SEEDS = range(4)
    METHODS = ['tt_eilf', 'tt_eilf_r2', 'tt_eilf_r8', 'tt_eilf_l015', 'tt_eilf_l040', 'tt_eilf_zero']
elif study == 'e11':
    PROBS = [('rosenbrock10', P('rosenbrock', 10), 100), ('zakharov10', P('zakharov', 10), 100),
             ('hartmann6', P('hartmann6'), 80)]
    SEEDS = range(5)
    METHODS = ['random', 'gpp_ei', 'tt_eilf', 'bt_logei']
    NOISE = dict(rel=0.1, mode='std')
elif study == 'e13':
    PROBS = E9_PROBS
    SEEDS = range(10)
    # cheap ones first inside each (problem, seed) block does not matter: one run per process
    METHODS = ['tt_eilf_warp', 'tt_tr', 'tt_tr_warp', 'gpp_ei_warp', 'bt_logei_warp', 'bt_turbo_warp']
elif study == 'e11b':
    PROBS = [('hartmann6', P('hartmann6'), 80), ('rosenbrock10', P('rosenbrock', 10), 100),
             ('styblinski10', P('styblinski', 10), 100), ('powell12', P('powell', 12), 100)]
    SEEDS = range(10)
    METHODS = ['random', 'gpp_ei', 'tt_eilf', 'tt_eilf_warp', 'bt_lognei', 'bt_turbo', 'tt_tr', 'tt_tr_warp']
    NOISE = dict(rel=0.01, mode='median_gap')
elif study == 'e14':
    # d = 20 first (longest jobs first -> better load balancing in run_all)
    PROBS = [('rosenbrock20', P('rosenbrock', 20), 200), ('dixonprice20', P('dixonprice', 20), 200),
             ('levy20', P('levy', 20), 200),
             ('levy10', P('levy', 10), 150), ('levy10_rot', P('levy', 10, rotate=True), 150),
             ('ackley10', P('ackley', 10), 150), ('griewank10', P('griewank', 10), 150),
             ('michalewicz10', P('michalewicz', 10), 150), ('rastrigin10', P('rastrigin', 10), 150)]
    SEEDS = range(10)
    METHODS = ['tt_tr', 'tt_tr_warp', 'tt_eilf', 'tt_tr_r2', 'bt_logei_warp', 'bt_turbo', 'bt_turbo_warp', 'cmaes',
               'random']
else:
    raise ValueError(study)
if len(sys.argv) > 4:
    METHODS = sys.argv[4].split(',')

# method name -> (base method, keyword arguments)
VARIANTS = {'tt_eilf_r2': ('tt_eilf', dict(rank=2)), 'tt_eilf_r8': ('tt_eilf', dict(rank=8)),
            'tt_eilf_l015': ('tt_eilf', dict(ell=0.15)), 'tt_eilf_l040': ('tt_eilf', dict(ell=0.40)),
            'tt_eilf_zero': ('tt_eilf', dict(prior='zero')),
            'tt_eilf_warp': ('tt_eilf', dict(warp=True)), 'tt_tr': ('tt_eilf', dict(tr=True)),
            'tt_tr_warp': ('tt_eilf', dict(tr=True, warp=True)), 'tt_tr_r2': ('tt_eilf', dict(tr=True, rank=2)),
            'gpp_ei_warp': ('gpp_ei', dict(warp=True)),
            'bt_logei_warp': ('bt_logei', dict(warp=True)), 'bt_turbo_warp': ('bt_turbo', dict(warp=True))}


class Noisy:
    """wraps a problem: returns f + N(0, sd^2).
    mode 'std'       : sd = rel * std f(U)            (E11; U uniform on the cube, 4000 points)
    mode 'median_gap': sd = rel * median(f(U) - f*)   (E11b; robust to heavy upper tails)"""
    def __init__(self, prob, seed, rel=0.1, mode='std'):
        self.p, self.d, self.name, self.fopt = prob, prob.d, prob.name, prob.fopt
        v = prob(np.random.default_rng(1).random((4000, prob.d)))
        self.sd = rel * (float(np.std(v)) if mode == 'std' else float(np.median(v - prob.fopt)))
        self.rng = np.random.default_rng(10_000 + seed)

    def __call__(self, U):
        v = self.p(U)
        return v + self.sd * self.rng.standard_normal(len(v))


PTT_RANK_HIGH_D = 2   # E14 protocol: PTT rank 4 for d <= 12, rank 2 for d = 20 (rank 4 costs > 8 s/iteration there)


def run(prob, meth, budget, seed, noisy):
    n_init = 20
    base, kw = VARIANTS.get(meth, (meth, {}))
    if study == 'e14' and base.startswith('tt_') and prob.d > 12 and 'rank' not in kw:
        kw = dict(kw, rank=PTT_RANK_HIGH_D)
    if base.startswith('bt_'):
        from pttbo.bo_botorch import run_botorch
        return run_botorch(prob, base, budget, n_init, seed, noisy=noisy, **kw)
    if noisy and base not in ('random', 'cmaes'):
        kw = dict(kw, noisy=True)
    return run_bo(prob, base, budget, n_init, seed, **kw)


os.makedirs(f'results/{study}', exist_ok=True)
jobs = [(p, m, s) for p in PROBS for s in SEEDS for m in METHODS]
if study == 'e14':  # at d = 20 the rank-2 variant IS the main TR-PTT (see PTT_RANK_HIGH_D): do not run it twice
    jobs = [j for j in jobs if not (j[1] == 'tt_tr_r2' and j[0][0].endswith('20'))]
if part == -1:
    print(len(jobs)); sys.exit()
jobs = jobs[part::nparts]
if any(m.startswith('bt_') for _, m, _ in jobs) and not os.environ.get('TTGP_SKIP_BOTORCH'):
    # Windows + conda MKL: torch must load its OpenMP DLL before numpy's MKL first touches BLAS/LAPACK,
    # otherwise "WinError 127 ... torch\\lib\\shm.dll" (hit all *_rot problems: get_problem calls np.linalg.qr)
    try:
        import torch  # noqa: F401
    except Exception:
        pass
t0 = time.time()
for (name, fac, budget), meth, seed in jobs:
    fn = f'results/{study}/{name}__{meth}__{seed}.json'
    if os.path.exists(fn) or (meth.startswith('bt_') and os.environ.get('TTGP_SKIP_BOTORCH')):
        continue
    prob = fac()
    t = time.time()
    if NOISE is not None:
        noisy = Noisy(prob, seed, **NOISE)
        r = run(noisy, meth, budget, seed, noisy=(study == 'e11b'))
        y_true = prob(np.asarray(r['X']))
        out = dict(problem=name, method=meth, seed=seed, y=y_true.tolist(), y_noisy=np.asarray(r['y']).tolist(),
                   noise_sd=noisy.sd, t=list(map(float, r['t'])), fopt=prob.fopt, wall=time.time() - t)
    else:
        r = run(prob, meth, budget, seed, noisy=False)
        out = dict(problem=name, method=meth, seed=seed, y=np.asarray(r['y']).tolist(), t=list(map(float, r['t'])),
                   fopt=prob.fopt, wall=time.time() - t)
    if 'tr' in r:
        out['tr'] = r['tr']; out['tr_restarts'] = r['tr_restarts']
    json.dump(out, open(fn, 'w'))
    print(f'{study} {name} {meth} {seed} best={min(out["y"]):.4g} wall={time.time() - t:.0f}s total={time.time() - t0:.0f}s',
          flush=True)
print('finished')
