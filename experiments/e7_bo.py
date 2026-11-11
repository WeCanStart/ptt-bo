"""E7. Continuous BO: PTT surrogates vs GP vs model-free baselines.  Resumable; each run -> results/e7/*.json
usage: python experiments/e7_bo.py PART NPARTS [methods,comma,separated]
"""
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pttbo.bo import run_bo
from pttbo.functions import get_problem, smooth_tt_problem

part, nparts = int(sys.argv[1]), int(sys.argv[2])
METHODS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['random', 'cmaes', 'gp_ei', 'gp_ts', 'tt_mean', 'tt_ts', 'tt_rmts', 'tt_eilf']
PROBS = [  # name, factory, n_init, budget, seeds
    ('branin', lambda: get_problem('branin'), 10, 50, 6),
    ('hartmann6', lambda: get_problem('hartmann6'), 20, 80, 4),
    ('rosenbrock10', lambda: get_problem('rosenbrock', 10), 20, 100, 4),
    ('styblinski10', lambda: get_problem('styblinski', 10), 20, 100, 4),
    ('styblinski10_rot', lambda: get_problem('styblinski', 10, rotate=True), 20, 100, 4),
    ('smoothTT20_r3', lambda: smooth_tt_problem(20, 3), 20, 100, 4),
]
os.makedirs('results/e7', exist_ok=True)
jobs = [(p, m, s) for p in PROBS for s in range(p[4]) for m in METHODS]
# cost-balanced split: expensive TT jobs interleaved
jobs = jobs[part::nparts]
t0 = time.time()
for (name, fac, n_init, budget, _), meth, seed in jobs:
    fn = f'results/e7/{name}__{meth}__{seed}.json'
    if os.path.exists(fn):
        continue
    prob = fac()
    t = time.time()
    r = run_bo(prob, meth, budget, n_init, seed)
    json.dump(dict(problem=name, method=meth, seed=seed, y=r['y'].tolist(), t=list(map(float, r['t'])),
                   fopt=prob.fopt, wall=time.time() - t), open(fn, 'w'))
    print(f'{name} {meth} {seed} best={np.min(r["y"]):.4g} wall={time.time() - t:.0f}s total={time.time() - t0:.0f}s', flush=True)
print('finished')
