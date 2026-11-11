"""E8. Discrete / categorical BO: grid (one-hot) PTT vs Hamming-kernel GP vs PROTES / SA / random.
Problems:
  lowrank_cat20 : d=20, 4 categories, f = random TT of rank 3 (model-matched case)
  bqp20         : binary quadratic x^T Q x, Q_ij ~ N(0,1) exp(-(i-j)^2/L^2), L=2 (banded => low TT rank)
  bqp20_perm    : same with randomly permuted variables (TT rank grows, GP-Hamming invariant)
  contam25      : contamination-control-like chain problem (BOCS-style re-implementation), d=25 binary
usage: python experiments/e8_discrete.py PART NPARTS
"""
import numpy as np, json, sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pttbo.bo import run_bo_discrete, run_protes
from pttbo.tt import PTT, OneHotBasis

part, nparts = int(sys.argv[1]), int(sys.argv[2])
os.makedirs('results/e8', exist_ok=True)


def make_lowrank(d=20, n=4, r=3, seed=0):
    rng = np.random.default_rng(900 + seed)
    rr = [1] + [r] * (d - 1) + [1]
    cores = []
    for k in range(d):
        W = 0.5 * rng.standard_normal((rr[k], n, rr[k + 1])) / np.sqrt(r)
        W += np.einsum('ab,j->ajb', np.eye(rr[k], rr[k + 1]), np.ones(n))
        cores.append(W)
    B = [OneHotBasis(n)] * d
    f = lambda X: PTT.eval_cores(cores, [B[k](np.asarray(X)[:, k]) for k in range(d)])
    return f, cores


def make_bqp(d=20, L=2.0, seed=0, permute=False):
    rng = np.random.default_rng(700 + seed)
    i = np.arange(d)
    Q = rng.standard_normal((d, d)) * np.exp(-(i[:, None] - i[None]) ** 2 / L ** 2)
    perm = rng.permutation(d) if permute else i
    f = lambda X: np.einsum('ni,ij,nj->n', np.asarray(X, float)[:, perm], Q, np.asarray(X, float)[:, perm])
    return f, Q, perm


def make_contam(d=25, T=100, seed=0, lam=1e-2):
    rng = np.random.default_rng(800 + seed)
    Z0 = rng.beta(1, 30, T)
    La = rng.beta(1, 17 / 3, (d, T))
    Ga = rng.beta(11 / 3, 1, (d, T))
    U, eps, rho, cost = 0.1, 0.05, 1.0, np.ones(d)

    def f(X):
        X = np.asarray(X, float)
        N = len(X)
        Z = np.repeat(Z0[None], N, 0)
        pen = np.zeros(N)
        for k in range(d):
            x = X[:, k:k + 1]
            Z = La[k][None] * (1 - x) * (1 - Z) + (1 - Ga[k][None] * x) * Z
            pen += rho * (np.mean(Z >= U, axis=1) - eps)
        return X @ cost + pen + lam * X.sum(1)
    return f


def exact_min_binary(f, d, chunk=1 << 18):
    best = np.inf
    for s in range(0, 1 << d, chunk):
        idx = np.arange(s, min(s + chunk, 1 << d))
        X = ((idx[:, None] >> np.arange(d)[None]) & 1)
        best = min(best, f(X).min())
    return float(best)


PROBS = []
for seed in range(4):
    f, cores = make_lowrank(seed=seed); PROBS.append(('lowrank_cat20', seed, f, 20, 4))
    f, _, _ = make_bqp(seed=seed); PROBS.append(('bqp20', seed, f, 20, 2))
    f, _, _ = make_bqp(seed=seed, permute=True); PROBS.append(('bqp20_perm', seed, f, 20, 2))
    f = make_contam(seed=seed); PROBS.append(('contam25', seed, f, 25, 2))
METHODS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['random', 'sa', 'protes', 'hgp_ts', 'hgp_ei', 'tt_mean', 'tt_ts', 'tt_rmts']
BUDGET, NINIT = 200, 20
jobs = [(p, m) for p in PROBS for m in METHODS]
jobs = jobs[part::nparts]
t0 = time.time()
for (name, seed, f, d, n), meth in jobs:
    fn = f'results/e8/{name}__{meth}__{seed}.json'
    if os.path.exists(fn):
        continue
    t = time.time()
    r = run_bo_discrete(f, d, n, meth, BUDGET, NINIT, seed)
    json.dump(dict(problem=name, method=meth, seed=seed, y=np.asarray(r['y']).tolist(), t=list(map(float, r['t'])), wall=time.time() - t),
              open(fn, 'w'))
    print(f'{name} {meth} {seed} best={np.min(r["y"]):.4g} wall={time.time() - t:.0f}s total={time.time() - t0:.0f}s', flush=True)
# long-budget model-free references (PROTES / SA with 5000 evaluations) + exact optima for BQP
if part == 0:
    for (name, seed, f, d, n) in PROBS:
        for meth in ['protes', 'sa']:
            fn = f'results/e8/{name}__{meth}5000__{seed}.json'
            if os.path.exists(fn):
                continue
            r = run_bo_discrete(f, d, n, meth, 5000, NINIT, seed)
            json.dump(dict(problem=name, method=meth + '5000', seed=seed, y=np.asarray(r['y']).tolist()), open(fn, 'w'))
        if name.startswith('bqp'):
            fn = f'results/e8/{name}__exactmin__{seed}.json'
            if not os.path.exists(fn):
                json.dump(dict(problem=name, seed=seed, fmin=exact_min_binary(f, d)), open(fn, 'w'))
        print('ref', name, seed, flush=True)
print('finished')
