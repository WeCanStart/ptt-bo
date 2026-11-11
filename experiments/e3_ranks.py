"""E3. Numerical TT-ranks of standard benchmark functions on uniform grids, plain vs rotated vs permuted.
Rank = max TT-rank after truncation with relative Frobenius error eps.  d<=6: full tensor + TT-SVD;
d=10,20: TT-cross (teneva) with rank growth, then truncation; validation error reported.
"""
import numpy as np, json, sys, time
sys.path.insert(0, '.')
import teneva
from pttbo.functions import get_problem, smooth_tt_problem

EPS = [1e-2, 1e-4, 1e-6]


def grid_fun(prob, n):
    def f(I):
        U = (np.asarray(I) + 0.5) / n
        return prob(U)
    return f


def ranks_full(prob, n):
    d = prob.d
    I = np.indices([n] * d).reshape(d, -1).T
    Y = grid_fun(prob, n)(I).reshape([n] * d)
    out = {}
    for e in EPS:
        T = teneva.svd(Y, e=e)
        out[e] = int(max(teneva.ranks(T)))
    return out, 0.0


def ranks_cross(prob, n, r_max=40, seed=0, m=int(3e6)):
    d = prob.d
    f = grid_fun(prob, n)
    rng = np.random.default_rng(seed)
    I_vld = rng.integers(0, n, (4000, d)); y_vld = f(I_vld)
    Y = teneva.rand([n] * d, 2, seed=seed)
    info = {}
    Y = teneva.cross(f, Y, m=m, e=1e-10, nswp=50, dr_min=1, dr_max=2, info=info,
                     I_vld=I_vld, y_vld=y_vld, e_vld=1e-9)
    Y = teneva.truncate(Y, 1e-12, r=r_max)
    err = teneva.accuracy_on_data(Y, I_vld, y_vld)
    out = {}
    for e in EPS:
        out[e] = int(max(teneva.ranks(teneva.truncate(Y, e))))
    return out, float(err)


cases = [('branin', None, 64), ('hartmann3', None, 32), ('hartmann6', None, 12)]
for name in ['rosenbrock', 'rastrigin', 'ackley', 'griewank', 'levy', 'styblinski', 'michalewicz']:
    cases.append((name, 6, 12))
for name in ['rosenbrock', 'rastrigin', 'levy', 'styblinski', 'ackley']:
    cases.append((name, 10, 16))
    cases.append((name, 20, 16))
cases += [('dixonprice', 10, 16), ('powell', 12, 16), ('zakharov', 10, 16)]  # functions added for E9
import os
res = json.load(open('results/e3_ranks.json')) if os.path.exists('results/e3_ranks.json') else []
done = {r['problem'] for r in res}
t0 = time.time()
for name, d, n in cases:
    variants = [dict(rotate=False, permute=False), dict(rotate=True, permute=False)]
    if name in ('rosenbrock', 'dixonprice', 'powell', 'zakharov'):
        variants.append(dict(rotate=False, permute=True))
    for v in variants:
        prob = get_problem(name, d, **v)
        if prob.name in done:
            continue
        try:
            if prob.d <= 6:
                rk, err = ranks_full(prob, n)
                how = 'full+svd'
            else:
                rk, err = ranks_cross(prob, n, r_max=30 if prob.d >= 20 else 40, m=int(6e5) if prob.d >= 20 else int(1e6))
                how = 'cross'
        except Exception as ex:
            print('fail', name, d, v, ex); continue
        row = dict(problem=prob.name, d=prob.d, n=n, how=how, cross_err=err, **{f'rank_eps{e:g}': rk[e] for e in EPS})
        res.append(row)
        print(row, round(time.time() - t0, 1), flush=True)
        json.dump(res, open('results/e3_ranks.json', 'w'), indent=1)
for d in [10, 20]:
    for r in [2, 4]:
        prob = smooth_tt_problem(d, r)
        if prob.name in done: continue
        rk, err = ranks_cross(prob, 16, r_max=30, m=int(1e6))
        res.append(dict(problem=prob.name, d=d, n=16, how='cross', cross_err=err, **{f'rank_eps{e:g}': rk[e] for e in EPS}))
        print(res[-1], flush=True)
json.dump(res, open('results/e3_ranks.json', 'w'), indent=1)
