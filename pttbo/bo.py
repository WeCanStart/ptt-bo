"""Bayesian optimisation loops (minimisation) on the unit cube or on discrete grids.

Continuous methods: random, cmaes, gp_ei, gp_ts, tt_ts (Laplace-full TS), tt_vbts, tt_lcb (VB moments),
                    tt_mean (greedy on MAP mean, no uncertainty), tt_ts_nocd (TS without TT-native polish)
Discrete methods:   random, sa, protes, hgp_ts, hgp_ei (Hamming GP), tt_ts, tt_mean
All surrogate methods share the same candidate generator / acquisition optimiser.
"""
import numpy as np, time
from scipy.stats import norm, qmc
from .tt import FourierBasis, OneHotBasis, PTT
from .infer import PTTRegressor
from .gp import GPRegressor, HammingGP, ARDGP, AdditiveGP


# ------------------------------------------------------------------ candidate generation (continuous)
def cont_candidates(X, y, rng, n_global=1000, n_local=1000):
    d = X.shape[1]
    G = rng.random((n_global, d))
    order = np.argsort(y)
    top = X[order[:5]]
    w = np.array([0.4, 0.2, 0.15, 0.15, 0.1])[:len(top)]; w /= w.sum()
    idx = rng.choice(len(top), n_local, p=w)
    scales = rng.choice([0.01, 0.03, 0.1, 0.3], n_local)
    p = min(1.0, 10.0 / d)
    mask = rng.random((n_local, d)) < p
    mask[np.arange(n_local), rng.integers(0, d, n_local)] = True
    L = top[idx] + mask * rng.standard_normal((n_local, d)) * scales[:, None]
    return np.vstack([G, np.clip(L, 0, 1)])


def refine(acq, C, vals, rng, rounds=2, n_each=100, lb=0.0, ub=1.0, scale=1.0):
    """local refinement of the best candidates under a (cheap, deterministic) acquisition to MINIMISE.
    lb/ub/scale: box and step scale (trust-region variants); defaults reproduce the original behaviour."""
    d = C.shape[1]
    for r in range(rounds):
        best = C[np.argsort(vals)[:5]]
        sc = [0.02, 0.005][min(r, 1)] * scale
        L = np.repeat(best, n_each, 0)
        mask = rng.random(L.shape) < min(1.0, 5.0 / d)
        L = np.clip(L + mask * rng.standard_normal(L.shape) * sc, lb, ub)
        lv = acq(L)
        C = np.vstack([C, L]); vals = np.concatenate([vals, lv])
    i = int(np.argmin(vals))
    return C[i], vals[i]


def coord_descent(g, x0, rng, grid=64, passes=2, discrete_n=None):
    """Minimise a cheap callable g by exact 1-D grid line searches (TT-native polish of a TS sample)."""
    x = x0.copy(); d = len(x)
    fx = g(x[None])[0]
    for _ in range(passes):
        improved = False
        for k in rng.permutation(d):
            vals = np.arange(discrete_n) if discrete_n else np.linspace(0, 1, grid)
            Xk = np.repeat(x[None], len(vals), 0); Xk[:, k] = vals
            gv = g(Xk)
            j = int(np.argmin(gv))
            if gv[j] < fx - 1e-12:
                x, fx, improved = Xk[j].copy(), gv[j], True
        if not improved:
            break
    return x, fx


def power_warp(y):
    """Output warping as in HEBO (Cowen-Rivers et al., JAIR 2022): Box-Cox of y/std(y) if y > 0 (Yeo-Johnson if that
    leaves std < 0.5), Yeo-Johnson otherwise; lambda by maximum likelihood, refitted at every iteration. Monotone
    increasing, so the arg-min is unchanged; it compresses heavy upper tails so the model can resolve the low end."""
    from sklearn.preprocessing import power_transform
    import warnings
    y = np.asarray(y, float)
    ys = (y / (y.std() + 1e-12))[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if ys.min() > 0:
                z = power_transform(ys, method='box-cox')
                if z.std() < 0.5:
                    z = power_transform(ys, method='yeo-johnson')
            else:
                z = power_transform(ys, method='yeo-johnson')
            z = z.ravel()
        except Exception:
            z = None
    if z is None or not np.all(np.isfinite(z)) or z.std() < 1e-12:
        z = (y - y.mean()) / (y.std() + 1e-12)
    return z


class TrustRegion:
    """TuRBO-1 trust-region bookkeeping (Eriksson et al., NeurIPS 2019; constants of the BoTorch tutorial, as in
    pttbo.bo_botorch). On collapse the region is reset to its initial size around the incumbent (data are kept)."""
    def __init__(self, d):
        self.L, self.L_min, self.L_max = 0.8, 0.5 ** 7, 1.6
        self.succ = self.fail = 0
        self.succ_tol, self.fail_tol = 10, int(np.ceil(max(4.0, float(d))))
        self.n_restarts = 0

    def update(self, y_new, y_best_before):
        if y_new < y_best_before - 1e-3 * abs(y_best_before):
            self.succ += 1; self.fail = 0
        else:
            self.succ = 0; self.fail += 1
        if self.succ == self.succ_tol:
            self.L = min(2.0 * self.L, self.L_max); self.succ = 0
        elif self.fail == self.fail_tol:
            self.L /= 2.0; self.fail = 0
        if self.L < self.L_min:
            self.L = 0.8; self.n_restarts += 1


def ei_min(mu, sd, best):
    sd = np.maximum(sd, 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


# ------------------------------------------------------------------ CMA-ES (pycma) with a fixed budget
def run_cmaes(f, d, budget, seed):
    import cma
    es = cma.CMAEvolutionStrategy(0.5 * np.ones(d) + 0.0, 0.25, {'bounds': [0, 1], 'seed': seed + 1, 'verbose': -9})
    ys, Xs = [], []
    while len(ys) < budget:
        P = es.ask()
        P = P[:budget - len(ys)]
        v = f(np.array(P))
        ys.extend(v.tolist()); Xs.extend(P)
        if len(P) == es.popsize:
            es.tell(P, v.tolist())
    return np.array(Xs), np.array(ys)


# ------------------------------------------------------------------ PROTES (numpy port, positive TT parametrisation)
def run_protes(f, d, n, budget, seed, k=20, k_top=4, r=5, lr=0.05, k_gd=1, X_init=None):
    """Port of PROTES (Batsheva et al., NeurIPS'23) without JAX: probability tensor P = TT with positive cores,
    sample k points, maximise mean log-likelihood of the k_top best via Adam. Returns X (indices), y."""
    rng = np.random.default_rng(seed)
    cores = [rng.uniform(0.0, 1.0, (1 if i == 0 else r, n, 1 if i == d - 1 else r)) for i in range(d)]
    m1 = [np.zeros_like(c) for c in cores]; m2 = [np.zeros_like(c) for c in cores]; t = 0
    Xs, ys = [], []

    def sample(K):
        # right interfaces
        Z = [None] * (d + 1); Z[d] = np.ones(1)
        for i in range(d - 1, -1, -1):
            z = cores[i].sum(1) @ Z[i + 1]; Z[i] = z / np.linalg.norm(z)
        out = np.zeros((K, d), int)
        Q = np.ones((K, 1))
        for i in range(d):
            G = np.einsum('kr,rjq,q->kj', Q, cores[i], Z[i + 1])
            G = np.abs(G); G /= G.sum(1, keepdims=True)
            u = rng.random((K, 1))
            out[:, i] = np.minimum((G.cumsum(1) < u).sum(1), n - 1)
            Q = np.einsum('kr,krq->kq', Q, cores[i][:, out[:, i], :].transpose(1, 0, 2))
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        return out

    def grad_loglik(I):
        """gradient of mean log P(i) = log prod Y_k[i_k] - log prod sum_j Y_k[j]  (positive cores)."""
        K = len(I)
        grads = [np.zeros_like(c) for c in cores]
        # left/right products for numerator
        L = [np.ones((K, 1))]
        for i in range(d):
            l = np.einsum('kr,krq->kq', L[-1], cores[i][:, I[:, i], :].transpose(1, 0, 2))
            L.append(l)
        R = [None] * (d + 1); R[d] = np.ones((K, 1))
        for i in range(d - 1, -1, -1):
            R[i] = np.einsum('krq,kq->kr', cores[i][:, I[:, i], :].transpose(1, 0, 2), R[i + 1])
        num = L[d][:, 0]
        for i in range(d):
            gi = np.einsum('kr,kq->krq', L[i], R[i + 1]) / num[:, None, None]
            for kk in range(K):
                grads[i][:, I[kk, i], :] += gi[kk] / K
        # normaliser
        S = [c.sum(1) for c in cores]
        Lz = [np.ones(1)]
        for i in range(d):
            Lz.append(Lz[-1] @ S[i])
        Rz = [None] * (d + 1); Rz[d] = np.ones(1)
        for i in range(d - 1, -1, -1):
            Rz[i] = S[i] @ Rz[i + 1]
        Zt = Lz[d][0]
        for i in range(d):
            grads[i] -= (np.outer(Lz[i], Rz[i + 1]) / Zt)[:, None, :]
        return grads

    if X_init is not None:
        Xs.extend(list(X_init)); ys.extend(list(f(np.asarray(X_init))))
    while len(ys) < budget:
        K = min(k, budget - len(ys))
        I = sample(K)
        v = f(I)
        Xs.extend(list(I)); ys.extend(list(v))
        Ia, ya = np.array(Xs[-k:]), np.array(ys[-k:])
        top = Ia[np.argsort(ya)[:k_top]]
        for _ in range(k_gd):
            g = grad_loglik(top)
            t += 1
            for i in range(d):
                gt = g[i] * cores[i]  # gradient w.r.t. log-cores (positive parametrisation)
                m1[i] = 0.9 * m1[i] + 0.1 * gt; m2[i] = 0.999 * m2[i] + 0.001 * gt ** 2
                step = (m1[i] / (1 - 0.9 ** t)) / (np.sqrt(m2[i] / (1 - 0.999 ** t)) + 1e-8)
                cores[i] = cores[i] * np.exp(lr * step)
                cores[i] /= np.abs(cores[i]).max()
    return np.array(Xs), np.array(ys)


# ------------------------------------------------------------------ main continuous loop
def run_bo(prob, method, budget, n_init, seed, rank=4, nf=5, ell=0.25, prior='anova', log=False,
           beta_every=15, tt_kw=None, warp=False, noisy=False, tr=False, n_min_local=20):
    """warp : fit surrogates to power_warp(y) instead of y (EI computed in the warped scale)
    noisy: EI incumbent = min of the posterior mean over evaluated points (plug-in EI) instead of min observed y
    tr   : (PTT methods) TuRBO-style trust region with a LOCAL PTT fitted in TR coordinates (see below)"""
    rng = np.random.default_rng(seed)
    d = prob.d
    f = lambda U: prob(U)
    t_start = time.time()
    if method == 'random':
        X = rng.random((budget, d)); return dict(X=X, y=f(X), t=[0.0] * budget)
    if method == 'cmaes':
        X, y = run_cmaes(f, d, budget, seed); return dict(X=X, y=y, t=[0.0] * budget)
    X = qmc.Sobol(d, scramble=True, seed=seed).random(n_init)
    y = f(X)
    times = [0.0] * n_init
    model = None
    it = 0
    trs = TrustRegion(d) if tr else None
    tr_log = []
    while len(y) < budget:
        t0 = time.time()
        C = cont_candidates(X, y, rng)
        yfit = power_warp(y) if warp else y
        best = yfit.min()
        incumbent = (lambda m_, X_: float(np.min(m_.predict(X_)[0]))) if noisy else (lambda m_, X_: float(yfit.min()))
        if method == 'gpa_ei':  # first-order additive GP + EI
            m = AdditiveGP(seed=seed, n_restarts=1)
            if model is not None:
                m.p_warm = model.p_best
            m.fit(X, yfit)
            model = m
            best = incumbent(m, X)
            acq = lambda Z: -ei_min(*m.predict(Z), best)
            x, _ = refine(acq, C, acq(C), rng)
        elif method.startswith('gpp'):  # GP with dimension-scaled lengthscale priors (BoTorch-style), MAP each iter
            m = ARDGP(seed=seed, n_restarts=1 if model is not None else 3)
            if model is not None:
                m.p = model.p
            m.fit(X, yfit)
            model = m
            best = incumbent(m, X)
            if method == 'gpp_ei':
                acq = lambda Z: -ei_min(*m.predict(Z), best)
                x, _ = refine(acq, C, acq(C), rng)
            else:
                sub = C[rng.permutation(len(C))[:1500]]
                x = sub[int(np.argmin(m.sample_on(sub, rng)[0]))]
        elif method.startswith('gp'):
            m = GPRegressor(seed=seed, n_restarts=1 if it % 5 else 2)
            if model is not None and it % 5:
                m.gp = model.gp  # reuse hyper-parameters (refit each 5 iters)
                m.fit(X, yfit, optimize=False)
            else:
                m.fit(X, yfit)
            model = m
            best = incumbent(m, X)
            if method == 'gp_ei':
                acq = lambda Z: -ei_min(*m.predict(Z), best)
                x, _ = refine(acq, C, acq(C), rng)
            elif method == 'gp_ts':
                sub = C[rng.permutation(len(C))[:1500]]
                S = m.sample_on(sub, rng)[0]
                x = sub[int(np.argmin(S))]
            elif method == 'gp_lcb':
                acq = lambda Z: (lambda mu, sd: mu - 2.0 * sd)(*m.predict(Z))
                x, _ = refine(acq, C, acq(C), rng)
        else:
            kw = dict(rank=rank, prior=prior, seed=seed, maxiter=1500 if model is None else 250)
            kw.update(tt_kw or {})
            tm = {'tt_ts': 'lafull', 'tt_ts_nocd': 'lafull', 'tt_vbts': 'vb', 'tt_lcb': 'vb', 'tt_mean': 'map',
                  'tt_ensts': 'ens', 'tt_ei': 'vb', 'tt_eila': 'lablock', 'tt_rmts': 'map', 'tt_eilf': 'lafull'}[method]
            B = [FourierBasis(nf, ell) for _ in range(d)]
            if tm == 'vb':
                kw['n_sweeps'] = 4
            if tm == 'ens':
                kw['n_ens'] = 5
            m = PTTRegressor(B, method=tm, **kw)
            if tr:
                # Trust-region PTT: TR box of side L around the incumbent (TuRBO-1 rules). The PTT is fitted only to
                # points in the MODEL box (TR box enlarged x2, grown until it holds >= n_min_local points), mapped to
                # [0,1]^d, so the Fourier basis resolution and the output standardisation follow the TR.
                if method != 'tt_eilf':
                    raise ValueError('tr=True is implemented for tt_eilf only')
                c = X[int(np.argmin(y))]
                tlb, tub = np.clip(c - trs.L / 2, 0, 1), np.clip(c + trs.L / 2, 0, 1)
                s = trs.L
                while True:
                    mlb, mub = np.clip(c - s, 0, 1), np.clip(c + s, 0, 1)
                    inside = np.all((X >= mlb) & (X <= mub), 1)
                    if inside.sum() >= min(n_min_local, len(y)) or s >= 1.0:
                        break
                    s *= 1.25
                loc = lambda Z, a=mlb, b=mub: (Z - a) / (b - a)
                m.fit(loc(X[inside]), yfit[inside], warm=model, reselect_beta=(model is None or it % beta_every == 0))
                model = m
                best = float(np.min(m.predict(loc(X[inside]))[0])) if noisy else float(yfit.min())
                n_c = 2000  # TuRBO-style candidates: perturb a random subset of coordinates of the incumbent
                P = tlb + (tub - tlb) * rng.random((n_c, d))
                mask = rng.random((n_c, d)) < min(1.0, 20.0 / d)
                mask[np.arange(n_c), rng.integers(0, d, n_c)] = True
                Ct = np.where(mask, P, c[None, :])
                acq = lambda Z: -ei_min(*m.predict(loc(Z)), best)
                x, _ = refine(acq, Ct, acq(Ct), rng, n_each=60, lb=tlb, ub=tub, scale=trs.L)
                tr_log.append([float(trs.L), int(inside.sum())])
                method_branch = 'done'
            else:
                m.fit(X, yfit, warm=model, reselect_beta=(model is None or it % beta_every == 0))
                model = m
                best = incumbent(m, X)
                method_branch = method
            if method_branch in ('tt_ts', 'tt_vbts', 'tt_ts_nocd', 'tt_ensts', 'tt_rmts'):
                g, _ = m.sample_function(rng, rms=(method == 'tt_rmts'))
                gv = g(C)
                if method == 'tt_ts_nocd':
                    x = C[int(np.argmin(gv))]
                else:
                    starts = C[np.argsort(gv)[:3]]
                    sols = [coord_descent(g, s, rng) for s in starts]
                    x = min(sols, key=lambda t: t[1])[0]
            elif method_branch == 'tt_lcb':
                acq = lambda Z: (lambda mu, sd: mu - 2.0 * sd)(*m.predict(Z))
                x, _ = refine(acq, C, acq(C), rng)
            elif method_branch == 'tt_mean':
                acq = lambda Z: m.predict(Z)[0]
                x, _ = refine(acq, C, acq(C), rng)
            elif method_branch in ('tt_ei', 'tt_eila', 'tt_eilf'):
                acq = lambda Z: -ei_min(*m.predict(Z), best)
                x, _ = refine(acq, C, acq(C), rng, n_each=60)
        # avoid exact duplicates
        if np.min(np.sum((X - x) ** 2, 1)) < 1e-14:
            x = np.clip(x + 1e-3 * rng.standard_normal(d), 0, 1)
        y_best_before = y.min()
        X = np.vstack([X, x]); y = np.append(y, f(x[None]))
        if tr:
            trs.update(y[-1], y_best_before)
        times.append(time.time() - t0)
        it += 1
        if log and it % 10 == 0:
            print(f'  {method} it={it} best={y.min():.4g} t/it={np.mean(times[-10:]):.2f}s', flush=True)
    out = dict(X=X, y=y, t=times)
    if tr:
        out['tr'] = tr_log
        out['tr_restarts'] = trs.n_restarts
    return out


# ------------------------------------------------------------------ discrete loop
def disc_candidates(X, y, rng, n, n_global=500, n_local=1500):
    d = X.shape[1]
    G = rng.integers(0, n, (n_global, d))
    top = X[np.argsort(y)[:5]]
    idx = rng.choice(len(top), n_local, p=np.array([0.4, 0.2, 0.15, 0.15, 0.1])[:len(top)] / np.array([0.4, 0.2, 0.15, 0.15, 0.1])[:len(top)].sum())
    L = top[idx].copy()
    nflip = rng.choice([1, 1, 2, 3], n_local)
    for i in range(n_local):
        ks = rng.choice(d, nflip[i], replace=False)
        L[i, ks] = (L[i, ks] + rng.integers(1, n, nflip[i])) % n
    return np.vstack([G, L])


def run_bo_discrete(f, d, n, method, budget, n_init, seed, rank=4, prior='anova', log=False, beta_every=15):
    rng = np.random.default_rng(seed)
    if method == 'random':
        X = rng.integers(0, n, (budget, d)); return dict(X=X, y=f(X), t=[0.0] * budget)
    if method == 'protes':
        X, y = run_protes(f, d, n, budget, seed); return dict(X=X, y=y, t=[0.0] * budget)
    if method == 'sa':
        x = rng.integers(0, n, d); fx = f(x[None])[0]; X = [x]; y = [fx]
        T0 = 1.0
        while len(y) < budget:
            xn = x.copy(); k = rng.integers(d); xn[k] = (xn[k] + rng.integers(1, n)) % n
            fn = f(xn[None])[0]; X.append(xn); y.append(fn)
            T = T0 * (1 - len(y) / budget) + 1e-3
            scale = np.std(y) + 1e-12
            if fn < fx or rng.random() < np.exp(-(fn - fx) / (T * scale)):
                x, fx = xn, fn
        return dict(X=np.array(X), y=np.array(y), t=[0.0] * budget)
    X = rng.integers(0, n, (n_init, d)); y = f(X)
    times = [0.0] * n_init
    model = None; it = 0
    while len(y) < budget:
        t0 = time.time()
        C = disc_candidates(X, y, rng, n)
        # drop already evaluated
        seen = {tuple(r) for r in X}
        C = np.array([c for c in C if tuple(c) not in seen])
        best = y.min()
        if method.startswith('hgp'):
            m = HammingGP(seed=seed, n_restarts=1)
            if model is not None:
                m.p = model.p
            m.fit(X, y); model = m
            if method == 'hgp_ts':
                sub = C[:1200]
                s = m.sample_on(sub, rng)[0]
                x = sub[int(np.argmin(s))]
            else:
                mu, sd = m.predict(C)
                x = C[int(np.argmax(ei_min(mu, sd, best)))]
        else:
            tm = {'tt_ts': 'lafull', 'tt_mean': 'map', 'tt_vbts': 'vb', 'tt_rmts': 'map', 'tt_eilf': 'lafull'}[method]
            B = [OneHotBasis(n) for _ in range(d)]
            kw = dict(rank=rank, prior=prior, seed=seed, maxiter=1500 if model is None else 250)
            if tm == 'vb':
                kw['n_sweeps'] = 4
            m = PTTRegressor(B, method=tm, **kw)
            m.fit(X, y, warm=model, reselect_beta=(model is None or it % beta_every == 0)); model = m
            if method == 'tt_eilf':
                mu, sd = m.predict(C)
                x = C[int(np.argmax(ei_min(mu, sd, best)))]
            elif method == 'tt_mean':
                x = C[int(np.argmin(m.predict(C)[0]))]
            else:
                g, _ = m.sample_function(rng, rms=(method == 'tt_rmts'))
                gv = g(C)
                starts = C[np.argsort(gv)[:3]]
                sols = [coord_descent(g, s.astype(float), rng, discrete_n=n) for s in starts]
                x = min(sols, key=lambda t: t[1])[0].astype(int)
                if tuple(x) in seen:
                    x = C[int(np.argmin(gv))]
        X = np.vstack([X, x]); y = np.append(y, f(x[None]))
        times.append(time.time() - t0); it += 1
        if log and it % 10 == 0:
            print(f'  {method} it={it} best={y.min():.4g} t/it={np.mean(times[-10:]):.2f}s', flush=True)
    return dict(X=X, y=y, t=times)
