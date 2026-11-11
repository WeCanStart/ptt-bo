"""Inference for probabilistic TT regression  y = f(x) + eps,  eps ~ N(0, 1/beta)  (y standardised).

Prior: W_k = W0_k + Delta_k,  Delta_k[a,j,b] ~ (0, v0_k[a,j,b] / alpha) independent.
   prior='zero'  : W0 = 0,      v0 = lam_j / sqrt(r_{k-1} r_k)        -> product kernel (GP limit r->inf)
   prior='anova' : W0 = I on the constant basis function, v0 = s2 * lam_j, s2 = 2^{1/d}-1
                   -> f = prod (I + Delta_k) = 1 + sum_k Delta_k + sum_{j<k} Delta_j Delta_k + ...
                      i.e. an ANOVA (additive + interactions) prior; Var f(x) ~ 1.

Methods (same model / basis / rank / MAP):
  'map'      : joint L-BFGS MAP (no uncertainty)
  'la1'      : MAP + Laplace on ONE core (others fixed) == "TT features + Bayesian linear regression"
  'lablock'  : MAP + plug-in Laplace on every core (block-diag, cores independent) + exact moment propagation
  'ladiag'   : same, diagonal covariances ("diagonal EKF-like")
  'lafull'   : MAP + joint Gauss-Newton Laplace over all cores (cross-core correlations), linearised predictive
  'vb'       : mean-field VB over cores (full Gaussian per core; CAVI accounts for other cores' variance);
               Gamma hyper-posteriors on beta and a global alpha
  'vbdiag'   : fully factorised VB
  'ens'      : ensemble of K MAP fits from random initialisations
  `gibbs()`  : gold-standard sampler (Normal / Student-t / Uniform priors, matched mean & variance)
"""
import numpy as np
from scipy.optimize import minimize
from .tt import PTT, _safe_chol


def _kron_stats(QL, Phi, QR):
    """sum_n QL[n] (x) phi_n phi_n^T (x) QR[n]  in (a,j,b),(c,k,d) order -> (D, D)."""
    N, r1, _ = QL.shape
    m = Phi.shape[1]
    r2 = QR.shape[1]
    Z = np.einsum('nac,nj,nk->najck', QL, Phi, Phi, optimize=True).reshape(N, -1)
    A = (Z.T @ QR.reshape(N, -1)).reshape(r1, m, r1, m, r2, r2).transpose(0, 1, 4, 2, 3, 5)
    D = r1 * m * r2
    return A.reshape(D, D)


def _kron_vec(pL, Phi, pR, w):
    return np.einsum('n,na,nj,nb->ajb', w, pL, Phi, pR, optimize=True).reshape(-1)


def _chol_solve(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


class PTTRegressor:
    def __init__(self, bases, rank=4, method='vb', prior='anova', beta=None, alpha=1.0,
                 beta_grid=(1., 3., 10., 30., 100., 300., 1000.), n_sweeps=10, n_ens=8, seed=0, la_core=0,
                 maxiter=1500, noise_floor=1e-6, a0=1e-3, b0=1e-3, learn_alpha=True, init_noise=0.05):
        self.bases, self.rank, self.method, self.prior = bases, rank, method, prior
        self.beta_fixed, self.alpha0, self.beta_grid = beta, alpha, beta_grid
        self.n_sweeps, self.n_ens, self.seed, self.la_core = n_sweeps, n_ens, seed, la_core
        self.maxiter, self.noise_floor, self.a0, self.b0 = maxiter, noise_floor, a0, b0
        self.learn_alpha, self.init_noise = learn_alpha, init_noise
        self.d = len(bases)
        self.rng = np.random.default_rng(seed)
        r = [1] + [rank] * (self.d - 1) + [1]
        self.r = r
        self.shapes = [(r[k], bases[k].m, r[k + 1]) for k in range(self.d)]
        self.W0, self.v0 = [], []
        s2 = 2.0 ** (1.0 / self.d) - 1.0
        for k, sh in enumerate(self.shapes):
            W0 = np.zeros(sh)
            if prior == 'anova':
                W0 = np.einsum('ab,j->ajb', np.eye(sh[0], sh[2]), bases[k].const)
                v = s2 * np.broadcast_to(bases[k].lam[None, :, None], sh)
            else:
                v = np.broadcast_to(bases[k].lam[None, :, None], sh) / np.sqrt(sh[0] * sh[2])
            self.W0.append(W0)
            self.v0.append(np.array(v).reshape(-1))
        self.ptt, self.members, self.beta_cv = None, None, None

    # ------------------------------------------------------------------ helpers
    def _init_cores(self, rng, scale=None):
        scale = self.init_noise if scale is None else scale
        out = []
        for k, sh in enumerate(self.shapes):
            W = rng.standard_normal(sh) * np.sqrt(self.v0[k]).reshape(sh) * scale
            if self.prior == 'anova':
                W = W + self.W0[k]
            else:  # zero prior: start near identity on the constant function (well conditioned)
                W = W + np.einsum('ab,j->ajb', np.eye(sh[0], sh[2]), self.bases[k].const)
            out.append(W)
        return out

    def _phis(self, X):
        X = np.asarray(X)
        return [self.bases[k](X[:, k]) for k in range(self.d)]

    def _pack(self, cores):
        return np.concatenate([c.reshape(-1) for c in cores])

    def _unpack(self, w):
        out, s = [], 0
        for sh in self.shapes:
            n = int(np.prod(sh)); out.append(w[s:s + n].reshape(sh)); s += n
        return out

    @staticmethod
    def _right_stats(ptt, Phis, second=True):
        d = ptt.d
        N = Phis[0].shape[0]
        pR, QR = [None] * d, [None] * d
        p = np.ones((N, 1)); Q = np.ones((N, 1, 1))
        pR[d - 1], QR[d - 1] = p, Q
        for k in range(d - 1, 0, -1):
            M = np.einsum('ajb,nj->nab', ptt.means[k], Phis[k], optimize=True)
            pn = np.einsum('nab,nb->na', M, p)
            if second:
                Qn = np.einsum('nab,nbd,ncd->nac', M, Q, M, optimize=True) + ptt._V_apply(k, Phis[k], Q, 'right')
            else:
                Qn = None
            p, Q = pn, Qn
            pR[k - 1], QR[k - 1] = p, Q
        return pR, QR

    def _jacobians(self, cores, Phis):
        ptt = PTT(cores, None, self.bases)
        pR, _ = self._right_stats(ptt, Phis, second=False)
        N = Phis[0].shape[0]
        p = np.ones((N, 1)); Js = []
        for k in range(self.d):
            Js.append(np.einsum('na,nj,nb->najb', p, Phis[k], pR[k], optimize=True).reshape(N, -1))
            M = np.einsum('ajb,nj->nab', cores[k], Phis[k], optimize=True)
            p = np.einsum('na,nab->nb', p, M)
        return Js, p[:, 0]

    # ------------------------------------------------------------------ MAP (joint L-BFGS)
    def _map(self, Phis, y, beta, alpha, init, maxiter=None, w0=None):
        w0 = self._pack(self.W0) if w0 is None else w0
        pv = np.concatenate(self.v0) / alpha

        def obj(w):
            cores = self._unpack(w)
            Js, f = self._jacobians(cores, Phis)
            res = y - f
            dw = w - w0
            o = 0.5 * beta * res @ res + 0.5 * np.sum(dw ** 2 / pv)
            g = np.concatenate([-beta * J.T @ res for J in Js]) + dw / pv
            return o, g
        r = minimize(obj, self._pack(init), jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter or self.maxiter, 'maxcor': 30})
        return self._unpack(r.x), r.fun

    def _select_beta(self, Phis, y, rng):
        N = len(y)
        if N < 12:
            return 100.0
        folds = np.array_split(rng.permutation(N), 3)
        scores = []
        for beta in self.beta_grid:
            err = 0.0
            for f in folds:
                tr = np.setdiff1d(np.arange(N), f)
                cores, _ = self._map([P[tr] for P in Phis], y[tr], beta, self.alpha0, self._init_cores(rng), maxiter=300)
                pred = PTT.eval_cores(cores, [P[f] for P in Phis])
                err += np.sum((pred - y[f]) ** 2)
            scores.append(err)
        return float(self.beta_grid[int(np.argmin(scores))])

    # ------------------------------------------------------------------ VB sweep
    def _vb_sweep(self, ptt, Phis, y, beta, alpha, diag=False):
        d, N = ptt.d, len(y)
        pR, QR = self._right_stats(ptt, Phis, True)
        pL, QL = [None] * d, [None] * d
        pL[0], QL[0] = np.ones((N, 1)), np.ones((N, 1, 1))
        order = list(range(d)) + list(range(d - 2, -1, -1))
        prev = None
        for k in order:
            if prev is not None:
                if k > prev:
                    M = np.einsum('ajb,nj->nab', ptt.means[prev], Phis[prev], optimize=True)
                    pL[k] = np.einsum('na,nab->nb', pL[prev], M)
                    QL[k] = np.einsum('nab,nac,ncd->nbd', M, QL[prev], M, optimize=True) + ptt._V_apply(prev, Phis[prev], QL[prev], 'left')
                else:
                    M = np.einsum('ajb,nj->nab', ptt.means[prev], Phis[prev], optimize=True)
                    pR[k] = np.einsum('nab,nb->na', M, pR[prev])
                    QR[k] = np.einsum('nab,nbd,ncd->nac', M, QR[prev], M, optimize=True) + ptt._V_apply(prev, Phis[prev], QR[prev], 'right')
            A = _kron_stats(QL[k], Phis[k], QR[k])
            b = _kron_vec(pL[k], Phis[k], pR[k], y)
            pp = alpha / self.v0[k]
            P = beta * A + np.diag(pp)
            L = _safe_chol(P)
            mu = _chol_solve(L, beta * b + pp * self.W0[k].reshape(-1))
            ptt.means[k] = mu.reshape(self.shapes[k])
            if diag:
                ptt.covs[k] = 1.0 / np.diag(P)
            else:
                Li = np.linalg.inv(L)
                ptt.covs[k] = Li.T @ Li
            prev = k
        return ptt

    def _vb_hyper(self, ptt, Phis, y):
        mu, var = ptt.moments(Phis=Phis)
        N = len(y)
        beta = (self.a0 + 0.5 * N) / (self.b0 + 0.5 * np.sum((y - mu) ** 2 + var))
        beta = min(beta, 1.0 / self.noise_floor)
        alpha = self.alpha
        if self.learn_alpha:
            num, den = self.a0, self.b0
            for k in range(ptt.d):
                m = ptt.means[k].reshape(-1) - self.W0[k].reshape(-1)
                C = ptt.covs[k]
                dv = np.diag(C) if C.ndim == 2 else C
                num += 0.5 * m.size
                den += 0.5 * np.sum((m ** 2 + dv) / self.v0[k])
            alpha = num / den
        return beta, alpha

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, warm=None, reselect_beta=True):
        y = np.asarray(y, float)
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        z = (y - self.ym) / self.ys
        Phis = self._phis(X)
        self._Phis, self._z = Phis, z
        rng = self.rng
        self.alpha = self.alpha0
        if self.beta_fixed is not None:
            beta = self.beta_fixed
        elif (not reselect_beta) and warm is not None and warm.beta_cv is not None:
            beta = warm.beta_cv
        else:
            beta = self._select_beta(Phis, z, rng)
        self.beta_cv = beta
        init = [W.copy() for W in warm.ptt.means] if (warm is not None and warm.ptt is not None) else self._init_cores(rng)
        meth = self.method
        if meth in ('ens', 'rms'):
            # 'ens': plain deep ensemble (random inits).  'rms': randomised-MAP / anchored ensemble
            # (perturbed targets + prior-sampled anchors; exact posterior samples for linear-Gaussian models)
            self.members = []
            w0 = self._pack(self.W0)
            pv = np.concatenate(self.v0) / self.alpha
            prev = warm.members if (warm is not None and getattr(warm, 'members', None) is not None
                                    and len(warm.members) == self.n_ens) else None
            for e in range(self.n_ens):
                if prev is not None:
                    ini = [W.copy() for W in prev[e].means]
                else:
                    ini = init if (e == 0 and warm is not None and meth == 'ens') else self._init_cores(rng, scale=1.0)
                if meth == 'rms':
                    if prev is not None and hasattr(prev[e], 'anchor'):
                        anc, eps_seed = prev[e].anchor, prev[e].eps_seed
                    else:
                        anc, eps_seed = w0 + np.sqrt(pv) * rng.standard_normal(len(w0)), int(rng.integers(1 << 31))
                    eps = np.random.default_rng(eps_seed).standard_normal(len(z) + 100000)[:len(z)] / np.sqrt(beta)
                    cores, _ = self._map(Phis, z + eps, beta, self.alpha, ini, w0=anc,
                                         maxiter=None if prev is None else self.maxiter)
                    p = PTT(cores, None, self.bases); p.anchor, p.eps_seed = anc, eps_seed
                else:
                    cores, _ = self._map(Phis, z, beta, self.alpha, ini)
                    p = PTT(cores, None, self.bases)
                self.members.append(p)
            self.ptt, self.beta = self.members[0], beta
            return self
        cores, _ = self._map(Phis, z, beta, self.alpha, init)
        ptt = PTT(cores, None, self.bases)
        if meth in ('vb', 'vbdiag'):
            diag = meth == 'vbdiag'
            ptt.covs = [(1e-3 * v if diag else np.diag(1e-3 * v)) for v in self.v0]
            for it in range(self.n_sweeps):
                ptt = self._vb_sweep(ptt, Phis, z, beta, self.alpha, diag)
                if self.beta_fixed is None:
                    beta, self.alpha = self._vb_hyper(ptt, Phis, z)
        elif meth in ('la1', 'lablock', 'ladiag', 'lafull'):
            Js, _ = self._jacobians(cores, Phis)
            Ps = [beta * J.T @ J + np.diag(self.alpha / v) for J, v in zip(Js, self.v0)]
            ptt.covs = [None] * self.d
            if meth == 'la1':
                ptt.covs[self.la_core] = np.linalg.inv(Ps[self.la_core])
            elif meth == 'lablock':
                ptt.covs = [np.linalg.inv(P) for P in Ps]
            elif meth == 'ladiag':
                ptt.covs = [1.0 / np.diag(P) for P in Ps]
            else:
                J = np.concatenate(Js, axis=1)
                Pf = beta * J.T @ J + np.diag(np.concatenate(self.v0) ** -1 * self.alpha)
                self.Lfull = _safe_chol(Pf)
        elif meth != 'map':
            raise ValueError(meth)
        self.ptt, self.beta = ptt, beta
        return self

    # ------------------------------------------------------------------ predict
    def predict_std(self, X, linearized=False):
        Phis = self._phis(X)
        if self.method in ('ens', 'rms'):
            F = np.stack([PTT.eval_cores(m.means, Phis) for m in self.members])
            return F.mean(0), F.std(0)
        if self.method == 'map':
            return self.ptt.mean(Phis=Phis), np.zeros(Phis[0].shape[0])
        if self.method == 'lafull':
            N = Phis[0].shape[0]
            mu = self.ptt.mean(Phis=Phis)
            var = np.empty(N)
            for s in range(0, N, 512):
                Js, _ = self._jacobians(self.ptt.means, [P[s:s + 512] for P in Phis])
                V = np.linalg.solve(self.Lfull, np.concatenate(Js, axis=1).T)
                var[s:s + 512] = np.sum(V ** 2, axis=0)
            return mu, np.sqrt(var)
        mu, var = self.ptt.moments(Phis=Phis, linearized=linearized)
        return mu, np.sqrt(var)

    def predict(self, X, linearized=False):
        mu, sd = self.predict_std(X, linearized)
        return mu * self.ys + self.ym, sd * self.ys

    @property
    def noise_std(self):
        return self.ys / np.sqrt(self.beta)

    def sample_cores(self, rng):
        if self.method in ('ens', 'rms'):
            return [W.copy() for W in self.members[rng.integers(len(self.members))].means]
        if self.method == 'lafull':
            w = np.linalg.solve(self.Lfull.T, rng.standard_normal(self.Lfull.shape[0]))
            return [W + dw for W, dw in zip(self.ptt.means, self._unpack(w))]
        if self.method == 'map':
            return [W.copy() for W in self.ptt.means]
        return self.ptt.sample_cores(rng, 1)[0]

    def sample_rms(self, rng, maxiter=200):
        """One randomised-MAP posterior sample: perturbed targets + prior-sampled anchor, warm-started at MAP."""
        w0 = self._pack(self.W0)
        pv = np.concatenate(self.v0) / self.alpha
        anc = w0 + np.sqrt(pv) * rng.standard_normal(len(w0))
        eps = rng.standard_normal(len(self._z)) / np.sqrt(self.beta)
        cores, _ = self._map(self._Phis, self._z + eps, self.beta, self.alpha,
                             [W.copy() for W in self.ptt.means], maxiter=maxiter, w0=anc)
        return cores

    def sample_function(self, rng, rms=False):
        cores = self.sample_rms(rng) if rms else self.sample_cores(rng)
        ym, ys, bases = self.ym, self.ys, self.bases

        def g(X):
            X = np.asarray(X)
            return PTT.eval_cores(cores, [bases[k](X[:, k]) for k in range(len(bases))]) * ys + ym
        return g, cores


# ======================================================================================================
# Gibbs sampler (gold-standard posterior) for fixed hyper-parameters
# ======================================================================================================
def gibbs(model, X, y_std, beta, alpha, n_samples=300, burn=300, thin=2, prior='normal', nu=5.0, init=None,
          rng=None):
    """Gibbs over cores; y_std are standardised targets. prior in {'normal','t','uniform'} for Delta_k with
    matched mean 0 and variance v0/alpha. Normal & t: block-core conjugate updates (t via Gamma scale mixture).
    Uniform: element-wise truncated normal. Returns list of core samples."""
    from scipy.stats import truncnorm
    rng = rng or np.random.default_rng(0)
    d = model.d
    Phis = model._phis(X)
    y = np.asarray(y_std, float)
    N = len(y)
    pv = [model.v0[k] / alpha for k in range(d)]
    W0 = [w.reshape(-1) for w in model.W0]
    cores = [W.copy() for W in init] if init is not None else model._init_cores(rng)
    lam = [np.ones_like(v) for v in pv]
    samples = []
    for it in range(burn + n_samples * thin):
        for k in range(d):
            Js, _ = model._jacobians(cores, Phis)  # J_k independent of core k
            J = Js[k]
            if prior in ('normal', 't'):
                if prior == 't':
                    st2 = pv[k] * (nu - 2) / nu
                    dw = cores[k].reshape(-1) - W0[k]
                    lam[k] = rng.gamma((nu + 1) / 2, 1.0 / ((nu + dw ** 2 / st2) / 2))
                    pp = lam[k] / st2
                else:
                    pp = 1.0 / pv[k]
                P = beta * J.T @ J + np.diag(pp)
                L = _safe_chol(P)
                mu = _chol_solve(L, beta * J.T @ y + pp * W0[k])
                w = mu + np.linalg.solve(L.T, rng.standard_normal(len(mu)))
                cores[k] = w.reshape(cores[k].shape)
            else:
                w = cores[k].reshape(-1).copy()
                f = J @ w
                hw = np.sqrt(3 * pv[k])
                for e in rng.permutation(len(w)):
                    a = J[:, e]
                    aa = a @ a
                    lo_b, hi_b = W0[k][e] - hw[e], W0[k][e] + hw[e]
                    if aa < 1e-14:
                        new = rng.uniform(lo_b, hi_b)
                    else:
                        m = (a @ (y - (f - a * w[e]))) / aa
                        s = 1.0 / np.sqrt(beta * aa)
                        new = truncnorm.rvs((lo_b - m) / s, (hi_b - m) / s, loc=m, scale=s, random_state=rng)
                    f += a * (new - w[e])
                    w[e] = new
                cores[k] = w.reshape(cores[k].shape)
        if it >= burn and (it - burn) % thin == 0:
            samples.append([W.copy() for W in cores])
    return samples


def predict_samples(model, samples, X):
    Phis = model._phis(X)
    return np.stack([PTT.eval_cores(c, Phis) for c in samples])
