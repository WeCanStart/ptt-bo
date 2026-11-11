"""GP baselines.
- GPRegressor: sklearn GP, Matern-5/2 (or RBF) ARD + constant + white noise, standardised y, MLE with restarts.
- HammingGP: own GP for categorical inputs with ARD overlap kernel k = s^2 exp(-sum_i theta_i [x_i != x'_i]).
"""
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


class GPRegressor:
    def __init__(self, kernel='matern', n_restarts=2, seed=0, noise_lb=1e-6):
        self.kernel, self.n_restarts, self.seed, self.noise_lb = kernel, n_restarts, seed, noise_lb
        self.gp = None

    def _make(self, d, init=None):
        from sklearn.gaussian_process import kernels as K
        ls = np.full(d, 0.3) if init is None else init['ls']
        base = K.Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=2.5) if self.kernel == 'matern' \
            else K.RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e2))
        return K.ConstantKernel(1.0, (1e-2, 1e2)) * base + K.WhiteKernel(1e-3, (self.noise_lb, 1e-1))

    def fit(self, X, y, optimize=True):
        from sklearn.gaussian_process import GaussianProcessRegressor
        X = np.asarray(X); y = np.asarray(y, float)
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        z = (y - self.ym) / self.ys
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if optimize or self.gp is None:
                kern = self._make(X.shape[1]) if self.gp is None else self.gp.kernel_
                self.gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=self.n_restarts,
                                                   random_state=self.seed)
            else:
                self.gp = GaussianProcessRegressor(self.gp.kernel_, optimizer=None, normalize_y=False)
            self.gp.fit(X, z)
        return self

    def predict(self, X, latent=True):
        mu, sd = self.gp.predict(np.asarray(X), return_std=True)
        if latent:  # remove white noise from predictive variance
            noise = self.gp.kernel_.k2.noise_level
            sd = np.sqrt(np.maximum(sd ** 2 - noise, 1e-12))
        return mu * self.ys + self.ym, sd * self.ys

    @property
    def noise_std(self):
        return np.sqrt(self.gp.kernel_.k2.noise_level) * self.ys

    def sample_on(self, X, rng, n=1):
        mu, C = self.gp.predict(np.asarray(X), return_cov=True)
        noise = self.gp.kernel_.k2.noise_level
        C = C - noise * np.eye(len(X)) + 1e-8 * np.eye(len(X))
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C); L = V * np.sqrt(np.maximum(w, 0))
        return (mu[None] + (L @ rng.standard_normal((len(X), n))).T) * self.ys + self.ym


class HammingGP:
    def __init__(self, seed=0, n_restarts=2):
        self.seed, self.n_restarts = seed, n_restarts

    @staticmethod
    def _diff(A, B):
        return (A[:, None, :] != B[None, :, :]).astype(float)

    def _nll(self, p, Dm, z):
        d = Dm.shape[2]
        th, ls2, lnoise = np.exp(p[:d]), np.exp(p[d]), np.exp(p[d + 1])
        K = ls2 * np.exp(-Dm @ th) + (lnoise + 1e-8) * np.eye(len(z))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(p)
        a = cho_solve(c, z)
        nll = 0.5 * z @ a + np.sum(np.log(np.diag(c[0]))) + 0.5 * len(z) * np.log(2 * np.pi)
        Kinv = cho_solve(c, np.eye(len(z)))
        W = np.outer(a, a) - Kinv
        Ks = ls2 * np.exp(-Dm @ th)
        g = np.empty_like(p)
        for i in range(d):
            g[i] = -0.5 * np.sum(W * (-Ks * Dm[:, :, i] * th[i]))
        g[d] = -0.5 * np.sum(W * Ks)
        g[d + 1] = -0.5 * np.trace(W) * lnoise
        # weak log-normal priors for stability
        nll += 0.5 * np.sum((p[:d] - np.log(0.5)) ** 2) / 4.0
        g[:d] += (p[:d] - np.log(0.5)) / 4.0
        return nll, g

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y, float)
        self.X = X
        self.ym, self.ys = y.mean(), y.std() + 1e-12
        z = (y - self.ym) / self.ys
        Dm = self._diff(X, X)
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)
        best = None
        starts = [np.concatenate([np.full(d, np.log(0.3)), [0.0, np.log(1e-3)]])]
        starts += [np.concatenate([rng.uniform(-3, 1, d), [0.0, np.log(1e-2)]]) for _ in range(self.n_restarts)]
        if getattr(self, 'p', None) is not None and len(self.p) == d + 2:
            starts.append(self.p)
        bounds = [(-6, 3)] * d + [(-3, 3), (np.log(1e-6), np.log(0.5))]
        for s in starts:
            r = minimize(self._nll, s, args=(Dm, z), jac=True, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200})
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x
        th, ls2, ln = np.exp(self.p[:d]), np.exp(self.p[d]), np.exp(self.p[d + 1])
        self.th, self.ls2, self.noise = th, ls2, ln
        K = ls2 * np.exp(-Dm @ th) + (ln + 1e-8) * np.eye(len(z))
        self.c = cho_factor(K, lower=True)
        self.a = cho_solve(self.c, z)
        return self

    def predict(self, Xs):
        Ks = self.ls2 * np.exp(-self._diff(np.asarray(Xs), self.X) @ self.th)
        mu = Ks @ self.a
        v = cho_solve(self.c, Ks.T)
        var = np.maximum(self.ls2 - np.sum(Ks * v.T, 1), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    def sample_on(self, Xs, rng, n=1):
        Xs = np.asarray(Xs)
        Ks = self.ls2 * np.exp(-self._diff(Xs, self.X) @ self.th)
        Kss = self.ls2 * np.exp(-self._diff(Xs, Xs) @ self.th)
        mu = Ks @ self.a
        C = Kss - Ks @ cho_solve(self.c, Ks.T) + 1e-8 * np.eye(len(Xs))
        w, V = np.linalg.eigh(C)
        L = V * np.sqrt(np.maximum(w, 0))
        return (mu[None] + (L @ rng.standard_normal((len(Xs), n))).T) * self.ys + self.ym


class AdditiveGP:
    """First-order additive GP: k(x,x') = sum_i s_i * Matern52_1d(x_i, x'_i; l_i) + noise.  Own MLE (analytic grads)."""

    def __init__(self, seed=0, n_restarts=1):
        self.seed, self.n_restarts = seed, n_restarts

    @staticmethod
    def _m52(R, l):
        a = np.sqrt(5.0) * R / l
        e = np.exp(-a)
        return (1 + a + a ** 2 / 3) * e, (a ** 2 / 3) * (1 + a) * e  # k, dk/dlog l

    def _nll(self, p, Rs, z):
        d = len(Rs); n = len(z)
        ls, ss, nz = np.exp(p[:d]), np.exp(p[d:2 * d]), np.exp(p[-1])
        K = nz * np.eye(n) + 1e-8 * np.eye(n)
        parts = []
        for i in range(d):
            k, dk = self._m52(Rs[i], ls[i]); parts.append((k, dk)); K += ss[i] * k
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(p)
        a = cho_solve(c, z)
        nll = 0.5 * z @ a + np.sum(np.log(np.diag(c[0])))
        W = np.outer(a, a) - cho_solve(c, np.eye(n))
        g = np.empty_like(p)
        for i, (k, dk) in enumerate(parts):
            g[i] = -0.5 * np.sum(W * ss[i] * dk)
            g[d + i] = -0.5 * np.sum(W * ss[i] * k)
        g[-1] = -0.5 * np.trace(W) * nz
        # weak log-normal priors: lengthscales around 0.3, noise around 1e-3
        nll += np.sum((p[:d] - np.log(0.3)) ** 2) / 2.0; g[:d] += (p[:d] - np.log(0.3))
        nll += (p[-1] - np.log(1e-3)) ** 2 / 18.0; g[-1] += (p[-1] - np.log(1e-3)) / 9.0
        return nll, g

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y, float)
        self.X = X; self.ym, self.ys = y.mean(), y.std() + 1e-12
        z = (y - self.ym) / self.ys
        d = X.shape[1]
        Rs = [np.abs(X[:, i:i + 1] - X[:, i][None]) for i in range(d)]
        rng = np.random.default_rng(self.seed)
        starts = [np.concatenate([np.full(d, np.log(0.3)), np.full(d, np.log(1.0 / d)), [np.log(1e-3)]])]
        starts += [np.concatenate([rng.uniform(-3, 0.5, d), rng.uniform(-4, 0, d), [np.log(1e-2)]]) for _ in range(self.n_restarts)]
        if getattr(self, 'p_warm', None) is not None:
            starts = [self.p_warm] + starts[:1]
        bounds = [(np.log(1e-2), np.log(1e2))] * d + [(np.log(1e-4), np.log(1e2))] * d + [(np.log(1e-6), np.log(0.5))]
        best = None
        for s in starts:
            r = minimize(self._nll, s, args=(Rs, z), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300})
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        self.p_best = p
        self.ls, self.ss, self.noise = np.exp(p[:d]), np.exp(p[d:2 * d]), np.exp(p[-1])
        K = self._K(X, X) + (self.noise + 1e-8) * np.eye(len(z))
        self.c = cho_factor(K, lower=True); self.a = cho_solve(self.c, z)
        return self

    def _K(self, A, B):
        K = 0.0
        for i in range(A.shape[1]):
            K = K + self.ss[i] * self._m52(np.abs(A[:, i:i + 1] - B[:, i][None]), self.ls[i])[0]
        return K

    def predict(self, Xs):
        Ks = self._K(np.asarray(Xs), self.X)
        mu = Ks @ self.a
        v = cho_solve(self.c, Ks.T)
        var = np.maximum(self.ss.sum() - np.sum(Ks * v.T, 1), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    @property
    def noise_std(self):
        return np.sqrt(self.noise) * self.ys


class ARDGP:
    """ARD Matern-5/2 GP with MAP hyper-parameters and the dimension-scaled log-normal lengthscale prior
    LogNormal(sqrt(2) + log(d)/2, sqrt(3)) (Hvarfner et al., 2024; BoTorch default since v0.12).
    Inputs in the unit cube, y standardised internally."""

    def __init__(self, seed=0, n_restarts=2, noise_lb=1e-6):
        self.seed, self.n_restarts, self.noise_lb = seed, n_restarts, noise_lb
        self.p = None

    @staticmethod
    def _k(D2s, ls, s):
        """D2s: list of squared differences per dim (n1,n2). returns K, list dK/dlog l_i."""
        Q = sum(D / l ** 2 for D, l in zip(D2s, ls))
        r = np.sqrt(np.maximum(Q, 1e-30))
        e = np.exp(-np.sqrt(5) * r)
        K = s * (1 + np.sqrt(5) * r + 5.0 / 3.0 * Q) * e
        base = s * 5.0 / 3.0 * (1 + np.sqrt(5) * r) * e
        return K, [base * D / l ** 2 for D, l in zip(D2s, ls)]

    def _obj(self, p, D2s, z):
        d = len(D2s); n = len(z)
        ls, s, nz = np.exp(p[:d]), np.exp(p[d]), np.exp(p[d + 1])
        K, dK = self._k(D2s, ls, s)
        Kn = K + (nz + 1e-8) * np.eye(n)
        try:
            c = cho_factor(Kn, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(p)
        a = cho_solve(c, z)
        nll = 0.5 * z @ a + np.sum(np.log(np.diag(c[0])))
        W = np.outer(a, a) - cho_solve(c, np.eye(n))
        g = np.empty_like(p)
        for i in range(d):
            g[i] = -0.5 * np.sum(W * dK[i])
        g[d] = -0.5 * np.sum(W * K)
        g[d + 1] = -0.5 * np.trace(W) * nz
        # priors (negative log densities, in log-parameter space)
        mu_l, sd_l = np.sqrt(2) + 0.5 * np.log(d), np.sqrt(3)
        nll += np.sum((p[:d] - mu_l) ** 2) / (2 * sd_l ** 2); g[:d] += (p[:d] - mu_l) / sd_l ** 2
        nll += 0.5 * p[d] ** 2; g[d] += p[d]
        mu_n, sd_n = np.log(1e-3), 3.0
        nll += (p[d + 1] - mu_n) ** 2 / (2 * sd_n ** 2); g[d + 1] += (p[d + 1] - mu_n) / sd_n ** 2
        return nll, g

    def fit(self, X, y, optimize=True):
        X = np.asarray(X); y = np.asarray(y, float)
        self.X = X; self.ym, self.ys = y.mean(), y.std() + 1e-12
        z = (y - self.ym) / self.ys
        d = X.shape[1]
        D2s = [(X[:, i:i + 1] - X[:, i][None]) ** 2 for i in range(d)]
        bounds = [(np.log(1e-2), np.log(1e3))] * d + [(np.log(1e-2), np.log(1e2)), (np.log(self.noise_lb), np.log(0.5))]
        if optimize or self.p is None:
            rng = np.random.default_rng(self.seed)
            mu_l = np.sqrt(2) + 0.5 * np.log(d)
            starts = [np.concatenate([np.full(d, mu_l - 1.0), [0.0, np.log(1e-3)]])]
            if self.p is not None:
                starts.append(self.p)
            starts += [np.concatenate([rng.normal(mu_l - 1.0, 1.0, d), [0.0, np.log(1e-2)]]) for _ in range(self.n_restarts - 1)]
            best = None
            for s in starts:
                r = minimize(self._obj, np.clip(s, [b[0] for b in bounds], [b[1] for b in bounds]), args=(D2s, z),
                             jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
                if best is None or r.fun < best.fun:
                    best = r
            self.p = best.x
        d_ = d
        self.ls, self.s, self.noise = np.exp(self.p[:d_]), np.exp(self.p[d_]), np.exp(self.p[d_ + 1])
        K, _ = self._k(D2s, self.ls, self.s)
        self.c = cho_factor(K + (self.noise + 1e-8) * np.eye(len(z)), lower=True)
        self.a = cho_solve(self.c, z)
        return self

    def _Kx(self, A, B):
        D2s = [(A[:, i:i + 1] - B[:, i][None]) ** 2 for i in range(A.shape[1])]
        return self._k(D2s, self.ls, self.s)[0]

    def predict(self, Xs):
        Ks = self._Kx(np.asarray(Xs), self.X)
        mu = Ks @ self.a
        v = cho_solve(self.c, Ks.T)
        var = np.maximum(self.s - np.sum(Ks * v.T, 1), 1e-12)
        return mu * self.ys + self.ym, np.sqrt(var) * self.ys

    @property
    def noise_std(self):
        return np.sqrt(self.noise) * self.ys

    def sample_on(self, Xs, rng, n=1):
        Xs = np.asarray(Xs)
        Ks = self._Kx(Xs, self.X)
        C = self._Kx(Xs, Xs) - Ks @ cho_solve(self.c, Ks.T) + 1e-8 * np.eye(len(Xs))
        mu = Ks @ self.a
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C); L = V * np.sqrt(np.maximum(w, 0))
        return (mu[None] + (L @ rng.standard_normal((len(Xs), n))).T) * self.ys + self.ym
