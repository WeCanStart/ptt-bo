"""Probabilistic (functional) Tensor Train: representation, bases and exact moment propagation.

Model:  f(x) = G_1(x_1) G_2(x_2) ... G_d(x_d),   G_k(x) = sum_j W_k[:, j, :] phi_kj(x)
W_k has shape (r_{k-1}, m_k, r_k), r_0 = r_d = 1.

A "probabilistic TT" (PTT) stores, for every core, the mean W_k and a covariance of vec(W_k)
(row-major order (a, j, b)). Covariance can be 'full' (D_k x D_k), 'diag' (D_k,) or None (deterministic).
Cores are assumed mutually independent. Under that assumption the mean, the variance and the
covariance of f are EXACT and depend only on first/second moments of the elements (no distributional
assumption) -- see `moments()`.

One-hot basis = classical grid/discrete TT.
"""
import numpy as np


# ----------------------------------------------------------------------------------------------
# Bases  (all bases satisfy sum_j lam_j phi_j(u)^2 == 1 for every u, so the iid prior is
# "stationary in amplitude": prior Var f(x) is the same for all x)
# ----------------------------------------------------------------------------------------------
class FourierBasis:
    """phi(u) = [1, cos(pi w u), sin(pi w u)]_{w=1..nf} on u in [0, 1] (period 2 => no forced periodicity).

    Prior weights lam follow the spectral density of an RBF kernel with lengthscale `ell` (unit cube units),
    normalised to sum 1. Then with i.i.d. zero-mean cores and r -> inf the PTT prior tends to a GP with the
    product (= ARD-RBF-like) kernel  prod_k sum_w lam_w cos(pi w (u_k - u'_k)).
    """

    def __init__(self, nf=6, ell=0.25):
        self.nf, self.ell = nf, ell
        self.m = 2 * nf + 1
        w = np.arange(nf + 1)
        s = np.exp(-0.5 * (np.pi * w * ell) ** 2)
        s[0] *= 0.5  # constant term gets half the weight of a (cos,sin) pair -- standard cosine-series
        s = s / s.sum()
        lam = np.empty(self.m)
        lam[0] = s[0]
        lam[1::2] = s[1:]
        lam[2::2] = s[1:]
        self.lam = lam
        self.freq = np.concatenate([[0.0], np.repeat(np.arange(1, nf + 1), 2)])
        self.const = np.eye(self.m)[0]  # coefficients of the constant function

    def __call__(self, u):
        u = np.asarray(u, float)
        out = np.empty(u.shape + (self.m,))
        out[..., 0] = 1.0
        for w in range(1, self.nf + 1):
            out[..., 2 * w - 1] = np.cos(np.pi * w * u)
            out[..., 2 * w] = np.sin(np.pi * w * u)
        return out

    def kernel1d(self, u, v):
        """Limit (r->inf) per-dimension kernel sum_j lam_j phi_j(u) phi_j(v)."""
        return self(u) @ (self.lam[:, None] * self(v).T)


class OneHotBasis:
    """Discrete / grid TT: phi_j(x) = [x == j], x integer in {0..n-1}."""

    def __init__(self, n):
        self.m = n
        self.lam = np.ones(n)
        self.const = np.ones(n)

    def __call__(self, x):
        x = np.asarray(x).astype(int)
        out = np.zeros(x.shape + (self.m,))
        np.put_along_axis(out, x[..., None], 1.0, axis=-1)
        return out


# ----------------------------------------------------------------------------------------------
# PTT container
# ----------------------------------------------------------------------------------------------
class PTT:
    def __init__(self, means, covs=None, bases=None):
        self.means = [np.asarray(W, float) for W in means]
        self.covs = covs if covs is not None else [None] * len(self.means)
        self.bases = bases
        self.d = len(self.means)

    @property
    def ranks(self):
        return [1] + [W.shape[2] for W in self.means]

    def copy(self):
        return PTT([W.copy() for W in self.means],
                   [None if C is None else C.copy() for C in self.covs], self.bases)

    def features(self, X):
        """list of (N, m_k) basis evaluations."""
        X = np.atleast_2d(X)
        return [self.bases[k](X[:, k]) for k in range(self.d)]

    # ---------------- deterministic evaluation (of the mean cores, or of any given cores) -------
    @staticmethod
    def eval_cores(cores, Phis):
        N = Phis[0].shape[0]
        l = np.ones((N, 1))
        for W, P in zip(cores, Phis):
            G = np.einsum('ajb,nj->nab', W, P, optimize=True)
            l = np.einsum('na,nab->nb', l, G)
        return l[:, 0]

    def mean(self, X=None, Phis=None):
        Phis = self.features(X) if Phis is None else Phis
        return self.eval_cores(self.means, Phis)

    def sample_cores(self, rng, n=1):
        """Sample cores assuming Gaussian independent cores (used for Thompson sampling)."""
        out = []
        for _ in range(n):
            cs = []
            for W, C in zip(self.means, self.covs):
                if C is None:
                    cs.append(W.copy())
                elif C.ndim == 1:
                    cs.append(W + np.sqrt(np.maximum(C, 0)).reshape(W.shape) * rng.standard_normal(W.shape))
                else:
                    L = _safe_chol(C)
                    cs.append(W + (L @ rng.standard_normal(W.size)).reshape(W.shape))
            out.append(cs)
        return out

    # ---------------- exact moments --------------------------------------------------------------
    def moments(self, X=None, Phis=None, linearized=False, chunk=2048, return_parts=False):
        """Exact E f(x) and Var f(x) under independence of cores (distribution-free).

        Stable recursion (no E[f^2]-E[f]^2 cancellation):
            p_k = p_{k-1} M_k
            S_k = M_k^T S_{k-1} M_k + V_k[ Q_{k-1} ],     Q_{k-1} = S_{k-1} + p_{k-1}^T p_{k-1}
        where V_k[Q]_{bd} = sum_{ac} Q_ac Cov(G_ab, G_cd).  `linearized=True` uses Q = p^T p (first-order
        / delta-method variance, = sum over single-core contributions only).
        """
        Phis = self.features(X) if Phis is None else Phis
        N = Phis[0].shape[0]
        mu = np.empty(N)
        var = np.empty(N)
        for s in range(0, N, chunk):
            P = [F[s:s + chunk] for F in Phis]
            m, v = self._moments_chunk(P, linearized)
            mu[s:s + chunk], var[s:s + chunk] = m, v
        return mu, np.maximum(var, 0.0)

    def _V_apply(self, k, P, Q, side='left'):
        """Apply covariance part of E[G (x) G] for core k at points P (N, m).
        side='left':  out[n,b,d] = sum_{a,c} Q[n,a,c] Cov(G_ab, G_cd)
        side='right': out[n,a,c] = sum_{b,d} Cov(G_ab, G_cd) Q[n,b,d]
        """
        W, C = self.means[k], self.covs[k]
        r1, m, r2 = W.shape
        N = P.shape[0]
        if C is None:
            return np.zeros((N, r2, r2)) if side == 'left' else np.zeros((N, r1, r1))
        if C.ndim == 1:  # diagonal: Cov(G_ab, G_cd) = delta_ac delta_bd sum_j phi_j^2 c[a,j,b]
            c = C.reshape(r1, m, r2)
            s = np.einsum('nj,ajb->nab', P ** 2, c)  # (N, r1, r2)
            if side == 'left':
                diagQ = np.einsum('naa->na', Q)
                out = np.zeros((N, r2, r2))
                idx = np.arange(r2)
                out[:, idx, idx] = np.einsum('na,nab->nb', diagQ, s)
                return out
            else:
                diagQ = np.einsum('nbb->nb', Q)
                out = np.zeros((N, r1, r1))
                idx = np.arange(r1)
                out[:, idx, idx] = np.einsum('nab,nb->na', s, diagQ)
                return out
        C6 = C.reshape(r1, m, r2, r1, m, r2)
        if side == 'left':
            # out[n,b,d] = sum_{a,c,j,k} Q[n,a,c] P[n,j] P[n,k] C6[a,j,b,c,k,d]
            T = np.einsum('nac,ajbckd->njbkd', Q, C6, optimize=True)
            return np.einsum('njbkd,nj,nk->nbd', T, P, P, optimize=True)
        else:
            T = np.einsum('ajbckd,nbd->najck', C6, Q, optimize=True)
            return np.einsum('najck,nj,nk->nac', T, P, P, optimize=True)

    def _moments_chunk(self, P, linearized):
        N = P[0].shape[0]
        p = np.ones((N, 1))
        S = np.zeros((N, 1, 1))
        for k in range(self.d):
            M = np.einsum('ajb,nj->nab', self.means[k], P[k], optimize=True)
            Q = np.einsum('na,nc->nac', p, p) if linearized else S + np.einsum('na,nc->nac', p, p)
            S = np.einsum('nab,nac,ncd->nbd', M, S, M, optimize=True) + self._V_apply(k, P[k], Q, 'left')
            p = np.einsum('na,nab->nb', p, M)
        return p[:, 0], S[:, 0, 0]

    def cov_matrix(self, X1, X2):
        """Exact Cov(f(x), f(x')) for all pairs (small sets only; O(N1 N2 d r^4))."""
        P1, P2 = self.features(X1), self.features(X2)
        n1, n2 = len(X1), len(X2)
        p1 = np.ones((n1, 1)); p2 = np.ones((n2, 1))
        S = np.zeros((n1, n2, 1, 1))
        for k in range(self.d):
            W, C = self.means[k], self.covs[k]
            r1, m, r2 = W.shape
            M1 = np.einsum('ajb,nj->nab', W, P1[k]); M2 = np.einsum('ajb,nj->nab', W, P2[k])
            Q = S + np.einsum('na,ic->niac', p1, p2)
            Snew = np.einsum('nab,niac,icd->nibd', M1, S, M2, optimize=True)
            if C is not None:
                Cf = np.diag(C) if C.ndim == 1 else C
                C6 = Cf.reshape(r1, m, r2, r1, m, r2)
                T = np.einsum('niac,ajbckd->nijbkd', Q, C6, optimize=True)
                Snew += np.einsum('nijbkd,nj,ik->nibd', T, P1[k], P2[k], optimize=True)
            S = Snew
            p1 = np.einsum('na,nab->nb', p1, M1); p2 = np.einsum('na,nab->nb', p2, M2)
        return S[:, :, 0, 0]


def _safe_chol(C):
    C = 0.5 * (C + C.T)
    jitter = 1e-12 * max(np.trace(C) / len(C), 1e-30)
    for _ in range(8):
        try:
            return np.linalg.cholesky(C + jitter * np.eye(len(C)))
        except np.linalg.LinAlgError:
            jitter *= 100
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.maximum(w, 0))


def random_tt(ranks, ms, rng, scale=1.0):
    d = len(ms)
    r = [1] + list(ranks) + [1]
    return [rng.standard_normal((r[k], ms[k], r[k + 1])) * scale for k in range(d)]


def prior_ptt(d, bases, rank, scale2=1.0, zero_mean=True, rng=None, mean_scale=0.0):
    """i.i.d. zero-mean prior: Var(W_k[a,j,b]) = scale2 * lam_j / sqrt(r_{k-1} r_k)  =>  Var f(x) = scale2^d."""
    r = [1] + [rank] * (d - 1) + [1]
    means, covs = [], []
    for k in range(d):
        m = bases[k].m
        v = np.einsum('a,j,b->ajb', np.ones(r[k]), bases[k].lam, np.ones(r[k + 1])) * scale2 / np.sqrt(r[k] * r[k + 1])
        means.append(np.zeros((r[k], m, r[k + 1])) if zero_mean else mean_scale * rng.standard_normal((r[k], m, r[k + 1])))
        covs.append(v.reshape(-1))
    return PTT(means, covs, bases)
