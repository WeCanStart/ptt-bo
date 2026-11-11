"""Benchmark problems. Every problem works on the unit cube u in [0,1]^d (minimisation)."""
import numpy as np


class Problem:
    def __init__(self, name, d, lb, ub, f, fopt=None, xopt=None):
        self.name, self.d = name, d
        self.lb, self.ub = np.broadcast_to(np.asarray(lb, float), (d,)).copy(), np.broadcast_to(np.asarray(ub, float), (d,)).copy()
        self._f, self.fopt, self.xopt = f, fopt, xopt

    def to_x(self, U):
        return self.lb + np.asarray(U) * (self.ub - self.lb)

    def __call__(self, U):
        U = np.atleast_2d(U)
        return self._f(self.to_x(U))


# ------------------------------------------------------------------ classic functions (X in original domain)
def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return (x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10


_H3A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
_H6A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_HAL = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann3(X):
    return -np.sum(_HAL * np.exp(-np.einsum('ij,nij->ni', _H3A, (X[:, None, :] - _H3P[None]) ** 2)), axis=1)


def hartmann6(X):
    return -np.sum(_HAL * np.exp(-np.einsum('ij,nij->ni', _H6A, (X[:, None, :] - _H6P[None]) ** 2)), axis=1)


def rosenbrock(X):
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)


def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley(X):
    d = X.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, 1) / d)) - np.exp(np.sum(np.cos(2 * np.pi * X), 1) / d) + 20 + np.e)


def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1 + np.sum(X ** 2, 1) / 4000 - np.prod(np.cos(X / np.sqrt(i)), 1)


def levy(X):
    w = 1 + (X - 1) / 4
    t1 = np.sin(np.pi * w[:, 0]) ** 2
    t2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), 1)
    t3 = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def styblinski_tang(X):
    return 0.5 * np.sum(X ** 4 - 16 * X ** 2 + 5 * X, 1)


def michalewicz(X, m=10):
    i = np.arange(1, X.shape[1] + 1)
    return -np.sum(np.sin(X) * np.sin(i * X ** 2 / np.pi) ** (2 * m), 1)


def dixon_price(X):
    """chain of neighbour interactions (TT-rank 3 in natural order)"""
    i = np.arange(2, X.shape[1] + 1)
    return (X[:, 0] - 1) ** 2 + np.sum(i * (2 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, 1)


def powell(X):
    """blocks of 4 variables with interactions inside each block (incl. non-neighbours x1-x4)"""
    a, b, c, e = X[:, 0::4], X[:, 1::4], X[:, 2::4], X[:, 3::4]
    return np.sum((a + 10 * b) ** 2 + 5 * (c - e) ** 2 + (b - 2 * c) ** 4 + 10 * (a - e) ** 4, 1)


def zakharov(X):
    """sum of squares + polynomial of ONE linear form: low TT-rank but every pair of variables interacts"""
    s = X @ (0.5 * np.arange(1, X.shape[1] + 1))
    return np.sum(X ** 2, 1) + s ** 2 + s ** 4


def get_problem(name, d=None, rotate=False, permute=False, seed=0, shift=True):
    """rotate: f(Q (x - c) + c) with random orthogonal Q (c = domain centre); destroys axis-aligned (low-rank)
    structure while keeping the landscape. permute: random variable order (matters only for chain-like f).
    shift: random shift of the optimum (avoids optimum at the cube centre for symmetric functions)."""
    rng = np.random.default_rng(1000 + seed)
    table = {
        'branin': (2, [-5, 0], [10, 15], branin, 0.397887),
        'hartmann3': (3, 0, 1, hartmann3, -3.86278),
        'hartmann6': (6, 0, 1, hartmann6, -3.32237),
        'rosenbrock': (d, -2.048, 2.048, rosenbrock, 0.0),
        'rastrigin': (d, -5.12, 5.12, rastrigin, 0.0),
        'ackley': (d, -32.768, 32.768, ackley, 0.0),
        'griewank': (d, -10, 10, griewank, 0.0),  # narrowed domain (classic is +-600, nearly flat)
        'levy': (d, -10, 10, levy, 0.0),
        'styblinski': (d, -5, 5, styblinski_tang, -39.16599 * (d or 0)),
        'michalewicz': (d, 0, np.pi, michalewicz, {2: -1.8013, 5: -4.687658, 10: -9.66015}.get(d)),  # known optima (m=10)
        'dixonprice': (d, -10, 10, dixon_price, 0.0),
        'powell': (d, -4, 5, powell, 0.0),
        'zakharov': (d, -5, 10, zakharov, 0.0),
    }
    dd, lb, ub, f, fopt = table[name]
    if dd is None:
        raise ValueError('need d')
    lbv = np.broadcast_to(np.asarray(lb, float), (dd,))
    ubv = np.broadcast_to(np.asarray(ub, float), (dd,))
    tag = name + (f'{dd}' if name not in ('branin', 'hartmann3', 'hartmann6') else '')
    c = 0.5 * (lbv + ubv)
    hr = 0.5 * (ubv - lbv)
    zopt = {'rosenbrock': np.ones(dd), 'rastrigin': np.zeros(dd), 'ackley': np.zeros(dd),
            'griewank': np.zeros(dd), 'levy': np.ones(dd), 'styblinski': -2.903534 * np.ones(dd),
            'dixonprice': 2.0 ** (-(2.0 ** np.arange(1, dd + 1) - 2) / 2.0 ** np.arange(1, dd + 1)),
            'powell': np.zeros(dd), 'zakharov': np.zeros(dd)}.get(name)
    s = np.zeros(dd)
    if shift and name in ('rastrigin', 'ackley', 'griewank'):  # optimum not at the cube centre
        s = rng.uniform(-0.25, 0.25, dd) * hr
    Q = np.eye(dd)
    if rotate:
        Q, _ = np.linalg.qr(rng.standard_normal((dd, dd)))
        tag += '_rot'
    perm = np.arange(dd)
    if permute:
        perm = rng.permutation(dd)
        tag += '_perm'
    if rotate or np.any(s != 0) or permute:
        # z = Q (x[perm] - c - s) + c ;  f(x) = f0(z)
        def g(X, Q=Q, c=c, s=s, perm=perm, f0=f):
            return f0((X[:, perm] - c - s) @ Q.T + c)
        if zopt is not None:
            xp = c + s + Q.T @ (zopt - c)          # optimum in permuted coordinates
            x = np.empty(dd); x[perm] = xp
            if np.any(x < lbv) or np.any(x > ubv):
                fopt = None                         # optimum left the box -> use best-found as reference
    else:
        g = f
    return Problem(tag, dd, lbv, ubv, g, fopt)


# ------------------------------------------------------------------ synthetic smooth low-rank functional TT
def smooth_tt_problem(d, rank, seed=0, nf=4, decay=0.6, name=None):
    """f(u) = G_1(u_1) ... G_d(u_d), entries of G_k are smooth random trig polynomials (low frequencies).
    Exactly TT-rank `rank`, smooth (GP-friendly). Normalised to unit std over the cube."""
    from .tt import FourierBasis, PTT
    rng = np.random.default_rng(5000 + seed)
    B = FourierBasis(nf=nf, ell=0.0)
    lam = np.concatenate([[1.0], np.repeat(decay ** np.arange(1, nf + 1), 2)])
    r = [1] + [rank] * (d - 1) + [1]
    cores = []
    for k in range(d):
        W = rng.standard_normal((r[k], B.m, r[k + 1])) * np.sqrt(lam)[None, :, None] / np.sqrt(np.sqrt(r[k] * r[k + 1]) * lam.sum())
        W[:, 0, :] += np.eye(r[k], r[k + 1]) * 0.9  # keep a dominant smooth component (avoids exponential decay)
        cores.append(W)
    Bs = [B] * d
    U = rng.random((4000, d))
    vals = PTT.eval_cores(cores, [B(U[:, k]) for k in range(d)])
    mu, sd = vals.mean(), vals.std()

    def f(X):
        return (PTT.eval_cores(cores, [B(X[:, k]) for k in range(d)]) - mu) / sd
    p = Problem(name or f'smoothTT{d}_r{rank}', d, 0.0, 1.0, f, None)
    p.cores, p.basis = cores, B
    return p
