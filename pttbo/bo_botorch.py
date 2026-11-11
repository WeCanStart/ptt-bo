"""BoTorch baselines (optional dependency: `pip install botorch`).

  bt_logei : SingleTaskGP (BoTorch defaults: Matern/RBF with dimension-scaled priors in recent versions)
             + analytic LogExpectedImprovement, optimize_acqf (10 restarts, 512 raw samples)
  bt_lognei: same GP + qLogNoisyExpectedImprovement (for noisy observations)
  bt_turbo : TuRBO-1 (Eriksson et al., 2019) following the official BoTorch tutorial: one trust region,
             Thompson sampling on 2000-5000 perturbed Sobol candidates, restarts when the region collapses.
All minimise f on the unit cube (internally maximise -f). Interface mirrors pttbo.bo.run_bo.
warp=True : the GP is fitted to -power_warp(y) (HEBO-style output warping, see pttbo.bo.power_warp).
noisy=True: TuRBO uses BoTorch's default (noise-inferring) likelihood instead of the near-noiseless one.
"""
import math, time
import numpy as np


def _imports():
    try:
        import torch
        import botorch  # noqa: F401
    except ImportError as e:
        raise ImportError('BoTorch не установлен: в окружении ttgp выполните  pip install botorch') from e
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    return torch


def _fit_gp(X, Y, covar_module=None, likelihood=None):
    import torch, warnings
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    kw = {}
    if covar_module is not None:
        kw['covar_module'] = covar_module
    if likelihood is not None:
        kw['likelihood'] = likelihood
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1), **kw)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # keep the initial hyper-parameters if the fit fails (rare)
    return model


def _target(y, warp):
    import torch
    from .bo import power_warp
    z = power_warp(y) if warp else np.asarray(y, float)
    return torch.tensor(-z, dtype=torch.float64).unsqueeze(-1)


def run_logei(f, d, budget, n_init, seed, warp=False, noisy_acq=False):
    torch = _imports()
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.acquisition import qLogNoisyExpectedImprovement
    from botorch.optim import optimize_acqf
    from torch.quasirandom import SobolEngine
    torch.manual_seed(seed)
    X = SobolEngine(d, scramble=True, seed=seed).draw(n_init).to(torch.float64)
    y = f(X.numpy())
    times = [0.0] * n_init
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    while len(y) < budget:
        t0 = time.time()
        Y = _target(y, warp)
        model = _fit_gp(X, Y)
        if noisy_acq:
            acq = qLogNoisyExpectedImprovement(model=model, X_baseline=X, prune_baseline=True)
        else:
            acq = LogExpectedImprovement(model, best_f=Y.max())
        cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
        x = cand.detach().clamp(0, 1)
        X = torch.cat([X, x]); y = np.append(y, f(x.numpy()))
        times.append(time.time() - t0)
    return dict(X=X.numpy(), y=y, t=times)


class _TurboState:
    def __init__(self, dim):
        self.dim, self.length, self.length_min, self.length_max = dim, 0.8, 0.5 ** 7, 1.6
        self.failure_counter, self.success_counter = 0, 0
        self.failure_tolerance = math.ceil(max(4.0, float(dim)))
        self.success_tolerance = 10
        self.best_value, self.restart_triggered = -float('inf'), False

    def update(self, y_next):
        if y_next > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1; self.failure_counter = 0
        else:
            self.success_counter = 0; self.failure_counter += 1
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max); self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0; self.failure_counter = 0
        self.best_value = max(self.best_value, y_next)
        if self.length < self.length_min:
            self.restart_triggered = True


def run_turbo(f, d, budget, n_init, seed, warp=False, noisy=False):
    torch = _imports()
    from botorch.generation import MaxPosteriorSampling
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.constraints import Interval
    from torch.quasirandom import SobolEngine
    torch.manual_seed(seed)
    n_cand = min(5000, max(2000, 200 * d))
    Xall, yall, times = [], [], []
    restart = 0
    while len(yall) < budget:
        # (re)start: fresh Sobol design and a fresh trust region
        m0 = min(n_init, budget - len(yall))
        Xl = SobolEngine(d, scramble=True, seed=seed * 100 + restart).draw(m0).to(torch.float64)
        yl = f(Xl.numpy())
        Xall.append(Xl.numpy()); yall.extend(yl.tolist()); times.extend([0.0] * m0)
        state = _TurboState(d)
        state.best_value = float(np.max(-yl))
        while len(yall) < budget and not state.restart_triggered:
            t0 = time.time()
            Y = _target(yl, warp)
            lik = None if noisy else GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            cov = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_constraint=Interval(0.005, 4.0)))
            model = _fit_gp(Xl, Y, covar_module=cov, likelihood=lik)
            x_center = Xl[Y.argmax(), :].clone()
            ls = model.covar_module.base_kernel.lengthscale.detach().view(-1)
            w = ls / ls.mean(); w = w / torch.prod(w.pow(1.0 / len(w)))
            lb = torch.clamp(x_center - w * state.length / 2.0, 0.0, 1.0)
            ub = torch.clamp(x_center + w * state.length / 2.0, 0.0, 1.0)
            sob = SobolEngine(d, scramble=True, seed=int(torch.randint(0, 10 ** 6, (1,))))
            pert = lb + (ub - lb) * sob.draw(n_cand).to(torch.float64)
            mask = torch.rand(n_cand, d) <= min(20.0 / d, 1.0)
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind):
                mask[ind, torch.randint(0, d, (len(ind),))] = True
            Xc = x_center.expand(n_cand, d).clone(); Xc[mask] = pert[mask]
            with torch.no_grad():
                x = MaxPosteriorSampling(model=model, replacement=False)(Xc, num_samples=1)
            yn = f(x.numpy())
            Xl = torch.cat([Xl, x]); yl = np.append(yl, yn)
            Xall.append(x.numpy()); yall.extend(yn.tolist())
            state.update(float(-yn[0]))
            times.append(time.time() - t0)
        restart += 1
    return dict(X=np.vstack(Xall)[:budget], y=np.array(yall)[:budget], t=times[:budget])


def run_botorch(prob, method, budget, n_init, seed, warp=False, noisy=False):
    f = lambda U: prob(np.asarray(U))
    if method == 'bt_logei':
        return run_logei(f, prob.d, budget, n_init, seed, warp=warp)
    if method == 'bt_lognei':
        return run_logei(f, prob.d, budget, n_init, seed, warp=warp, noisy_acq=True)
    if method == 'bt_turbo':
        return run_turbo(f, prob.d, budget, n_init, seed, warp=warp, noisy=noisy)
    raise ValueError(method)
