"""Anytime view of a BO study: aggregate score vs number of evaluations.
score(k) = IQM over (problem, seed) of log10( regret_k / median regret_k of random search on that problem ),
95% stratified bootstrap band. Lower is better; 0 = random search.
usage: python experiments/plot_anytime.py STUDY OUT_NAME method1,method2,... [kmax]
Study e13/e14 may borrow runs from e9 via the same SOURCES rule as plot_bo/stats_bo."""
import json, glob, sys
import numpy as np
sys.path.insert(0, '.')
from pttbo.plot import plt

study, out, methods = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
kmax = int(sys.argv[4]) if len(sys.argv) > 4 else None
SOURCES = {'e13': ['e13', 'e9']}
runs = [json.load(open(f)) for s in SOURCES.get(study, [study]) for f in glob.glob(f'results/{s}/*.json')]
runs = [r for r in runs if r['method'] in methods + ['random']]
NAMES = {'random': 'Случайный поиск', 'cmaes': 'CMA-ES', 'gpp_ei': 'GP-EI (ARD, приоры)', 'gpa_ei': 'Аддитивный GP-EI',
         'bt_logei': 'BoTorch LogEI', 'bt_turbo': 'TuRBO-1', 'bt_turbo_warp': 'TuRBO-1 + warp',
         'bt_logei_warp': 'LogEI + warp', 'tt_eilf': 'PTT-EI (наш, глобальный)', 'tt_tr': 'TR-PTT-EI (наш)',
         'tt_tr_warp': 'TR-PTT-EI + warp', 'tt_tr_r2': 'TR-PTT-EI, ранг 2', 'tt_eilf_warp': 'PTT-EI + warp'}
STYLE = {'random': ('#8a8984', ':'), 'cmaes': ('#e8590c', ':'), 'gpp_ei': ('#0b6e4f', '--'), 'gpa_ei': ('#52514e', '--'),
         'bt_logei': ('#2f9e44', '--'), 'bt_logei_warp': ('#2f9e44', '--'), 'bt_turbo': ('#f08c00', '--'),
         'bt_turbo_warp': ('#9c5d00', '-.'), 'tt_eilf': ('#c92a2a', '-'), 'tt_tr': ('#1c4fd8', '-'),
         'tt_tr_warp': ('#111111', '-'), 'tt_tr_r2': ('#9c36b5', '-'), 'tt_eilf_warp': ('#c92a2a', '-.')}
probs = sorted({r['problem'] for r in runs})
L = min(len(r['y']) for r in runs)
if kmax:
    L = min(L, kmax)
ks = np.arange(20, L + 1)


def curve(r):
    c = np.minimum.accumulate(np.asarray(r['y'][:L])) - r['fopt']
    return np.log10(np.maximum(c, 1e-10))


C = {}
for p in probs:
    R = [r for r in runs if r['problem'] == p]
    base = np.median(np.array([curve(r) for r in R if r['method'] == 'random']), 0)
    for r in R:
        C.setdefault(r['method'], {}).setdefault(p, []).append(curve(r) - base)


def iqm(a, axis=0):
    a = np.sort(a, axis=axis); n = a.shape[axis]
    return np.take(a, range(int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))), axis=axis).mean(axis)


rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for m in methods:
    if m not in C:
        continue
    blocks = [np.array(v) for v in C[m].values()]
    allc = np.concatenate(blocks)
    est = iqm(allc)
    boots = np.array([iqm(np.concatenate([b[rng.integers(0, len(b), len(b))] for b in blocks])) for _ in range(300)])
    lo, hi = np.quantile(boots, 0.025, 0), np.quantile(boots, 0.975, 0)
    col, ls = STYLE.get(m, ('#333333', '-'))
    ax.plot(np.arange(1, L + 1)[19:], est[19:], color=col, ls=ls, lw=2.0 if m.startswith('tt') else 1.6,
            label=NAMES.get(m, m) + (f'  ({len(C[m])} задач)' if len(C[m]) != len(probs) else ''))
    ax.fill_between(np.arange(1, L + 1)[19:], lo[19:], hi[19:], color=col, alpha=0.12, lw=0)
ax.axhline(0, color='#8a8984', lw=0.8)
ax.set_xlabel('число вычислений целевой функции $N$')
ax.set_ylabel('IQM $\\log_{10}$(регрет / регрет случайного поиска)')
ax.legend(fontsize=7.5, frameon=False, loc='lower left')
plt.tight_layout(); plt.savefig(f'figures/{out}.png', dpi=200)
print('saved', out, 'problems', len(probs), 'L', L)
