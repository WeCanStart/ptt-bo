"""Regret curves and tables for E7 (continuous) and E8 (discrete)."""
import json, sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from pttbo.plot import plt, PAL, GRAY

which = sys.argv[1] if len(sys.argv) > 1 else 'e7'
# e13 re-uses the unwarped baselines of e9 (same problems and seeds)
SOURCES = {'e13': [('e13', None), ('e9', {'random', 'gpp_ei', 'tt_eilf', 'bt_logei', 'bt_turbo'})]}
runs = []
for src, keep in SOURCES.get(which, [(which, None)]):
    for f in glob.glob(f'results/{src}/*.json'):
        if '__exactmin__' in f:
            continue
        r = json.load(open(f))
        if keep is None or r['method'] in keep:
            runs.append(r)
exact = {}
for f in glob.glob(f'results/{which}/*__exactmin__*.json'):
    r = json.load(open(f)); exact[(r['problem'], r['seed'])] = r['fmin']
probs = sorted({r['problem'] for r in runs}, key=lambda p: ['branin', 'hartmann6', 'rosenbrock10', 'styblinski10', 'styblinski10_rot',
                                                            'smoothTT20_r3', 'lowrank_cat20', 'bqp20', 'bqp20_perm', 'contam25'].index(p)
               if p in ['branin', 'hartmann6', 'rosenbrock10', 'styblinski10', 'styblinski10_rot', 'smoothTT20_r3',
                        'lowrank_cat20', 'bqp20', 'bqp20_perm', 'contam25'] else 99)
probs = sorted(probs, key=lambda p: (['branin', 'hartmann6', 'rosenbrock10', 'styblinski10', 'styblinski10_rot', 'smoothTT20_r3',
                                      'lowrank_cat20', 'bqp20', 'bqp20_perm', 'contam25'].index(p)
                                     if p in ['branin', 'hartmann6', 'rosenbrock10', 'styblinski10', 'styblinski10_rot',
                                              'smoothTT20_r3', 'lowrank_cat20', 'bqp20', 'bqp20_perm', 'contam25'] else 99, p))
order = ['random', 'cmaes', 'sa', 'protes', 'gp_ei', 'gp_ts', 'gpp_ei', 'gpp_ts', 'gpa_ei', 'bt_logei', 'bt_turbo', 'hgp_ei', 'hgp_ts', 'tt_mean', 'tt_ts', 'tt_rmts', 'tt_eilf',
         'tt_eilf_r2', 'tt_eilf_r8', 'tt_eilf_l015', 'tt_eilf_l040', 'tt_eilf_zero',
         'gpp_ei_warp', 'bt_logei_warp', 'bt_lognei', 'bt_turbo_warp', 'tt_eilf_warp', 'tt_tr', 'tt_tr_warp', 'tt_tr_r2']
names = {'random': 'Random', 'cmaes': 'CMA-ES', 'sa': 'Sim. annealing', 'protes': 'PROTES (numpy)', 'gp_ei': 'GP-EI',
         'gp_ts': 'GP-TS', 'gpp_ei': 'GP-EI (+priors)', 'gpp_ts': 'GP-TS (+priors)', 'gpa_ei': 'Additive GP-EI', 'hgp_ei': 'GP-Hamming EI', 'hgp_ts': 'GP-Hamming TS', 'tt_mean': 'PTT mean (no uncert.)',
         'tt_ts': 'PTT-TS (Laplace-full)', 'tt_rmts': 'PTT-TS (randomised MAP)', 'tt_eilf': 'PTT-EI (Laplace-full)'}
color = {m: PAL[i % 8] for i, m in enumerate(['random', 'cmaes', 'gp_ei', 'gp_ts', 'tt_mean', 'tt_ts', 'tt_rmts', 'tt_eilf'])}
color.update({'sa': PAL[1], 'protes': PAL[7], 'hgp_ei': PAL[2], 'hgp_ts': PAL[3], 'gpp_ei': '#0b6e4f', 'gpp_ts': '#a35f00', 'gpa_ei': '#52514e',
              'bt_logei': PAL[2], 'bt_turbo': PAL[3], 'tt_eilf_r2': PAL[0], 'tt_eilf_r8': PAL[1], 'tt_eilf_l015': PAL[4],
              'tt_eilf_l040': PAL[5], 'tt_eilf_zero': PAL[6]})
names.update({'bt_logei': 'BoTorch LogEI', 'bt_turbo': 'TuRBO-1 (BoTorch)', 'tt_eilf_r2': 'PTT-EI r=2', 'tt_eilf_r8': 'PTT-EI r=8',
              'tt_eilf_l015': 'PTT-EI ell=0.15', 'tt_eilf_l040': 'PTT-EI ell=0.40', 'tt_eilf_zero': 'PTT-EI zero prior',
              'gpp_ei_warp': 'GP-EI (+priors) + warp', 'bt_logei_warp': 'BoTorch LogEI + warp', 'bt_lognei': 'BoTorch qLogNEI',
              'bt_turbo_warp': 'TuRBO-1 + warp', 'tt_eilf_warp': 'PTT-EI + warp', 'tt_tr': 'TR-PTT-EI',
              'tt_tr_warp': 'TR-PTT-EI + warp', 'tt_tr_r2': 'TR-PTT-EI rank 2'})
if which != 'e13':  # e13 draws warped and unwarped versions in one colour (unwarped dotted); elsewhere keep them apart
    PAL_WARP = {'bt_turbo_warp': '#9c5d00', 'bt_logei_warp': '#2b8a3e'}
else:
    PAL_WARP = {'bt_turbo_warp': PAL[3], 'bt_logei_warp': PAL[2]}
color.update({'gpp_ei_warp': '#0b6e4f', 'bt_logei_warp': PAL_WARP['bt_logei_warp'], 'bt_lognei': PAL[2],
              'bt_turbo_warp': PAL_WARP['bt_turbo_warp'],
              'tt_eilf_warp': PAL[7], 'tt_tr': '#3b5bdb', 'tt_tr_warp': '#111111', 'tt_tr_r2': '#9c36b5'})


def ref_value(p, seed):
    """f* : known optimum, exact enumeration (BQP), else best value seen by ANY run on that problem/seed."""
    known = [r.get('fopt') for r in runs if r['problem'] == p and r.get('fopt') is not None]
    if known:
        return known[0]
    if (p, seed) in exact:
        return exact[(p, seed)]
    return min(min(r['y']) for r in runs if r['problem'] == p and r['seed'] == seed)


import math
ncol = len(probs) if len(probs) <= 6 else 4
nrow = math.ceil(len(probs) / ncol)
fig, ax = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow + 0.4), squeeze=False)
ax = ax.ravel()
for a in ax[len(probs):]:
    a.axis('off')
rows = []
for j, p in enumerate(probs):
    meths = [m for m in order if any(r['problem'] == p and r['method'] == m for r in runs)]
    for m in meths:
        R = [r for r in runs if r['problem'] == p and r['method'] == m]
        curves = []
        for r in R:
            fstar = ref_value(p, r['seed'])
            if 'y_noisy' in r:  # noisy study: true f at the recommended point (= best noisy observation so far)
                idx = np.array([int(np.argmin(r['y_noisy'][:k + 1])) for k in range(len(r['y']))])
                c = np.array(r['y'])[idx] - fstar
            else:
                c = np.minimum.accumulate(np.array(r['y'])) - fstar
            curves.append(np.maximum(c, 1e-8))
        L = min(len(c) for c in curves)
        C = np.array([c[:L] for c in curves])
        med = np.median(C, 0); lo = np.quantile(C, 0.25, 0); hi = np.quantile(C, 0.75, 0)
        x = np.arange(1, L + 1)
        ls = '-' if m.startswith('tt') else ('--' if m.startswith(('gp', 'hgp', 'bt')) else ':')
        if which == 'e13' and m in ('gpp_ei', 'bt_logei', 'bt_turbo', 'tt_eilf'):
            ls = (0, (1, 1.5))  # unwarped references from e9: dotted, same colour as their warped versions
        if not (m in ('gp_ts', 'gpp_ts') and which == 'e7'):  # keep the figure readable (numbers are in the table)
            ax[j].plot(x, med, color=color[m], ls=ls, lw=1.8, label=names[m])
            ax[j].fill_between(x, lo, hi, color=color[m], alpha=0.10, lw=0)
        tt = [np.mean(r['t'][-20:]) for r in R if 't' in r]
        rows.append(dict(problem=p, method=m, n_runs=len(R), final_median=float(np.median(C[:, -1])),
                         final_q25=float(np.quantile(C[:, -1], 0.25)), final_q75=float(np.quantile(C[:, -1], 0.75)),
                         auc_log=float(np.mean(np.log10(C))), sec_per_iter=float(np.mean(tt)) if tt else np.nan))
    ax[j].set_yscale('log'); ax[j].set_title(p, fontsize=10); ax[j].set_xlabel('evaluations')
ax[0].set_ylabel('simple regret (median, IQR)')
h, l = ax[0].get_legend_handles_labels()
for a in ax[1:len(probs)]:
    hh, ll = a.get_legend_handles_labels()
    for x_, y_ in zip(hh, ll):
        if y_ not in l:
            h.append(x_); l.append(y_)
fig.legend(h, l, loc='lower center', ncol=min(len(l), 6), fontsize=8, bbox_to_anchor=(0.5, -0.12 / nrow))
plt.tight_layout(); plt.savefig(f'figures/{which}_regret.png')
T = pd.DataFrame(rows)
T.to_csv(f'results/{which}_table.csv', index=False)
pd.set_option('display.width', 200)
print(T.pivot_table(index='method', columns='problem', values='final_median').round(4).to_string())
print(T.pivot_table(index='method', columns='problem', values='auc_log').round(2).to_string())
print(T.pivot_table(index='method', columns='problem', values='sec_per_iter').round(2).to_string())
# rank-based summary: per problem/seed rank of final regret
rk = []
for p in probs:
    for s in sorted({r['seed'] for r in runs if r['problem'] == p}):
        R = [r for r in runs if r['problem'] == p and r['seed'] == s and r['method'] in order]
        fstar = ref_value(p, s)
        L = min(len(r['y']) for r in R)
        vals = {r['method']: min(r['y'][:L]) - fstar for r in R}
        ranks = pd.Series(vals).rank()
        for m, v in ranks.items():
            rk.append(dict(problem=p, seed=s, method=m, rank=v))
RK = pd.DataFrame(rk)
print(RK.pivot_table(index='method', columns='problem', values='rank', aggfunc='mean').round(2).to_string())
RK.to_csv(f'results/{which}_ranks.csv', index=False)
