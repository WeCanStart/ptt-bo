import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from pttbo.plot import plt, PAL, GRAY

r = json.load(open('results/e4_e5_all.json'))
A = pd.DataFrame(r['e4'])
if os.path.exists('results/e4_e5_rms.json'):
    extra = pd.DataFrame(json.load(open('results/e4_e5_rms.json'))['e4'])
    A = pd.concat([A, extra[extra.method.isin(['rms'])]])
B = pd.DataFrame(r['e5'])
order = ['gibbs', 'lafull', 'lablock', 'lablock_lin', 'ladiag', 'la1', 'vb', 'vbdiag', 'ens', 'rms']
order = [o for o in order if o in A.method.unique()]
labels = {'gibbs': 'Gibbs (gold)', 'lafull': 'Laplace full (lin.)', 'lablock': 'Laplace block + moments',
          'lablock_lin': 'Laplace block (lin.)', 'ladiag': 'Laplace diag (EKF-like)', 'la1': 'Laplace 1 core (=TT feat.+BLR)',
          'vb': 'VB block', 'vbdiag': 'VB diag', 'ens': 'deep ensemble (8)', 'rms': 'randomised-MAP ens. (8)'}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for j, prior in enumerate(['zero', 'anova']):
    sub = A[A.prior == prior]
    ys = np.arange(len(order))
    med = [sub[sub.method == m].log_sd_ratio.median() for m in order]
    lo = [sub[sub.method == m].log_sd_ratio.quantile(0.25) for m in order]
    hi = [sub[sub.method == m].log_sd_ratio.quantile(0.75) for m in order]
    off = -0.15 + 0.3 * j
    ax[0].errorbar(np.clip(med, -3, 1), ys + off, xerr=[np.clip(med, -3, 1) - np.clip(lo, -3, 1), np.clip(hi, -3, 1) - np.clip(med, -3, 1)],
                   fmt='o', color=PAL[j], label=f'{prior} prior', ms=5, lw=1.5)
    cov = [sub[sub.method == m].cov95_truth.median() for m in order]
    ax[1].scatter(cov, ys + off, color=PAL[j], s=25, label=f'{prior} prior')
    md = [sub[sub.method == m].mean_dev.median() for m in order]
    ax[2].scatter(md, ys + off, color=PAL[j], s=25)
ax[0].axvline(0, color=GRAY, lw=1)
ax[0].set_yticks(np.arange(len(order))); ax[0].set_yticklabels([labels[o] for o in order]); ax[0].invert_yaxis()
ax[0].set_xlabel('log(sd_method / sd_Gibbs)  (median, IQR; clipped at -3)'); ax[0].set_title('E4: predictive std vs gold standard')
ax[0].legend(fontsize=8)
ax[1].axvline(0.95, color=GRAY, lw=1, ls='--'); ax[1].set_yticks(np.arange(len(order))); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
ax[1].set_xlabel('95% coverage of the TRUE function (median)'); ax[1].set_title('calibration w.r.t. truth')
ax[2].set_yticks(np.arange(len(order))); ax[2].set_yticklabels([]); ax[2].invert_yaxis()
ax[2].set_xlabel('|mean - Gibbs mean| / mean Gibbs sd'); ax[2].set_title('mean shift vs gold standard')
plt.tight_layout(); plt.savefig('figures/e4_posterior_approx.png')
tab = A.groupby(['prior', 'method'])[['mean_dev', 'log_sd_ratio', 'rms_log_sd_ratio', 'cov95_truth', 'nll_truth', 'rmse_truth', 'rhat']].median().round(3)
tab.to_csv('results/e4_table.csv'); print(tab.to_string())
tab5 = B.groupby(['prior', 'n', 'family'])[['mean_dev', 'log_sd_ratio', 'q_width_ratio', 'cov95_truth', 'nll_truth', 'rmse_truth', 'rhat_f']].median().round(3)
tab5.to_csv('results/e5_table.csv'); print(tab5.to_string())
