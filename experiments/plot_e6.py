import json, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from pttbo.plot import plt, PAL, GRAY

rows = []
for f in glob.glob('results/e6_part*.json'):
    rows += json.load(open(f))
A = pd.DataFrame(rows)
probs = ['branin', 'hartmann6', 'hartmann6_rot', 'rosenbrock10', 'rosenbrock10_perm', 'styblinski10', 'styblinski10_rot',
         'levy10', 'ackley10', 'smoothTT20_r3']
probs = [p for p in probs if p in A.problem.unique()]
main = [('gp_matern', 'GP Matern-5/2 ARD'), ('tt_anova_lafull', 'PTT anova prior, Laplace-full'),
        ('tt_anova_vb', 'PTT anova prior, VB'), ('tt_zero_lafull', 'PTT zero prior, Laplace-full')]
fig, ax = plt.subplots(3, len(probs), figsize=(2.6 * len(probs), 7.6), sharex=False)
for j, p in enumerate(probs):
    sub = A[A.problem == p]
    for i, (m, lab) in enumerate(main):
        s = sub[sub.method == m].groupby('n')[['nrmse', 'nll', 'cov95']].mean()
        if len(s) == 0:
            continue
        ax[0, j].plot(s.index, s.nrmse, marker='o', color=PAL[i], label=lab)
        ax[1, j].plot(s.index, s.nll, marker='o', color=PAL[i])
        ax[2, j].plot(s.index, s.cov95, marker='o', color=PAL[i])
    ax[0, j].set_title(p, fontsize=9)
    for a in ax[:, j]:
        a.set_xscale('log'); a.set_xticks(sorted(sub.n.unique())); a.set_xticklabels(sorted(sub.n.unique()), fontsize=7)
        a.minorticks_off()
    ax[2, j].axhline(0.95, color=GRAY, ls='--', lw=1); ax[2, j].set_ylim(0.3, 1.02)
    ax[2, j].set_xlabel('n train')
ax[0, 0].set_ylabel('normalised RMSE'); ax[1, 0].set_ylabel('NLL (Gaussian)'); ax[2, 0].set_ylabel('95% coverage')
ax[0, 0].legend(fontsize=7, loc='upper right')
plt.tight_layout(); plt.savefig('figures/e6_surrogate.png')
tab = A.groupby(['problem', 'n', 'method'])[['nrmse', 'nrmse_top10', 'nll', 'cov95', 'time']].mean().round(3)
tab.to_csv('results/e6_table.csv')
piv = A.pivot_table(index=['problem', 'n'], columns='method', values='nrmse', aggfunc='mean').round(3)
cols = [c for c in ['gp_matern', 'tt_anova_map', 'tt_anova_vb', 'tt_anova_ens', 'tt_zero_map', 'tt_zero_vb'] if c in piv.columns]
print(piv[cols].to_string())
piv = A.pivot_table(index=['problem', 'n'], columns='method', values='nll', aggfunc='mean').round(2)
cols = [c for c in ['gp_matern', 'tt_anova_lafull', 'tt_anova_lablock', 'tt_anova_ladiag', 'tt_anova_la1', 'tt_anova_vb', 'tt_anova_ens', 'tt_zero_lafull', 'tt_zero_vb'] if c in piv.columns]
print(piv[cols].to_string())
piv = A.pivot_table(index=['problem', 'n'], columns='method', values='cov95', aggfunc='mean').round(2)
print(piv[cols].to_string())
print(A.groupby('method')[['nrmse', 'nll', 'cov95', 'time']].median().round(3).to_string())
