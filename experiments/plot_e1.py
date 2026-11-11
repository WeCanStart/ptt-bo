import json, sys
import numpy as np
sys.path.insert(0, '.')
from pttbo.plot import plt, PAL
from matplotlib.ticker import ScalarFormatter, NullFormatter
res = json.load(open('results/e1_moments.json'))
dists = ['normal', 'uniform', 't5', 'laplace', 'twopoint']
# ---------------------------------------------------------------- figures
import pandas as pd
A = pd.DataFrame(res['a'])
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for i, dist in enumerate(dists):
    sub = A[A.dist == dist]
    ax[0].scatter(sub.z_mean, sub.z_var, s=12, color=PAL[i], label=dist, alpha=0.8)
ax[0].axhline(0, color='#999', lw=0.8); ax[0].axvline(0, color='#999', lw=0.8)
ax[0].set_xlabel('z-score of mean (MC vs exact)'); ax[0].set_ylabel('z-score of variance')
ax[0].set_title('(a) exact moments = MC for all distributions'); ax[0].legend(fontsize=8)
for i, dist in enumerate(dists):
    sub = A[(A.dist == dist) & (A.r == 4) & (A.cv == 0.3)].sort_values('d')
    ax[1].plot(sub.d, sub.exkurt, marker='o', color=PAL[i], label=dist)
    sub = A[(A.dist == dist) & (A.r == 4) & (A.cv == 0.6)].sort_values('d')
    ax[2].plot(sub.d, sub.cover95_gauss, marker='o', color=PAL[i], label=dist)
ax[1].set_xscale('log'); ax[1].set_xlabel('d'); ax[1].set_ylabel('excess kurtosis of f'); ax[1].set_title('(b) tails, r=4, cv=0.3')
ax[2].axhline(0.95, color='#999', lw=0.8, ls='--'); ax[2].set_xscale('log'); ax[2].set_xlabel('d')
ax[2].set_ylabel('coverage of mean +- 1.96 sd'); ax[2].set_title('(c) Gaussian interval, r=4, cv=0.6')
for j, cv in enumerate([0.1, 0.3, 0.6]):
    sub = A[(A.dist == 'normal') & (A.r == 4) & (A.cv == cv)].sort_values('d')
    ax[3].plot(sub.d, sub.var_lin / sub.var_exact, marker='o', color=PAL[j], label=f'cv={cv}')
ax[3].set_xscale('log'); ax[3].set_xlabel('d'); ax[3].set_ylabel('Var_linearised / Var_exact')
ax[3].set_title('(d) delta-method underestimates variance'); ax[3].legend(fontsize=8)
for a in ax[1:]:
    a.set_xticks([2, 5, 10, 20, 40]); a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(NullFormatter())
plt.tight_layout(); plt.savefig('figures/e1_moments.png'); print('saved')
