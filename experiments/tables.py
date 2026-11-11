"""Tables for E3 (TT ranks) and E6b (additive GP / GP with priors / TT rank ablation)."""
import json, glob, os, sys
import pandas as pd
sys.path.insert(0, '.')
pd.set_option('display.width', 220)

if os.path.exists('results/e3_ranks.json'):
    E3 = pd.DataFrame(json.load(open('results/e3_ranks.json')))
    E3['lower_bound'] = E3['cross_err'] > 1e-2  # TT-cross hit the rank cap: rank is a lower bound
    E3.to_csv('results/e3_table.csv', index=False)
    print('E3: TT ranks\n', E3[['problem', 'd', 'n', 'how', 'rank_eps0.01', 'rank_eps1e-06', 'cross_err', 'lower_bound']].to_string(index=False))

rows = []
for f in glob.glob('results/e6b_part*.json'):
    rows += json.load(open(f))
for f in glob.glob('results/e6_part*.json'):
    rows += [dict(r, method='tt_anova_map_r4' if r['method'] == 'tt_anova_map' else r['method'])
             for r in json.load(open(f)) if r['method'] in ('gp_matern', 'tt_anova_map')]
if rows:
    A = pd.DataFrame(rows)
    P = A.pivot_table(index=['problem', 'n'], columns='method', values='nrmse', aggfunc='mean').round(3)
    cols = [c for c in ['gp_matern', 'gp_prior', 'gp_additive', 'tt_anova_map_r1', 'tt_anova_map_r2',
                        'tt_anova_map_r4', 'tt_anova_map_r8'] if c in P.columns]
    P[cols].to_csv('results/e6b_table.csv')
    print('\nE6b: nRMSE\n', P[cols].to_string())
