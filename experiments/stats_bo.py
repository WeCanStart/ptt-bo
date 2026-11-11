"""Robust statistics for a BO study (results/<study>/*.json):
 - per problem: median and IQM of final simple regret with 95% bootstrap CIs (over seeds)
 - paired Wilcoxon signed-rank tests (by seed) of a reference method against every other method
 - aggregate over problems: IQM of log10(regret / median regret of random search), stratified bootstrap CI,
   at the final budget and after 40 evaluations ("anytime" view)
Noisy studies (runs with 'y_noisy'): regret of the RECOMMENDED point = true f at the best noisy observation so far
(model-free, same rule for every method); the oracle regret (best true f among evaluated points) is also saved.
Study e13 also loads the unwarped baselines of e9 (same problems and seeds).
usage: python experiments/stats_bo.py STUDY [reference_method]"""
import json, glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, '.')
from pttbo.plot import plt, PAL

study = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else 'tt_eilf'
SOURCES = {'e13': [('e13', None), ('e9', {'random', 'cmaes', 'gpp_ei', 'gpa_ei', 'tt_eilf', 'bt_logei', 'bt_turbo'})]}
runs = []
for src, keep in SOURCES.get(study, [(study, None)]):
    for f in glob.glob(f'results/{src}/*.json'):
        r = json.load(open(f))
        if '__exactmin__' in f or (keep is not None and r['method'] not in keep):
            continue
        runs.append(r)
if not runs:
    sys.exit('no runs')
rng = np.random.default_rng(0)
noisy_study = any('y_noisy' in r for r in runs)


def iqm(x):
    x = np.sort(np.asarray(x)); n = len(x); lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def value_at(r, k):
    """true objective value credited to the run after k evaluations"""
    y = np.asarray(r['y'][:k])
    if 'y_noisy' in r:
        return float(y[int(np.argmin(r['y_noisy'][:k]))])
    return float(y.min())


rows = []
for p in sorted({r['problem'] for r in runs}):
    R = [r for r in runs if r['problem'] == p]
    seeds = sorted({r['seed'] for r in R})
    fstar = {}
    for s in seeds:
        known = [r.get('fopt') for r in R if r['seed'] == s and r.get('fopt') is not None]
        fstar[s] = known[0] if known else min(min(r['y']) for r in R if r['seed'] == s)
    for r in R:
        n = len(r['y'])
        rows.append(dict(problem=p, method=r['method'], seed=r['seed'],
                         regret=max(value_at(r, n) - fstar[r['seed']], 1e-10),
                         regret40=max(value_at(r, min(40, n)) - fstar[r['seed']], 1e-10),
                         regret100=max(value_at(r, min(100, n)) - fstar[r['seed']], 1e-10),
                         regret_oracle=max(min(r['y']) - fstar[r['seed']], 1e-10)))
D = pd.DataFrame(rows)
pd.set_option('display.width', 250)
LONG = max(len(r['y']) for r in runs) > 100           # budgets above 100: also report the state after 100 evaluations
COLS = {'regret': 'final', 'regret40': 'k40', **({'regret100': 'k100'} if LONG else {})}

tab = []
for (p, m), g in D.groupby(['problem', 'method']):
    x = g.regret.values
    boots = [iqm(rng.choice(x, len(x))) for _ in range(1000)]
    tab.append(dict(problem=p, method=m, n_seeds=len(x), median=float(np.median(x)), iqm=iqm(x),
                    iqm_lo=float(np.quantile(boots, 0.025)), iqm_hi=float(np.quantile(boots, 0.975)),
                    median_at40=float(np.median(g.regret40)), median_at100=float(np.median(g.regret100)),
                    median_oracle=float(np.median(g.regret_oracle))))
T = pd.DataFrame(tab)
T.to_csv(f'results/{study}_stats.csv', index=False)
print('== median final regret' + (' (recommended point = best noisy observation)' if noisy_study else ''))
print(T.pivot_table(index='method', columns='problem', values='median').round(3).to_string())
print('== median regret after 40 evaluations')
print(T.pivot_table(index='method', columns='problem', values='median_at40').round(3).to_string())

# paired tests vs reference
tests = []
for col in COLS:
    for p, g in D.groupby('problem'):
        a = g[g.method == ref].set_index('seed')[col]
        for m in sorted(g.method.unique()):
            if m == ref:
                continue
            b = g[g.method == m].set_index('seed')[col]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            x, y = a[common].values, b[common].values
            try:
                pval = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            except ValueError:
                pval = 1.0
            tests.append(dict(at=COLS[col], problem=p, other=m, n=len(common),
                              ref_wins=int(np.sum(x < y)), other_wins=int(np.sum(y < x)),
                              median_ratio=float(np.median(y / x)), p_wilcoxon=float(pval)))
W = pd.DataFrame(tests)
if len(W):
    W.to_csv(f'results/{study}_wilcoxon_vs_{ref}.csv', index=False)
    W['cell'] = W.apply(lambda r: f"{r.ref_wins}:{r.other_wins} p={r.p_wilcoxon:.3f}", axis=1)
    for at in COLS.values():
        print(f'== {ref} vs others ({at}): wins:losses and Wilcoxon p (paired by seed)')
        print(W[W['at'] == at].pivot_table(index='other', columns='problem', values='cell', aggfunc='first').to_string())

# pooled paired test vs reference over all (problem, seed) pairs, on log regret
pool = []
for col in COLS:
    a = D[D.method == ref].set_index(['problem', 'seed'])[col]
    for m in sorted(D.method.unique()):
        if m == ref:
            continue
        b = D[D.method == m].set_index(['problem', 'seed'])[col]
        c = a.index.intersection(b.index)
        if len(c) < 5:
            continue
        x, y = np.log10(a[c].values), np.log10(b[c].values)
        pool.append(dict(at=COLS[col], other=m, n=len(c), ref_wins=int(np.sum(x < y)), other_wins=int(np.sum(y < x)),
                         p_wilcoxon=float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0))
if pool:
    PL = pd.DataFrame(pool)
    PL.to_csv(f'results/{study}_pooled_vs_{ref}.csv', index=False)
    print(f'== {ref} vs others, pooled over problems and seeds')
    print(PL.to_string(index=False))

# aggregate normalised score
if 'random' in D.method.unique():
    agg = []
    for col in COLS:
        base = D[D.method == 'random'].groupby('problem')[col].median()
        D['score'] = np.log10(D[col] / D.problem.map(base))
        for m, g in D.groupby('method'):
            probs = sorted(g.problem.unique())
            est = iqm(g.score.values)
            boots = []
            for _ in range(1000):
                smp = np.concatenate([rng.choice(g[g.problem == p].score.values, (g.problem == p).sum()) for p in probs])
                boots.append(iqm(smp))
            agg.append(dict(at=COLS[col], method=m, iqm_score=est,
                            lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                            n_problems=len(probs)))
    A = pd.DataFrame(agg)
    A.to_csv(f'results/{study}_aggregate.csv', index=False)
    print('== aggregate IQM of log10(regret / median random regret); lower is better')
    print(A.pivot_table(index='method', columns='at', values='iqm_score').sort_values('final').round(3).to_string())
    panels = [a for a in ['k40', 'k100', 'final'] if a in COLS.values()]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 0.45 * A.method.nunique() + 1.2), sharey=True)
    order = A[A['at'] == 'final'].sort_values('iqm_score').method.tolist()
    titles = {'k40': 'after 40 evaluations', 'k100': 'after 100 evaluations', 'final': 'final budget'}
    for ax, at in zip(axs, panels):
        title = titles[at]
        B = A[A['at'] == at].set_index('method').loc[order]
        ys = np.arange(len(B))
        ax.errorbar(B.iqm_score, ys, xerr=[B.iqm_score - B.lo, B.hi - B.iqm_score], fmt='o', color=PAL[0], ms=6, lw=1.5)
        ax.set_yticks(ys); ax.set_yticklabels(B.index)
        ax.axvline(0, color='#8a8984', lw=1)
        ax.set_xlabel('IQM log10(regret / random), 95% CI'); ax.set_title(f'{study}: {title}')
    axs[0].invert_yaxis()
    plt.tight_layout(); plt.savefig(f'figures/{study}_aggregate.png')
