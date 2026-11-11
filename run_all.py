#!/usr/bin/env python
"""Запуск экспериментов на своём компьютере, параллельно, с логами и докачкой.

  python run_all.py --check               # быстрая проверка окружения (~1-2 мин)
  python run_all.py --check-botorch       # проверка BoTorch-бейзлайнов (после pip install botorch)
  python run_all.py --workers 4           # основной набор E1-E8 (часы; см. README)
  python run_all.py --workers 6 --only new   # дополнительные эксперименты E3+, E4b, E9-E12
  python run_all.py --workers 6 --only new2  # E13 (почему PTT застревает: warp / trust region) и E11b (шум)
  python run_all.py --workers 10 --only new3 # E14 (проверка TR-PTT на отложенных задачах) + TR-PTT в E11b
  python run_all.py --workers 4 --only e7,e8
  python run_all.py --dry-run --workers 4 # только показать список задач
  python run_all.py --plots               # только перерисовать рисунки/таблицы по готовым results/

Каждая задача - отдельный процесс python с 1 потоком BLAS; workers = сколько процессов одновременно
(разумно: число ФИЗИЧЕСКИХ ядер минус 1). BO-прогоны (E7, E8) докачиваются: уже посчитанные
results/e7/*.json и results/e8/*.json пропускаются, поэтому прерванный запуск можно просто повторить.
"""
import argparse, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
THREAD_VARS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS']

E7_METHODS = 'random,cmaes,gp_ei,gp_ts,gpp_ei,gpp_ts,gpa_ei,tt_mean,tt_ts,tt_rmts,tt_eilf'
E8_METHODS = 'random,sa,protes,hgp_ts,hgp_ei,tt_mean,tt_ts,tt_rmts,tt_eilf'


def child_env():
    e = os.environ.copy()
    e['PYTHONPATH'] = ROOT + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    for v in THREAD_VARS:
        e[v] = '1'
    e['PYTHONIOENCODING'] = 'utf-8'
    e['MKL_THREADING_LAYER'] = 'SEQUENTIAL'  # 1 thread anyway; avoids a 2nd OpenMP DLL next to torch on Windows
    e['MPLBACKEND'] = 'Agg'
    return e


def build_jobs():
    """(stage, name, script+args). Long single-process jobs first, then the many small BO jobs."""
    J = [('e4', 'e4e5_all', ['experiments/e4_e5_posterior.py', 'all']),
         ('e3', 'e3_ranks', ['experiments/e3_ranks.py']),
         ('e4', 'e4_rms', ['experiments/e4_e5_posterior.py', 'e4', 'rms,ens', 'rms']),
         ('e2', 'e2_prior', ['experiments/e2_prior.py']),
         ('e1', 'e1_moments', ['experiments/e1_moments.py'])]
    J += [('e6', f'e6_{i:02d}', ['experiments/e6_surrogate.py', str(i), '60']) for i in range(60)]
    J += [('e6', f'e6b_{i:02d}', ['experiments/e6b_ablation.py', str(i), '60']) for i in range(60)]
    # one BO run per process (perfect load balancing); part 0 of E8 also does the 5000-eval references
    J += [('e8', 'e8_000', ['experiments/e8_discrete.py', '0', '150', E8_METHODS])]
    J += [('e7', f'e7_{i:03d}', ['experiments/e7_bo.py', str(i), '300', E7_METHODS]) for i in range(300)]
    J += [('e8', f'e8_{i:03d}', ['experiments/e8_discrete.py', str(i), '150', E8_METHODS]) for i in range(1, 150)]
    # ---- additional studies (stage group "new")
    J += [('e12', 'e12_scaling', ['experiments/e12_scaling.py'])]
    J += [('e4b', f'e4b_{i:02d}', ['experiments/e4b_posterior_hd.py', str(i), '24']) for i in range(24)]
    J += [('e9', f'e9_{i:03d}', ['experiments/e9_bo_extra.py', 'e9', str(i), '770']) for i in range(770)]
    J += [('e10', f'e10_{i:02d}', ['experiments/e9_bo_extra.py', 'e10', str(i), '72']) for i in range(72)]
    J += [('e11', f'e11_{i:02d}', ['experiments/e9_bo_extra.py', 'e11', str(i), '60']) for i in range(60)]
    # ---- stage group "new2": why PTT stalls (E13) and the redesigned noisy study (E11b)
    J += [('e13', f'e13_{i:03d}', ['experiments/e9_bo_extra.py', 'e13', str(i), '660']) for i in range(660)]
    # ---- stage group "new3": held-out validation (E14, 780 runs, long d=20 jobs first) + TR-PTT under noise (E11b)
    J += [('e14', f'e14_{i:03d}', ['experiments/e9_bo_extra.py', 'e14', str(i), '780']) for i in range(780)]
    J += [('e11b', f'e11b_{i:03d}', ['experiments/e9_bo_extra.py', 'e11b', str(i), '320']) for i in range(320)]
    return J


PLOTS = {'e1': [['experiments/plot_e1.py']], 'e4': [['experiments/plot_e4_e5.py']],
         'e6': [['experiments/plot_e6.py'], ['experiments/tables.py']], 'e3': [['experiments/tables.py']],
         'e7': [['experiments/plot_bo.py', 'e7']], 'e8': [['experiments/plot_bo.py', 'e8']],
         'e4b': [['experiments/table_e4b.py']],
         'e9': [['experiments/plot_bo.py', 'e9'], ['experiments/stats_bo.py', 'e9', 'tt_eilf']],
         'e10': [['experiments/plot_bo.py', 'e10'], ['experiments/stats_bo.py', 'e10', 'tt_eilf']],
         'e11': [['experiments/plot_bo.py', 'e11'], ['experiments/stats_bo.py', 'e11', 'tt_eilf']],
         'e13': [['experiments/plot_bo.py', 'e13'], ['experiments/stats_bo.py', 'e13', 'tt_eilf'],
                 ['experiments/stats_bo.py', 'e13', 'bt_turbo']],
         'e11b': [['experiments/plot_bo.py', 'e11b'], ['experiments/stats_bo.py', 'e11b', 'tt_eilf'],
                  ['experiments/stats_bo.py', 'e11b', 'tt_tr']],
         'e14': [['experiments/plot_bo.py', 'e14'], ['experiments/stats_bo.py', 'e14', 'tt_tr'],
                 ['experiments/stats_bo.py', 'e14', 'bt_turbo']]}
NEW = {'e3', 'e4b', 'e9', 'e10', 'e11', 'e12'}
NEW2 = {'e13', 'e11b'}
NEW3 = {'e14', 'e11b'}


def run_one(name, args, logdir):
    t = time.time()
    log = os.path.join(logdir, name + '.log')
    with open(log, 'w', encoding='utf-8') as fh:
        p = subprocess.run([sys.executable] + args, cwd=ROOT, env=child_env(), stdout=fh, stderr=subprocess.STDOUT)
    return name, p.returncode, time.time() - t, log


def run_plots(logdir, stages=None):
    print('\n== рисунки и таблицы ==', flush=True)
    todo = []
    for st, lst in PLOTS.items():
        if stages is None or st in stages:
            todo += [a for a in lst if a not in todo]
    for args in todo:
        name = '_'.join(os.path.basename(a).replace('.py', '') for a in args)
        n, rc, dt, log = run_one(name, args, logdir)
        print(f'  {"ok  " if rc == 0 else "FAIL"} {name} ({dt:.0f} s)' + ('' if rc == 0 else f'  -> см. {log}'), flush=True)


def check():
    for v in THREAD_VARS:
        os.environ[v] = '1'
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import platform, warnings
    warnings.filterwarnings('ignore')
    import numpy, scipy, sklearn, pandas, matplotlib, cma, teneva
    print('python', platform.python_version(), platform.platform())
    print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| sklearn', sklearn.__version__,
          '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__, '| cma', cma.__version__,
          '| teneva', getattr(teneva, '__version__', '?'))
    print('cpu_count (логические ядра):', os.cpu_count())
    from pttbo.tt import FourierBasis, PTT
    from pttbo.infer import PTTRegressor
    from pttbo.functions import get_problem
    from pttbo.bo import run_bo, run_bo_discrete
    rng = numpy.random.default_rng(0)
    # 1) exact moments vs Monte Carlo
    B = [FourierBasis(3, 0.3) for _ in range(3)]
    m = PTTRegressor(B, rank=2, prior='anova')
    ptt = PTT([W.copy() for W in m.W0], [v * 4 for v in m.v0], B)
    X = rng.random((2, 3))
    mu, var = ptt.moments(X)
    S = numpy.stack([PTT.eval_cores(c, ptt.features(X)) for c in ptt.sample_cores(rng, 20000)])
    print(f'moments: exact var {var.round(4)}  MC var {S.var(0).round(4)}')
    # 2) tiny BO runs
    p = get_problem('branin')
    for meth in ['gp_ei', 'gpp_ei', 'cmaes', 'tt_eilf']:
        t = time.time(); r = run_bo(p, meth, 16, 10, 0)
        print(f'BO {meth:8s} branin best={r["y"].min():.3f}  ({time.time() - t:.1f} s)')
    Q = rng.standard_normal((8, 8))
    f = lambda Z: numpy.einsum('ni,ij,nj->n', Z, Q, Z)
    for meth in ['hgp_ei', 'protes', 'tt_eilf']:
        t = time.time(); r = run_bo_discrete(f, 8, 2, meth, 24, 10, 0)
        print(f'BO {meth:8s} bqp8   best={r["y"].min():.3f}  ({time.time() - t:.1f} s)')
    Y = teneva.rand([4] * 4, 2, seed=0)
    print('teneva ok, ranks', teneva.ranks(Y))
    print('\nПРОВЕРКА ПРОЙДЕНА — можно запускать: python run_all.py --workers N')


def check_botorch():
    for v in THREAD_VARS:
        os.environ[v] = '1'
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import warnings
    warnings.filterwarnings('ignore')
    import torch, botorch, gpytorch
    print('torch', torch.__version__, '| botorch', botorch.__version__, '| gpytorch', gpytorch.__version__)
    from pttbo.functions import get_problem
    from pttbo.bo_botorch import run_botorch
    p = get_problem('hartmann6')
    for meth in ['bt_logei', 'bt_turbo']:
        t = time.time(); r = run_botorch(p, meth, 30, 10, 0)
        print(f'{meth:9s} hartmann6, 30 вызовов: best={r["y"].min():.3f} (оптимум -3.322), {time.time() - t:.0f} s')
    import numpy as np
    from pttbo.bo_botorch import run_logei, run_turbo
    f = lambda U: p(np.asarray(U)) + 0.03 * np.random.default_rng(0).standard_normal(len(U))
    for name, fn in [('bt_lognei', lambda: run_logei(f, 6, 26, 10, 0, noisy_acq=True)),
                     ('bt_logei_warp', lambda: run_logei(lambda U: p(np.asarray(U)), 6, 26, 10, 0, warp=True)),
                     ('bt_turbo noisy+warp', lambda: run_turbo(f, 6, 26, 10, 0, warp=True, noisy=True))]:
        t = time.time(); r = fn()
        print(f'{name:20s} hartmann6, 26 вызовов: best={np.min(r["y"]):.3f}, {time.time() - t:.0f} s')
    print('\nBoTorch РАБОТАЕТ — можно запускать: python run_all.py --workers N --only new2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    ap.add_argument('--only', default='', help='стадии через запятую: e1,e2,e3,e4,e6,e7,e8')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--check-botorch', action='store_true')
    ap.add_argument('--skip-botorch', action='store_true', help='не запускать BoTorch-бейзлайны в E9/E11')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--plots', action='store_true', help='только рисунки/таблицы')
    ap.add_argument('--fresh', action='store_true', help='перенести старые results/ в results_prev_<время>/')
    a = ap.parse_args()
    if a.check:
        return check()
    if a.check_botorch:
        return check_botorch()
    if a.skip_botorch:
        os.environ['TTGP_SKIP_BOTORCH'] = '1'
    logdir = os.path.join(ROOT, 'results', 'logs')
    if a.fresh and os.path.isdir(os.path.join(ROOT, 'results')):
        dst = os.path.join(ROOT, 'results_prev_' + time.strftime('%Y%m%d_%H%M%S'))
        shutil.move(os.path.join(ROOT, 'results'), dst)
        print('старые результаты перенесены в', dst)
    for d in ['results', 'results/e7', 'results/e8', 'results/e9', 'results/e10', 'results/e11', 'results/e13',
              'results/e11b', 'results/e14', 'figures', 'results/logs']:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    st = None
    if a.only:
        st = set(s.strip() for s in a.only.split(','))
        if 'new3' in st:
            st = (st - {'new3'}) | NEW3
        if 'new2' in st:
            st = (st - {'new2'}) | NEW2
        if 'new' in st:
            st = (st - {'new'}) | NEW
    if a.plots:
        return run_plots(logdir, st)
    jobs = build_jobs()
    if st is not None:
        jobs = [j for j in jobs if j[0] in st]
    print(f'задач: {len(jobs)}, параллельно: {a.workers}, логи: {logdir}', flush=True)
    if a.dry_run:
        for s, n, args in jobs:
            print(f'  [{s}] {n}: python {" ".join(args)}')
        return
    t0 = time.time(); fails = []; done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, n, args, logdir) for s, n, args in jobs]
        try:
            for fu in as_completed(futs):
                name, rc, dt, log = fu.result(); done += 1
                if rc != 0:
                    fails.append((name, log))
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f'[{done}/{len(jobs)}] {"ok  " if rc == 0 else "FAIL"} {name:12s} {dt:6.0f} s | '
                      f'прошло {el / 3600:.2f} ч, осталось ~{eta / 3600:.2f} ч', flush=True)
        except KeyboardInterrupt:
            print('\nостановлено пользователем; посчитанное сохранено, повторный запуск докачает E7/E8')
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    run_plots(logdir, st)
    print(f'\nготово за {(time.time() - t0) / 3600:.2f} ч; ошибок: {len(fails)}')
    for n, log in fails:
        print('  ошибка:', n, '->', log)


if __name__ == '__main__':
    main()
