#!/usr/bin/env python3
"""
Do the inferential procedures in birth_seasonality_analysis.py behave as advertised?
(simulations behind the response to Reviewer 2, point 1)

Every block simulates series for which the answer is known and applies the procedure exactly as
it is applied to the data.

  A. Size of the sup-F test at the 5% level when there is NO shift: i.i.d. critical value,
     single bootstrap, double bootstrap (reported), and a bootstrap whose errors are estimated
     after removing the best-fitting shift. Simulated series come from simple autoregressions
     and from the null model fitted to each observed series.
  B. Coverage of the Bai (1997) 95% interval for the break year when one step is true.
  C. Size of the bootstrap test of an instantaneous step against a logistic transition when the
     step is true.
  D. Coverage of plain percentile bootstrap 95% intervals for the mid-point and duration of the
     logistic transition when the fitted transition is true. They prove too narrow, which is why
     the reported intervals are calibrated by a second bootstrap level. (The calibrated intervals
     cover the truth in 95% of series simulated from the fitted model by construction; checking
     them at other models would need a third level of simulation and has not been done.)

Run birth_seasonality_analysis.py first (this script reads results/annual_metrics.csv).
Usage:  python validation_simulations.py [results_dir] [processes]
Takes about 35 minutes of processor time (about 20 minutes of clock time on two processors).
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import birth_seasonality_analysis as bsa

RES = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
NPROC = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEED = 20260037
N_SIZE, B_SIZE = 500, 199          # block A: series per row; bootstrap replicates at each level
N_BAI = 2000                       # block B
N_LOG, B_LOG = 1000, 499           # blocks C and D
COUNTRIES = ('Germany', 'France')
METRICS = (('peak_month', 'peak month'), ('phase', 'acrophase'), ('amplitude', 'amplitude'))


def observed(country, metric):
    am = pd.read_csv(RES / 'annual_metrics.csv').set_index(['country', 'year'])
    return am.loc[country][metric].astype(float)


def step_model(y):
    """Best-fitting single step and the autoregression fitted to what is left."""
    _, k = bsa.sup_f(y)
    step = np.where(np.arange(len(y)) <= k, y[:k + 1].mean(), y[k + 1:].mean())
    return k, step, bsa._select_ar(y - step)


# ----------------------------------------------------------------------------- block A
def size_row(task):
    key, label, n, phi, res = task
    rng = np.random.default_rng([SEED, 1, key])
    rej = dict(iid=0, single=0, double=0, adjusted=0)
    for _ in range(N_SIZE):
        y = (lfilter([1.0], np.r_[1.0, -phi], rng.standard_normal(n + 200))[200:] if res is None
             else bsa._ar_draw(phi, res, n, rng))
        p_single, p_double, _ = bsa.bootstrap_supf(y, rng, reps=B_SIZE, inner=B_SIZE)
        p_adj, _, _ = bsa.bootstrap_supf(y, rng, reps=B_SIZE, inner=0, residuals='break-adjusted')
        rej['iid'] += bsa.sup_f(y)[0] > bsa.ANDREWS_CRIT[0.05]
        rej['single'] += p_single <= 0.05
        rej['double'] += p_double <= 0.05
        rej['adjusted'] += p_adj <= 0.05
    pc = {k: v / N_SIZE * 100 for k, v in rej.items()}
    return ('A', key, f'   {label}: i.i.d. critical value {pc["iid"]:.1f}%; single bootstrap {pc["single"]:.1f}%; '
                      f'double bootstrap (reported) {pc["double"]:.1f}%; errors estimated after removing the '
                      f'best shift {pc["adjusted"]:.1f}%')


# ----------------------------------------------------------------------------- block B
def bai_row(task):
    key, country, metric, name = task
    rng = np.random.default_rng([SEED, 2, key])
    y = observed(country, metric).values
    n = len(y)
    k, step, (p, phi, res) = step_model(y)
    cover = exact = 0
    width = []
    for _ in range(N_BAI):
        ys = step + bsa._ar_draw(phi, res, n, rng)
        if metric == 'peak_month':
            ys = np.clip(np.round(ys), 1, 12)             # the peak month is a whole number
        _, kb = bsa.sup_f(ys)
        lo, hi = bsa.bai_ci(ys, kb)
        cover += lo <= k <= hi
        exact += kb == k
        width.append(hi - lo + 1)
    return ('B', key, f'   {country} {name} (step {y[:k + 1].mean():.2f} -> {y[k + 1:].mean():.2f}, AR({p}) errors): '
                      f'interval covers the true break year in {cover / N_BAI * 100:.1f}% of series; median width '
                      f'{np.median(width):.0f} years; break year estimated exactly in {exact / N_BAI * 100:.0f}%')


# ----------------------------------------------------------------------------- blocks C and D
def step_row(task):
    key, country = task
    rng = np.random.default_rng([SEED, 3, key])
    s = observed(country, 'phase')
    y = s.values
    k, step, (p, phi, res) = step_model(y)
    pv, lr = [], []
    for _ in range(N_LOG):
        r = bsa.step_vs_logistic(pd.Series(step + bsa._ar_draw(phi, res, len(y), rng), index=s.index), rng,
                                 reps=B_LOG)
        pv.append(r['p'])
        lr.append(r['lr'])
    pv, lr = np.array(pv), np.array(lr)
    obs = len(y) * np.log(((y - step) ** 2).sum() / bsa.logistic_fit(s.index.values, y)[0]['rss'])
    return ('C', key, f'   {country} (one step with AR({p}) errors, as fitted): the test rejects the step at the 5% '
                      f'level in {np.mean(pv <= 0.05) * 100:.1f}% of series; largest likelihood ratio in '
                      f'{N_LOG:,} such series {lr.max():.1f} (observed in the data: {obs:.1f})')


def logistic_row(task):
    key, country = task
    rng = np.random.default_rng([SEED, 4, key])
    s = observed(country, 'phase')
    t, y = s.index.values.astype(float), s.values
    fit, _ = bsa.logistic_fit(t, y)
    e = y - fit['fitted']
    p, phi, res = bsa._select_ar(e - e.mean())
    cm = cd = 0
    for _ in range(N_LOG):
        f = bsa.logistic_with_ci(pd.Series(fit['fitted'] + bsa._ar_draw(phi, res, len(y), rng), index=s.index),
                                 rng, reps=B_LOG, outer=0)
        cm += f['mid_ci'][0] <= fit['mid'] <= f['mid_ci'][1]
        cd += f['dur_ci'][0] <= fit['duration'] <= f['dur_ci'][1]
    return ('D', key, f'   {country} (logistic transition with AR({p}) errors, as fitted): the percentile interval '
                      f'covers the true mid-point in {cm / N_LOG * 100:.1f}% and the true duration in '
                      f'{cd / N_LOG * 100:.1f}% of series')


def run_task(job):
    fn, task = job
    return fn(task)


def main():
    t0 = time.time()
    jobs = []
    for i, (n, rho) in enumerate(((73, 0.0), (77, 0.6), (73, 0.9))):
        jobs.append((size_row, (i, f'AR(1) coefficient {rho}, n = {n}', n,
                                np.array([rho]) if rho else np.zeros(0), None)))
    i = 3
    for c in COUNTRIES:
        for m, name in METRICS:
            y = observed(c, m).values
            p, phi, res = bsa._null_fit(y)
            jobs.append((size_row, (i, f'no-shift model fitted to {c} {name} (AR({p}), coefficients sum to '
                                       f'{phi.sum():.2f})', len(y), phi, res)))
            jobs.append((bai_row, (i, c, m, name)))
            i += 1
    for j, c in enumerate(COUNTRIES):
        jobs.append((step_row, (j, c)))
        jobs.append((logistic_row, (j, c)))
    jobs.sort(key=lambda job: job[0] in (size_row, bai_row))      # longest jobs first
    with Pool(NPROC) as pool:
        out = pool.map(run_task, jobs, chunksize=1)
    rows = {blk: [line for b, _, line in sorted(out, key=lambda r: (r[0], r[1])) if b == blk] for blk in 'ABCD'}
    lines = ['VALIDATION OF THE INFERENTIAL PROCEDURES BY SIMULATION',
             '',
             f'A. HOW OFTEN IS A SHIFT "FOUND" AT THE 5% LEVEL WHEN THERE IS NONE? ({N_SIZE} series per row; '
             f'{B_SIZE} bootstrap replicates at each level)', *rows['A'],
             '',
             f'B. COVERAGE OF THE 95% INTERVAL FOR THE BREAK YEAR ({N_BAI:,} series per row)', *rows['B'],
             '',
             f'C. TEST OF AN INSTANTANEOUS STEP AGAINST A GRADUAL TRANSITION, WHEN THE STEP IS TRUE '
             f'({N_LOG:,} series per row; {B_LOG} bootstrap replicates)', *rows['C'],
             '',
             f'D. COVERAGE OF PLAIN PERCENTILE 95% INTERVALS FOR THE LOGISTIC TRANSITION ({N_LOG:,} series per '
             f'row; {B_LOG} bootstrap replicates) - the reported intervals are calibrated instead', *rows['D'],
             '']
    print('\n'.join(lines))
    print(f'[{(time.time() - t0) / 60:.0f} minutes]')
    (RES / 'validation_simulations.txt').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
