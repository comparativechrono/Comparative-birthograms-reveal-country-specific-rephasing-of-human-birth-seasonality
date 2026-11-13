#!/usr/bin/env python3
"""
Revised analysis for
  "Comparative birthograms reveal country-specific rephasing of human birth seasonality"
  (Hearn & Lloyd-Owen; revised analysis, version 2.0.0)

What this script does
---------------------
1. Reads monthly live births for France (INSEE 000436391, 1946-2024) and Germany
   (Destatis 12612-0002, 1950-2024; former FRG and former GDR combined in all years).
2. Converts counts to births per day (month lengths differ by up to 10%) and expresses each
   month relative to a centred 12-month moving average, so that seasonal indices are relative
   to the current level of births and are NOT smoothed across years.
3. For every calendar year computes
      - phase:      acrophase of a single-harmonic (cosinor) fit to the 12 monthly indices,
      - amplitude:  peak-to-trough range of the 12 monthly indices, in % of trend,
      - peak month: the month with the highest index.
4. Tests each annual series for a shift in mean at an unknown date with the sup-F
   (Quandt-Andrews) statistic. P-values come from a double (calibrated) sieve bootstrap under
   the null hypothesis of no shift, which allows for serial correlation: a single bootstrap
   rejects a true null too often when the series is strongly persistent, and the second level
   corrects for this (see validation_simulations.py). The i.i.d. asymptotic critical values of Andrews
   (1993) are reported for reference only. The break year is reported with a Bai (1997) 95%
   confidence interval based on regime-specific long-run variances. For the peak month that
   interval is widened, where necessary, to the range of break years obtained when the monthly
   counts are redrawn as Poisson variates, because in some years the highest month leads the
   next by less than the counting noise.
5. Fits a logistic transition to each phase series to estimate its mid-point and how long the
   central 80% of the shift took (double-bootstrap calibrated confidence intervals), tests an
   instantaneous step against that gradual transition, and reports how many successive shifts
   information criteria select (descriptive only).

Changes from the archived v1.0.0 notebooks (Zenodo 10.5281/zenodo.19066197)
------------------------------------------------------------------------
* births per day instead of monthly counts;
* amplitude relative to trend instead of absolute births (absolute amplitude tracks the
  number of births: see reviewer_checks.py);
* seasonal indices not smoothed across years (the STL seasonal component averages ~7 years);
* peak month and acrophase are analysed directly, replacing the births-on-photoperiod regression;
* scanned Chow p-values from the F distribution replaced by valid sup-F inference, with
  confidence intervals for break years.

Usage:  python birth_seasonality_analysis.py [data_dir] [results_dir] [processes]
Takes about 20 minutes of processor time (the two countries run in parallel by default).
"""
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.stats import norm

# ----------------------------------------------------------------------------- settings
DAY_ADJUST = True      # births per day (True) or monthly counts as submitted (False)
TRIM = 0.15            # fraction of the sample excluded at each end as candidate break dates
N_BOOT = 9999          # bootstrap replicates
N_INNER = 999          # second-level replicates per first-level series (double bootstrap)
N_OUTER = 1999         # first-level series used to calibrate the logistic-transition intervals
N_INNER_LOGISTIC = 499 # second-level replicates for each of them
N_POISSON = 1999       # Poisson replicates of the monthly counts (how firmly is each year's peak month decided?)
SEED = 20260037
FILES = {'Germany': 'german data.csv', 'France': 'french data.csv'}

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
MON3 = [m[:3] for m in MONTHS]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}


# ============================================================================ 1. data
def load_births(path):
    """Headerless CSV: year, month name, live births, (fourth column ignored)."""
    df = pd.read_csv(path, header=None, names=['year', 'month_name', 'births', 'extra'])
    df['month'] = df['month_name'].map(MONTH_NUM)
    df['date'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=1))
    df = df.sort_values('date').set_index('date')
    df['days'] = [calendar.monthrange(y, m)[1] for y, m in zip(df['year'], df['month'])]
    df['rate'] = df['births'] / df['days']
    return df[['year', 'month', 'births', 'days', 'rate']]


def centred_ma(x, period=12):
    """Centred 2 x 12 moving average; undefined for the first and last six months."""
    return x.rolling(period, center=True).mean().shift(-1).rolling(2).mean()


def seasonal_index(df, day_adjust=DAY_ADJUST, extrapolate_ends=False):
    """log(births per day) minus its centred 12-month moving average.

    extrapolate_ends=True extends the trend linearly over the first and last six months
    (used for display only; inference uses complete years)."""
    v = np.log(df['rate' if day_adjust else 'births'].astype(float))
    trend = centred_ma(v)
    if extrapolate_ends:
        t = np.arange(len(trend), dtype=float)
        ok = trend.notna().values
        tr = trend.values.copy()
        head = np.polyfit(t[ok][:12], tr[ok][:12], 1)
        tail = np.polyfit(t[ok][-12:], tr[ok][-12:], 1)
        tr[:6] = np.polyval(head, t[:6])
        tr[-6:] = np.polyval(tail, t[-6:])
        trend = pd.Series(tr, index=trend.index)
    return v - trend


def index_table(df, **kw):
    """Year x month table of seasonal indices (log units)."""
    si = seasonal_index(df, **kw)
    return pd.DataFrame({'year': df['year'], 'month': df['month'], 'si': si}).pivot(
        index='year', columns='month', values='si')


# ============================================================================ 2. annual metrics
def month_mid_fraction(year, month):
    """Mid-point of a calendar month as a fraction of that year."""
    ylen = 366 if calendar.isleap(year) else 365
    start = sum(calendar.monthrange(year, m)[1] for m in range(1, month))
    return (start + calendar.monthrange(year, month)[1] / 2.0) / ylen


def cosinor(si12, year):
    """Single-harmonic least-squares fit. Returns (amplitude, acrophase in months since 1 Jan)."""
    t = np.array([month_mid_fraction(year, m) for m in range(1, 13)])
    X = np.column_stack([np.ones(12), np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(si12, float), rcond=None)
    amp = float(np.hypot(beta[1], beta[2]))
    phase = (np.arctan2(beta[2], beta[1]) / (2 * np.pi)) % 1.0
    return amp, phase * 12.0


def annual_metrics(df, **kw):
    """Per-calendar-year phase, amplitude and peak month (complete years only)."""
    tab = index_table(df, **kw).dropna()
    rows = []
    for yr, r in tab.iterrows():
        amp, ph = cosinor(r.values, yr)
        rows.append(dict(year=int(yr),
                         phase=ph,                                       # months since 1 Jan
                         amplitude=(np.exp(r.max()) - np.exp(r.min())) * 100,  # (peak - trough) / trend, %
                         amplitude_h1=(np.exp(amp) - np.exp(-amp)) * 100,       # same, for the fitted harmonic
                         peak_month=int(r.idxmax()), trough_month=int(r.idxmin())))
    return pd.DataFrame(rows).set_index('year'), tab


def phase_to_date(phase_months):
    """Months since 1 January -> 'day Mon' in a 365-day year."""
    doy = int(round(phase_months / 12.0 * 365)) % 365
    d = pd.Timestamp('2001-01-01') + pd.Timedelta(days=doy)
    return f"{d.day} {d.strftime('%b')}"


# ============================================================================ 3. sup-F test
def f_profile(y, trim=TRIM):
    """F statistic for a shift in mean after each candidate observation k
    (regime 1 = y[:k+1], regime 2 = y[k+1:])."""
    y = np.asarray(y, float)
    n = len(y)
    h = max(int(np.ceil(trim * n)), 2)
    cs, cs2 = np.cumsum(y), np.cumsum(y ** 2)
    rss0 = cs2[-1] - cs[-1] ** 2 / n
    ks = np.arange(h - 1, n - h)
    n1 = ks + 1
    rss = (cs2[ks] - cs[ks] ** 2 / n1) + ((cs2[-1] - cs2[ks]) - (cs[-1] - cs[ks]) ** 2 / (n - n1))
    return (rss0 - rss) / (rss / (n - 2)), ks


def sup_f(y, trim=TRIM):
    F, ks = f_profile(y, trim)
    j = int(np.argmax(F))
    return float(F[j]), int(ks[j])


_NULL = {}


def andrews_null(trim=TRIM, reps=200000, steps=1000):
    """Simulated limiting distribution of sup-F for one shifting parameter (Andrews 1993):
    sup over [trim, 1-trim] of B(l)^2 / (l (1 - l)), B a Brownian bridge."""
    if trim not in _NULL:
        rng = np.random.default_rng(12345)
        lam = np.arange(1, steps) / steps
        keep = (lam >= trim - 1e-12) & (lam <= 1 - trim + 1e-12)
        out = np.empty(reps)
        for s in range(0, reps, 5000):
            m = min(5000, reps - s)
            w = np.cumsum(rng.standard_normal((m, steps)) / np.sqrt(steps), axis=1)
            b = w[:, :-1] - lam * w[:, -1:]
            out[s:s + m] = (b[:, keep] ** 2 / (lam[keep] * (1 - lam[keep]))).max(axis=1)
        _NULL[trim] = np.sort(out)
    return _NULL[trim]


def andrews_pvalue(stat, trim=TRIM):
    null = andrews_null(trim)
    return max(1.0 - np.searchsorted(null, stat) / len(null), 1.0 / len(null))


# Published asymptotic critical values, one parameter, 15% trimming (Andrews 2003, Table 1)
ANDREWS_CRIT = {0.10: 7.12, 0.05: 8.68, 0.01: 12.16}


def sup_f_rows(Y, trim=TRIM):
    """sup-F statistic for every row of Y (one series per row)."""
    Y = np.atleast_2d(np.asarray(Y, float))
    n = Y.shape[1]
    h = max(int(np.ceil(trim * n)), 2)
    cs, cs2 = np.cumsum(Y, axis=1), np.cumsum(Y ** 2, axis=1)
    ks = np.arange(h - 1, n - h)
    n1 = ks + 1
    rss0 = cs2[:, -1:] - cs[:, -1:] ** 2 / n
    rss = (cs2[:, ks] - cs[:, ks] ** 2 / n1) + ((cs2[:, -1:] - cs2[:, ks])
                                                 - (cs[:, -1:] - cs[:, ks]) ** 2 / (n - n1))
    return ((rss0 - rss) / (rss / (n - 2))).max(axis=1)


def _select_ar(e, pmax=3):
    """AR(p) for a zero-mean series, p <= pmax chosen by BIC on a common sample. A fitted
    autoregression that is not stationary is shrunk until its largest root is 0.999."""
    best = None
    for p in range(pmax + 1):
        if p == 0:
            phi, res = np.zeros(0), e - e.mean()
        else:
            Y = e[p:]
            X = np.column_stack([e[p - j - 1:len(e) - j - 1] for j in range(p)])
            phi, *_ = np.linalg.lstsq(X, Y, rcond=None)
            res = Y - X @ phi
        rc = res[(pmax - p):]
        bic = len(rc) * np.log(np.mean(rc ** 2)) + p * np.log(len(rc))
        if best is None or bic < best[0]:
            best = (bic, p, phi, res)
    _, p, phi, res = best
    if p:
        big = np.abs(np.roots(np.r_[1.0, -phi])).max()
        if big > 0.999:
            phi = phi * (0.999 / big) ** np.arange(1, p + 1)
    return p, phi, res - res.mean()


def _ar_draw(phi, res, n, rng, burn=200):
    return lfilter([1.0], np.r_[1.0, -phi], rng.choice(res, n + burn))[burn:]


def _ar_draws(phi, res, n, rng, size, burn=200):
    """size series of length n (one per row) from the fitted autoregression."""
    return lfilter([1.0], np.r_[1.0, -phi], rng.choice(res, (size, n + burn)), axis=1)[:, burn:]


def _null_fit(y, k=None):
    """Autoregression fitted to the demeaned series (H0: no shift) or, if k is given, to the
    series after removing a shift after observation k (used for comparison only)."""
    y = np.asarray(y, float)
    e = y - y.mean()
    if k is not None:
        e = np.where(np.arange(len(y)) <= k, y - y[:k + 1].mean(), y - y[k + 1:].mean())
    return _select_ar(e)


def bootstrap_supf(y, rng, trim=TRIM, reps=N_BOOT, inner=N_INNER, residuals='null'):
    """Sieve-bootstrap p-values for sup-F under H0 (constant mean, autoregressive errors).

    Returns (p_single, p_double, ar_order).

    p_single: an autoregression (order <= 3 by BIC) is fitted to the demeaned series, `reps`
    series are drawn from it and sup-F is recomputed for each.
    p_double (reported): the single bootstrap rejects a true null too often when the series is
    strongly persistent, because autoregressive coefficients are underestimated in short
    series. Each first-level series is therefore treated in turn as if it were the data: its own
    bootstrap p-value is computed from `inner` second-level series, and p_double is the share of
    first-level series whose p-value is at least as small as p_single (Beran 1988). P-values
    below the resolution of the second level, 1 / (inner + 1), are compared at that resolution,
    which is conservative. inner=0 skips the second level and returns p_double = nan.

    residuals='break-adjusted' fits the autoregression after removing the best-fitting shift.
    That understates persistence when there is no shift and rejects a true null far too often;
    it is kept for comparison only (single level)."""
    y = np.asarray(y, float)
    n = len(y)
    stat, k = sup_f(y, trim)
    p, phi, res = _null_fit(y, k if residuals == 'break-adjusted' else None)
    Y1 = _ar_draws(phi, res, n, rng, reps)
    s1 = sup_f_rows(Y1, trim)
    p_single = (1 + np.sum(s1 >= stat)) / (reps + 1)
    if not inner or residuals != 'null':
        return p_single, np.nan, p
    p1 = np.empty(reps)
    for j in range(reps):
        _, phij, resj = _null_fit(Y1[j])
        s2 = sup_f_rows(_ar_draws(phij, resj, n, rng, inner), trim)
        p1[j] = (1 + np.sum(s2 >= s1[j])) / (inner + 1)
    p_double = (1 + np.sum(p1 <= max(p_single, 1 / (inner + 1)))) / (reps + 1)
    return p_single, p_double, p


# ============================================================================ 4. break-date CI
def long_run_variance(e):
    """Newey-West (Bartlett kernel) long-run variance, bandwidth floor(4 (n/100)^(2/9))."""
    e = np.asarray(e, float) - np.mean(e)
    n = len(e)
    L = min(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), n - 1)
    s = e @ e / n
    for lag in range(1, L + 1):
        s += 2 * (1 - lag / (L + 1)) * (e[lag:] @ e[:-lag]) / n
    return max(s, 1e-12)


def bai_cdf(x, phi=1.0):
    """CDF of argmax_s Z(s); Z(s) = W1(-s) - |s|/2 (s <= 0), sqrt(phi) W2(s) - |s|/2 (s > 0).
    Bai (1997) Rev Econ Stat 79:551-563; phi = ratio of regime-2 to regime-1 variance."""
    ax = abs(x) if x != 0 else 1e-12
    if x < 0:
        fr = 1.0 / phi
        return (-np.exp(np.log(ax) / 2 - ax / 8 - np.log(2 * np.pi) / 2)
                - phi * (phi + 2) / (phi + 1)
                * np.exp(fr * (1 + fr) * ax / 2 + norm.logcdf(-(0.5 + fr) * np.sqrt(ax)))
                + (ax / 2 - 2 + (phi + 2) ** 2 / (phi + 1)) * norm.cdf(-np.sqrt(ax) / 2))
    fr = 1.0 / phi
    return (1 + np.sqrt(fr) * np.exp(np.log(ax) / 2 - fr * ax / 8 - np.log(2 * np.pi) / 2)
            + (1 / phi) * (2 * phi + 1) / (phi + 1)
            * np.exp((phi + 1) * ax / 2 + norm.logcdf(-(phi + 0.5) / np.sqrt(phi) * np.sqrt(ax)))
            - ((2 * phi + 1) ** 2 / ((phi + 1) * phi) - 2 + fr * ax / 2)
            * norm.cdf(-np.sqrt(fr * ax) / 2))


def bai_ci(y, k, level=0.95):
    """Bai (1997) confidence interval for a shift in mean after observation k, using
    regime-specific long-run variances. Returns indices (lo, hi) of the last observation
    of the earlier regime."""
    y = np.asarray(y, float)
    y1, y2 = y[:k + 1], y[k + 1:]
    delta = y2.mean() - y1.mean()
    w1, w2 = long_run_variance(y1), long_run_variance(y2)
    a = (1 - level) / 2
    q_lo = brentq(lambda v: bai_cdf(v, w2 / w1) - a, -1e4, -1e-9)
    q_hi = brentq(lambda v: bai_cdf(v, w2 / w1) - (1 - a), 1e-9, 1e4)
    scale = delta ** 2 / w1
    # Bai (1997): [k - [c_upper / L] - 1, k - [c_lower / L] + 1], [.] = integer part (c_lower < 0),
    # i.e. k - ceil(c_upper / L) and k + ceil(|c_lower| / L), as in R's strucchange::confint
    lo = k - int(np.floor(q_hi / scale)) - 1
    hi = k + int(np.floor(-q_lo / scale)) + 1
    return max(lo, 0), min(hi, len(y) - 2)


def break_test(series, rng, trim=TRIM, reps=N_BOOT):
    """Full report for one annual series (pandas Series indexed by year)."""
    y, yrs = series.values.astype(float), series.index.values
    F, k = sup_f(y, trim)
    lo, hi = bai_ci(y, k)
    p_single, p_double, ar_order = bootstrap_supf(y, rng, trim, reps)
    return dict(n=len(y), first_year=int(yrs[0]), last_year=int(yrs[-1]),
                sup_F=F, crit_1pct=ANDREWS_CRIT[0.01], p_asymptotic=andrews_pvalue(F, trim),
                p_bootstrap=p_double, p_bootstrap_single=p_single, ar_order=ar_order,
                last_year_before=int(yrs[k]), first_year_after=int(yrs[k]) + 1,
                ci_low=int(yrs[lo]), ci_high=int(yrs[hi]),
                mean_before=float(y[:k + 1].mean()), mean_after=float(y[k + 1:].mean()))


# ============================================================================ 4b. counting noise
def peak_month_rows(counts, days):
    """Peak month of every complete calendar year, for each row of monthly counts (rows start in
    January). Same definition as annual_metrics, vectorised."""
    v = np.log(counts / days) if DAY_ADJUST else np.log(counts)
    v = np.atleast_2d(v)
    N = v.shape[1]
    w = np.r_[0.5, np.ones(11), 0.5] / 12.0
    trend = sum(w[j] * v[:, j:N - 12 + j] for j in range(13))       # centred on months 6 .. N-7
    si = (v[:, 6:N - 6] - trend)[:, 6:]                              # from January of the second year
    return si[:, :(si.shape[1] // 12) * 12].reshape(len(v), -1, 12).argmax(axis=2) + 1


def break_year_under_counting_noise(df, years, rng, reps=N_POISSON, level=0.95):
    """The Bai interval takes each year's peak month as given. In some years the highest month
    leads the next by less than the counting noise in monthly births, so the peak month itself is
    uncertain. Here every monthly count is redrawn from a Poisson distribution about the observed
    value, the peak-month series is recomputed and its break re-estimated. Returns the share of
    replicates giving each break year and the central 95% range of the estimate (both as the last
    year of the earlier regime)."""
    counts, days = df['births'].values.astype(float), df['days'].values.astype(float)
    assert df['month'].iloc[0] == 1
    assert np.array_equal(peak_month_rows(counts, days)[0], years['peak_month'].values)
    P = peak_month_rows(rng.poisson(counts, size=(reps, len(counts))).astype(float), days).astype(float)
    k = np.array([sup_f(row)[1] for row in P])
    est = years.index.values[k]
    a = (1 - level) / 2
    shares = {int(y): float(np.mean(est == y)) for y in np.unique(est)}
    return dict(shares=shares, low=int(np.quantile(est, a, method='lower')),
                high=int(np.quantile(est, 1 - a, method='higher')))


# ============================================================================ 5. logistic transition
LN81 = 2 * np.log(9.0)     # duration (10% -> 90% complete) = LN81 x logistic scale


def _logistic_grid(t, mids, durs):
    s = durs[None, None, :] / LN81
    z = np.clip(-(t[:, None, None] - mids[None, :, None]) / s, -700, 700)
    return 1.0 / (1.0 + np.exp(z))             # (n, n_mid, n_dur)


def logistic_grid(t):
    """Search grid: mid-points every quarter-year (not within 5 years of either end); durations
    of 0.1 (in effect a step), then 1 to 60 years in half-years, to 80 in years, to 120 in
    two-year steps (durations beyond the length of the series are not really identified)."""
    t = np.asarray(t, float)
    mids = np.arange(t.min() + 5, t.max() - 5 + 1e-9, 0.25)
    durs = np.r_[0.1, np.arange(1.0, 60.0, 0.5), np.arange(60.0, 80.0, 1.0), np.arange(80.0, 120.01, 2.0)]
    return _logistic_grid(t, mids, durs), mids, durs


def logistic_fit_rows(Y, G, mids, durs, chunk=250):
    """Least-squares fit of y = a + b / (1 + exp(-(t - mid)/s)) to every row of Y by exhaustive
    search over (mid, duration); a and b are solved in closed form at every grid point.
    Returns arrays mid, duration, rss and the fitted curves (one row per series)."""
    Y = np.atleast_2d(np.asarray(Y, float))
    n, M, D = G.shape
    g2 = G.reshape(n, M * D)
    gbar = g2.mean(axis=0)
    gc = g2 - gbar
    sgg = (gc ** 2).sum(axis=0)
    ok = sgg > 1e-12
    mid, dur, rss = np.empty(len(Y)), np.empty(len(Y)), np.empty(len(Y))
    fitted = np.empty_like(Y)
    for c0 in range(0, len(Y), chunk):
        y = Y[c0:c0 + chunk]
        ybar = y.mean(axis=1, keepdims=True)
        yc = y - ybar
        sgy = yc @ gc
        gain = np.where(ok, sgy ** 2 / np.where(ok, sgg, 1.0), -np.inf)       # reduction in RSS
        j = gain.argmax(axis=1)
        r = np.arange(len(j))
        b = sgy[r, j] / sgg[j]
        rss[c0:c0 + chunk] = (yc ** 2).sum(axis=1) - gain[r, j]
        mid[c0:c0 + chunk], dur[c0:c0 + chunk] = mids[j // D], durs[j % D]
        fitted[c0:c0 + chunk] = ybar + b[:, None] * (g2[:, j].T - gbar[j][:, None])
    return mid, dur, rss, fitted


def logistic_fit(t, y, grid=None):
    """Logistic transition fitted to one series. Returns (dict, grid)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    grid = grid or logistic_grid(t)
    G, mids, durs = grid
    mid, dur, rss, fitted = logistic_fit_rows(y[None, :], G, mids, durs)
    g = G[:, int(np.searchsorted(mids, mid[0])), int(np.searchsorted(durs, dur[0]))]
    b = float(((y - y.mean()) @ (g - g.mean())) / ((g - g.mean()) ** 2).sum())
    return dict(mid=float(mid[0]), duration=float(dur[0]), a=float(y.mean() - b * g.mean()), b=b,
                rss=float(rss[0]), fitted=fitted[0]), grid


def _percentile_rank(x, value):
    """Share of x below value (ties count half)."""
    return float(np.mean(x < value) + 0.5 * np.mean(x == value))


def logistic_with_ci(series, rng, reps=N_BOOT, outer=N_OUTER, inner=N_INNER_LOGISTIC, level=0.95):
    """Logistic transition with bootstrap confidence intervals for its mid-point and duration.

    First level: `reps` series are drawn from the fitted transition plus autoregressive errors
    (order <= 3 by BIC) and refitted. Plain percentile intervals from these replicates are too
    narrow (see validation_simulations.py), so the reported intervals are calibrated by a second
    level (Beran 1987; Loh 1987): each of `outer` first-level series is treated as if it were the
    data and given its own `inner` replicates, which shows where the 'true' value (the original
    estimate) falls in a bootstrap distribution. The reported limits are the first-level
    percentiles that would have missed the truth in 2.5% of those series on each side. The same
    calibration with one common level for both tails is returned for comparison: the two can
    differ where a limit lies far out in a long tail, as the upper limit for the French
    duration does. outer=0 skips the second level."""
    t, y = series.index.values.astype(float), series.values.astype(float)
    n = len(y)
    fit, grid = logistic_fit(t, y)
    e = y - fit['fitted']
    p, phi, res = _select_ar(e - e.mean())
    Y1 = fit['fitted'] + _ar_draws(phi, res, n, rng, reps)
    bm, bd, _, F1 = logistic_fit_rows(Y1, *grid)
    a = (1 - level) / 2
    fit.update(mid_ci_percentile=tuple(np.quantile(bm, [a, 1 - a])),
               dur_ci_percentile=tuple(np.quantile(bd, [a, 1 - a])),
               lag1_resid=float(np.corrcoef(e[1:], e[:-1])[0, 1]), n=n, ar_order=p)
    if not outer:
        fit.update(mid_ci=fit['mid_ci_percentile'], dur_ci=fit['dur_ci_percentile'])
        return fit
    um, ud = np.empty(outer), np.empty(outer)
    for j in range(outer):
        ej = Y1[j] - F1[j]
        _, phij, resj = _select_ar(ej - ej.mean())
        m2, d2, _, _ = logistic_fit_rows(F1[j] + _ar_draws(phij, resj, n, rng, inner), *grid)
        um[j], ud[j] = _percentile_rank(m2, fit['mid']), _percentile_rank(d2, fit['duration'])
    qm, qd = np.quantile(um, [a, 1 - a]), np.quantile(ud, [a, 1 - a])
    # the same calibration with one common level for both tails (for comparison)
    lm, ld = (min(1.0, 2 * float(np.quantile(np.abs(u - 0.5), level))) for u in (um, ud))
    fit.update(mid_ci=tuple(np.quantile(bm, qm)), dur_ci=tuple(np.quantile(bd, qd)),
               mid_levels=tuple(qm), dur_levels=tuple(qd),
               mid_ci_symmetric=tuple(np.quantile(bm, [0.5 - lm / 2, 0.5 + lm / 2])),
               dur_ci_symmetric=tuple(np.quantile(bd, [0.5 - ld / 2, 0.5 + ld / 2])),
               symmetric_levels=(lm, ld),
               percentile_coverage_mid=float(np.mean((um >= a) & (um <= 1 - a))),
               percentile_coverage_dur=float(np.mean((ud >= a) & (ud <= 1 - a))))
    return fit


def step_vs_logistic(series, rng, reps=N_BOOT):
    """Is the transition abrupt? Bootstrap likelihood-ratio test of H0: one instantaneous step
    (with autoregressive errors) against a logistic transition of any duration."""
    t, y = series.index.values.astype(float), series.values.astype(float)
    n = len(y)
    fit, grid = logistic_fit(t, y)
    F, k = sup_f(y)
    step = np.where(np.arange(n) <= k, y[:k + 1].mean(), y[k + 1:].mean())
    rss_step = float(((y - step) ** 2).sum())
    lr_obs = n * np.log(rss_step / fit['rss'])
    p, phi, res = _select_ar(y - step)
    Yb = step + _ar_draws(phi, res, n, rng, reps)
    rss0 = ((Yb - Yb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    rss_step_b = rss0 / (1 + sup_f_rows(Yb) / (n - 2))           # RSS of the best single step
    lr = n * np.log(rss_step_b / logistic_fit_rows(Yb, *grid)[2])
    return dict(lr=float(lr_obs), p=float((1 + np.sum(lr >= lr_obs)) / (reps + 1)),
                null_95=float(np.quantile(lr, 0.95)), rss_step=rss_step)


# ============================================================================ 6. several shifts
def best_partition(y, m, h):
    """Global least-squares partition of y into m + 1 regimes of at least h observations
    (dynamic programming, as in Bai & Perron 2003). Returns the last index of each regime."""
    y = np.asarray(y, float)
    n = len(y)
    cs, cs2 = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(y ** 2)]

    def rss(i, j):
        return (cs2[j] - cs2[i]) - (cs[j] - cs[i]) ** 2 / (j - i)

    cost = np.full((m + 1, n + 1), np.inf)
    back = np.zeros((m + 1, n + 1), int)
    for j in range(h, n + 1):
        cost[0, j] = rss(0, j)
    for b in range(1, m + 1):
        for j in range((b + 1) * h, n + 1):
            c = [cost[b - 1, i] + rss(i, j) for i in range(b * h, j - h + 1)]
            a = int(np.argmin(c))
            cost[b, j], back[b, j] = c[a], b * h + a
    brk, j = [], n
    for b in range(m, 0, -1):
        j = back[b, j]
        brk.append(j - 1)
    return sorted(brk), float(cost[m, n])


def number_of_shifts(series, mmax=5, min_frac=0.10):
    """Number of shifts in mean chosen by BIC and by the LWZ criterion (Liu, Wu & Zidek 1997)."""
    y, yrs = series.values.astype(float), series.index.values
    n = len(y)
    h = int(np.ceil(min_frac * n))
    rows = []
    for m in range(0, mmax + 1):
        if (m + 1) * h > n:
            break
        brk, rss = best_partition(y, m, h)
        k = 2 * m + 1
        rows.append(dict(m=m, after=[int(yrs[b]) for b in brk],
                         bic=np.log(rss / n) + k * np.log(n) / n,
                         lwz=np.log(rss / (n - k)) + k * 0.299 * np.log(n) ** 2.1 / n))
    return rows


# ============================================================================ main
def analyse_country(job):
    """Everything for one country. Each procedure has its own reproducible random stream, so the
    two countries can be run in parallel and results do not depend on the order of the steps."""
    ci, country, path = job
    lines = []
    say = lines.append

    def stream(*key):
        return np.random.default_rng([SEED, ci, *key])

    df = load_births(path)
    met, tab = annual_metrics(df)
    say(f'\n=== {country}: monthly data {df.index.min():%b %Y} to {df.index.max():%b %Y} '
        f'({len(df)} months); annual metrics {met.index.min()}-{met.index.max()} (n={len(met)}) ===')
    m = met.copy()
    m.insert(0, 'country', country)
    m['phase_date'] = [phase_to_date(v) for v in m['phase']]
    idx = (np.exp(tab) * 100).round(2)
    idx.columns = MON3
    idx.insert(0, 'country', country)

    tests = []
    for mi, (metric, label, unit) in enumerate([('peak_month', 'Peak month', 'month number'),
                                                ('phase', 'Phase (acrophase)', 'months since 1 Jan'),
                                                ('amplitude', 'Amplitude (peak-to-trough)', '% of trend')]):
        r = break_test(met[metric].astype(float), stream(mi))
        r.update(country=country, metric=metric, noise_low=np.nan, noise_high=np.nan,
                 ci_low_reported=r['ci_low'], ci_high_reported=r['ci_high'])
        if metric == 'peak_month':
            nz = break_year_under_counting_noise(df, met, stream(20))
            r.update(noise_low=nz['low'], noise_high=nz['high'],
                     ci_low_reported=min(r['ci_low'], nz['low']), ci_high_reported=max(r['ci_high'], nz['high']))
        tests.append(r)
        say(f'{label}: largest shift in mean between {r["last_year_before"]} and {r["first_year_after"]} '
            f'(95% CI {r["ci_low"]}-{r["ci_high"]}); mean {r["mean_before"]:.2f} -> {r["mean_after"]:.2f} {unit}'
            + (f'  = {phase_to_date(r["mean_before"])} -> {phase_to_date(r["mean_after"])}'
               if metric == 'phase' else ''))
        if metric == 'peak_month':
            say('    with monthly counts redrawn as Poisson variates the break falls after '
                + ', '.join(f'{y} in {v * 100:.1f}%' for y, v in nz['shares'].items())
                + f' of {N_POISSON} replicates (central 95% range {nz["low"]}-{nz["high"]})')
            say(f'    REPORTED 95% interval (the Bai interval widened to that range): last year of the earlier '
                f'regime {r["ci_low_reported"]}-{r["ci_high_reported"]}, i.e. the switch falls between '
                f'{r["ci_low_reported"]}/{str(r["ci_low_reported"] + 1)[2:]} and '
                f'{r["ci_high_reported"]}/{str(r["ci_high_reported"] + 1)[2:]}')
        say(f'    sup-F = {r["sup_F"]:.1f}; p = {r["p_bootstrap"]:.4f} (double bootstrap, AR({r["ar_order"]}) '
            f'errors fitted under the null)   [not reported: single bootstrap p = '
            f'{r["p_bootstrap_single"]:.4f}; i.i.d. asymptotic p = {r["p_asymptotic"]:.2g}]')

    lf = logistic_with_ci(met['phase'], stream(10))
    lf.update(country=country)
    say(f'Logistic transition in phase: mid-point {lf["mid"]:.2f} (95% CI {lf["mid_ci"][0]:.2f}-'
        f'{lf["mid_ci"][1]:.2f}); central 80% of the shift took {lf["duration"]:.1f} years (95% CI '
        f'{lf["dur_ci"][0]:.1f}-{lf["dur_ci"][1]:.1f}); plateaus {phase_to_date(lf["a"])} -> '
        f'{phase_to_date(lf["a"] + lf["b"])}; residual errors AR({lf["ar_order"]})')
    say(f'    [calibrated with one common level instead of tail by tail: {lf["mid_ci_symmetric"][0]:.2f}-'
        f'{lf["mid_ci_symmetric"][1]:.2f} and {lf["dur_ci_symmetric"][0]:.1f}-{lf["dur_ci_symmetric"][1]:.1f}]')
    say(f'    [not reported: plain percentile intervals {lf["mid_ci_percentile"][0]:.2f}-'
        f'{lf["mid_ci_percentile"][1]:.2f} and {lf["dur_ci_percentile"][0]:.1f}-'
        f'{lf["dur_ci_percentile"][1]:.1f}, which would have covered the truth in '
        f'{lf["percentile_coverage_mid"] * 100:.0f}% and {lf["percentile_coverage_dur"] * 100:.0f}% of the '
        f'{N_OUTER} series simulated from the fitted model]')
    sv = step_vs_logistic(met['phase'], stream(11))
    lf.update(lr_step=sv['lr'], p_step=sv['p'], rss_step=sv['rss_step'])
    say(f'    abrupt step vs gradual transition: residual sum of squares {sv["rss_step"]:.2f} (step) vs '
        f'{lf["rss"]:.2f} (logistic); likelihood ratio {sv["lr"]:.1f} (95th percentile under an '
        f'abrupt step: {sv["null_95"]:.1f}); bootstrap p = {sv["p"]:.4f}')
    sc = lf['duration'] / LN81
    for lab, target in (('1 June', 151 / 365 * 12), ('1 July', 181 / 365 * 12)):
        frac = (target - lf['a']) / lf['b']
        say(f'    fitted acrophase passes {lab} in {lf["mid"] - sc * np.log(1 / frac - 1):.1f}')
    say(f'    fitted transition 10% complete in {lf["mid"] - lf["duration"] / 2:.1f}, '
        f'90% complete in {lf["mid"] + lf["duration"] / 2:.1f}')
    ns = number_of_shifts(met['phase'])
    for crit in ('bic', 'lwz'):
        b = min(ns, key=lambda q: q[crit])
        say(f'    shifts in mean phase chosen by {crit.upper()} (descriptive; assumes independent errors): '
            f'{b["m"]} (after ' + ', '.join(str(v) for v in b['after']) + ')')

    # descriptive counts used in the text
    pm = met['peak_month']
    for a, b in [(met.index.min(), 1979), (1980, met.index.max())] if country == 'Germany' \
            else [(1953, 1989), (1990, 2004), (2005, met.index.max())]:
        c = pm.loc[a:b].map(lambda v: MON3[v - 1]).value_counts()
        say(f'    peak month {a}-{b}: ' + ', '.join(f'{k} {v}' for k, v in c.items()))
    if country == 'France':
        late = pm.loc[1990:]
        say(f'    peak in September or October: {int(late.isin([9, 10]).sum())} of {len(late)} years from 1990 '
            f'({", ".join(str(y) for y in late.index[late.isin([9, 10])])})')
    for a, b in [(1951, 1970), (2004, 2023)]:
        say(f'    mean phase {a}-{b}: {met.loc[a:b, "phase"].mean():.2f} '
            f'({phase_to_date(met.loc[a:b, "phase"].mean())}); '
            f'mean amplitude {met.loc[a:b, "amplitude"].mean():.1f}%')
    lf.pop('fitted')
    return dict(lines=lines, metrics=m.reset_index(), index=idx.reset_index(), tests=tests, fit=lf)


def run(data_dir='data', out_dir='results', processes=2):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    jobs = [(ci, country, data_dir / fn) for ci, (country, fn) in enumerate(FILES.items())]
    if int(processes) > 1:
        from multiprocessing import Pool
        with Pool(int(processes)) as pool:
            out = pool.map(analyse_country, jobs, chunksize=1)
    else:
        out = [analyse_country(j) for j in jobs]

    lines = ['Birth seasonality in France and Germany: revised analysis',
             f'births per day: {DAY_ADJUST};  trimming: {TRIM};  bootstrap replicates: {N_BOOT} '
             f'(second level for sup-F: {N_INNER} per first-level series; for the logistic transition: '
             f'{N_INNER_LOGISTIC} for each of {N_OUTER})',
             f'Asymptotic sup-F critical values for independent errors (Andrews 2003): {ANDREWS_CRIT}']
    for o in out:
        lines += o['lines']
    print('\n'.join(lines))
    tests = [r for o in out for r in o['tests']]
    fits = [o['fit'] for o in out]
    pd.concat([o['metrics'] for o in out]).to_csv(out_dir / 'annual_metrics.csv', index=False,
                                                  float_format='%.4f')
    pd.concat([o['index'] for o in out]).to_csv(out_dir / 'seasonal_index_by_month.csv', index=False)
    cols = ['country', 'metric', 'n', 'first_year', 'last_year', 'sup_F', 'crit_1pct', 'p_asymptotic',
            'p_bootstrap', 'p_bootstrap_single', 'ar_order', 'last_year_before',
            'first_year_after', 'ci_low', 'ci_high', 'noise_low', 'noise_high', 'ci_low_reported',
            'ci_high_reported', 'mean_before', 'mean_after']
    pd.DataFrame(tests)[cols].to_csv(out_dir / 'break_tests.csv', index=False, float_format='%.5g')
    pd.DataFrame([dict(country=f['country'], midpoint=f['mid'], mid_ci_low=f['mid_ci'][0],
                       mid_ci_high=f['mid_ci'][1], duration_10_90=f['duration'],
                       dur_ci_low=f['dur_ci'][0], dur_ci_high=f['dur_ci'][1],
                       mid_symmetric_low=f['mid_ci_symmetric'][0],
                       mid_symmetric_high=f['mid_ci_symmetric'][1],
                       dur_symmetric_low=f['dur_ci_symmetric'][0],
                       dur_symmetric_high=f['dur_ci_symmetric'][1],
                       mid_percentile_low=f['mid_ci_percentile'][0],
                       mid_percentile_high=f['mid_ci_percentile'][1],
                       dur_percentile_low=f['dur_ci_percentile'][0],
                       dur_percentile_high=f['dur_ci_percentile'][1],
                       lower_plateau=f['a'], upper_plateau=f['a'] + f['b'],
                       rss_logistic=f['rss'], rss_step=f['rss_step'], lr_step_vs_logistic=f['lr_step'],
                       p_step=f['p_step']) for f in fits]
                 ).to_csv(out_dir / 'phase_transition_fits.csv', index=False, float_format='%.4f')
    (out_dir / 'results_summary.txt').write_text('\n'.join(lines) + '\n')
    return tests, fits


if __name__ == '__main__':
    run(*sys.argv[1:4])
