#!/usr/bin/env python3
"""
Supporting checks behind the response to reviewers of the manuscript.

  1. Reproduces the statistics in the submitted manuscript from the v1.0.0 method.
  2. False-positive rate of a scanned Chow test read against the ordinary F distribution.
  3. Absolute seasonal amplitude tracks the number of births (the German "1971 break").
  4. Robustness of the revised peak-month break, acrophase transition and amplitude to analytic choices.
  5. Photoperiod across each country's latitudes, and its effect on the submitted regression.
  6. Number of shifts in mean selected by information criteria (is one step enough?).
  7. Coverage of the German series (whole of Germany or former FRG only?).
  8. How narrowly the peak month is decided, and how similar the two acrophase trajectories are.
  9. What a regression of births on day length can and cannot see (Reviewer 2, point 2).
 10. Peak month with and without the adjustment for month length.

The size and coverage of the bootstrap procedures are checked separately, in
validation_simulations.py.

Usage:  python reviewer_checks.py [data_dir] [results_dir]
Requires statsmodels (for STL), which the main analysis does not need.
"""
import calendar
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist
from statsmodels.tsa.seasonal import STL

import birth_seasonality_analysis as bsa

warnings.filterwarnings('ignore')
data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data')
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else 'results')
out_dir.mkdir(exist_ok=True)
LINES = []


def say(s=''):
    print(s)
    LINES.append(s)


def ols_rss(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return float(e @ e)


def submitted_amplitude(births):
    """v1.0.0: robust STL on monthly counts; annual max - min of the seasonal component."""
    seas = STL(births.astype(float), period=12, robust=True).fit().seasonal
    return seas.groupby(seas.index.year).agg(lambda x: x.max() - x.min())


def scan_v1(y):
    """v1.0.0 scan: constant-only Chow F at every candidate year from (first + 5) to (last - 5),
    the candidate being the last year of the earlier regime. Returns (largest F, its index)."""
    y = np.asarray(y, float)
    n = len(y)
    cs, cs2 = np.cumsum(y), np.cumsum(y ** 2)
    ks = np.arange(5, n - 5)
    n1 = ks + 1
    rss = (cs2[ks] - cs[ks] ** 2 / n1) + ((cs2[-1] - cs2[ks]) - (cs[-1] - cs[ks]) ** 2 / (n - n1))
    F = ((cs2[-1] - cs[-1] ** 2 / n) - rss) / (rss / (n - 2))
    j = int(np.argmax(F))
    return float(F[j]), int(ks[j])


def submitted_scan(amp):
    """v1.0.0: largest scanned Chow F, read against the ordinary F(1, n - 2) distribution."""
    y = amp.values.astype(float)
    F, k = scan_v1(y)
    return int(amp.index[k]), F, fdist.sf(F, 1, len(y) - 2)


def photoperiod_scan(seasonal_births, phot_seasonal, window=None):
    """v1.0.0: OLS of the seasonal births component on photoperiod; Chow F at every month."""
    y, x, dates = seasonal_births.values, phot_seasonal.values, seasonal_births.index
    X = np.column_stack([np.ones(len(y)), x])
    n, k = X.shape
    r0 = ols_rss(y, X)
    lo, hi = dates.min() + pd.DateOffset(years=5), dates.max() - pd.DateOffset(years=5)
    best = (-1.0, None)
    for i, d in enumerate(dates):
        if d <= lo or d >= hi or (window and not (window[0] <= d <= window[1])):
            continue
        pre = np.arange(n) <= i
        rs = ols_rss(y[pre], X[pre]) + ols_rss(y[~pre], X[~pre])
        F = ((r0 - rs) / k) / (rs / (n - 2 * k))
        if F > best[0]:
            best = (F, d)
    return best[1], best[0], fdist.sf(best[0], k, n - 2 * k)


def daylength(lat_deg, doy, ylen=365):
    """Hours between sunrise and sunset (sun centre at -0.833 deg; Spencer 1971 declination)."""
    g = 2 * np.pi * (doy - 0.5) / ylen
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    lat = np.radians(lat_deg)
    cosw = (np.sin(np.radians(-0.833)) - np.sin(lat) * np.sin(decl)) / (np.cos(lat) * np.cos(decl))
    return 24 / np.pi * np.arccos(np.clip(cosw, -1, 1))


def monthly_photoperiod(lat, year=2001):
    out, d0 = [], 0
    for m in range(1, 13):
        n = calendar.monthrange(year, m)[1]
        out.append(daylength(lat, np.arange(d0 + 1, d0 + n + 1)).mean())
        d0 += n
    return np.array(out)


def one_line(series, trim=bsa.TRIM):
    y, yrs = series.values.astype(float), series.index.values
    F, k = bsa.sup_f(y, trim)
    lo, hi = bsa.bai_ci(y, k)
    return (f'break {yrs[k]}/{str(yrs[k] + 1)[2:]} (95% CI {yrs[lo]}-{yrs[hi]}), sup-F {F:.0f}, '
            f'mean {y[:k + 1].mean():.2f} -> {y[k + 1:].mean():.2f}')


def metrics_from_index(df, si):
    tab = pd.DataFrame({'year': df['year'], 'month': df['month'], 'si': si}).pivot(
        index='year', columns='month', values='si').dropna()
    rows = []
    for yr, r in tab.iterrows():
        amp, ph = bsa.cosinor(r.values, yr)
        rows.append(dict(year=yr, phase=ph, amplitude=(np.exp(r.max()) - np.exp(r.min())) * 100,
                         peak_month=int(r.idxmax())))
    return pd.DataFrame(rows).set_index('year')


# ------------------------------------------------------------------ load
frames = {c: bsa.load_births(data_dir / fn) for c, fn in bsa.FILES.items()}
phot_files = {'Germany': 'german_photoperiod_data.csv', 'France': 'french_photoperiod_data.csv'}

# ------------------------------------------------------------------ 1
say('1. STATISTICS AS SUBMITTED (v1.0.0 method, re-run on the archived data)')
for c, df in frames.items():
    amp = submitted_amplitude(df['births'])
    yr, F, p = submitted_scan(amp)
    say(f'   {c} amplitude: break {yr}, F = {F:.3f}, ordinary-F p = {p:.3g}   '
        f'[submitted: {"1971, F = 246.08" if c == "Germany" else "1994, F = 33.77"}]')
    ph = pd.read_csv(data_dir / phot_files[c], header=None, names=['y', 'm', 'b', 'phot'])['phot']
    ph.index = df.index
    sb = STL(df['births'].astype(float), period=12, robust=True).fit().seasonal
    sp = STL(ph.astype(float), period=12, robust=True).fit().seasonal
    w = (pd.Timestamp('1955-01-01'), pd.Timestamp('1995-12-01'))
    d1, F1, p1 = photoperiod_scan(sb, sp, window=w)
    d2, F2, p2 = photoperiod_scan(sb, sp)
    say(f'   {c} photoperiod regression: scan 1955-1995 -> {d1:%b %Y}, F = {F1:.2f}, p = {p1:.3g};'
        f'  scan of the whole series -> {d2:%b %Y}, F = {F2:.2f}, p = {p2:.3g}')
say('   (Manuscript: France mid-1995, F = 30.94; Germany 1962, F = 1.96, p = 0.15. The German')
say('    value came from the 1955-1995 window; over the whole series the maximum is in 2012.)')

# ------------------------------------------------------------------ 2
say('\n2. FALSE-POSITIVE RATE OF A SCANNED CHOW TEST (Reviewer 2, point 1)')
rng = np.random.default_rng(1)
NSIM = 100000
for n in (75, 79):
    sims = np.array([scan_v1(rng.standard_normal(n))[0] for _ in range(NSIM)])
    say(f'   n = {n} years of pure noise, break scanned exactly as in v1.0.0: ordinary-F "p < 0.05" in '
        f'{np.mean(fdist.sf(sims, 1, n - 2) < 0.05) * 100:.1f}% of series; "p < 0.001" in '
        f'{np.mean(fdist.sf(sims, 1, n - 2) < 0.001) * 100:.1f}%   ({NSIM:,} simulations)')
sims = np.array([bsa.sup_f(rng.standard_normal(75))[0] for _ in range(NSIM)])
say(f'   sup-F with Andrews 5% critical value (8.68), 15% trimming, n = 75: rejects in '
    f'{np.mean(sims > 8.68) * 100:.1f}% of series')

# ------------------------------------------------------------------ 3
say('\n3. ABSOLUTE AMPLITUDE TRACKS THE NUMBER OF BIRTHS')
for c, df in frames.items():
    amp = submitted_amplitude(df['births'])
    ann = df.groupby('year')['births'].sum()
    rel = amp / (ann / 12) * 100
    say(f'   {c}: correlation of absolute amplitude with annual births r = '
        f'{np.corrcoef(amp.values, ann.values)[0, 1]:.2f}')
    yr, F, p = submitted_scan(rel)
    k = list(rel.index).index(yr)
    say(f'      same amplitude as % of mean monthly births: largest F at {yr} (F = {F:.1f}), '
        f'{rel.values[:k + 1].mean():.1f}% -> {rel.values[k + 1:].mean():.1f}%')
    # synthetic series: ONE fixed seasonal profile multiplied by the observed trend
    lb = np.log(df['births'].astype(float))
    tr = bsa.centred_ma(lb)
    t = np.arange(len(tr), dtype=float)
    ok = tr.notna().values
    trv = tr.values.copy()
    trv[:6] = np.polyval(np.polyfit(t[ok][:12], trv[ok][:12], 1), t[:6])
    trv[-6:] = np.polyval(np.polyfit(t[ok][-12:], trv[ok][-12:], 1), t[-6:])
    profile = (lb - tr).groupby(df['month']).mean()
    synth = pd.Series(np.exp(trv + df['month'].map(profile).values), index=df.index)
    yr, F, p = submitted_scan(submitted_amplitude(synth))
    say(f'      synthetic series with an UNCHANGING seasonal profile and the observed trend, through the '
        f'submitted pipeline: "break" {yr}, F = {F:.0f}, p = {p:.1g}')
g = frames['Germany'].groupby('year')['births'].sum()
say(f'   German births: {g.loc[1964]:,} (1964) -> {g.loc[1975]:,} (1975), '
    f'{(g.loc[1975] / g.loc[1964] - 1) * 100:.0f}%')

# ------------------------------------------------------------------ 4
say('\n4. ROBUSTNESS OF THE REVISED RESULTS TO ANALYTIC CHOICES')
for c, df in frames.items():
    say(f'   {c}')
    base, _ = bsa.annual_metrics(df)
    raw, _ = bsa.annual_metrics(df, day_adjust=False)
    lr = np.log(df['rate'])
    stl = STL(lr, period=12, robust=True).fit()
    trend24 = lr.rolling(24, center=True).mean().shift(-1).rolling(2).mean()      # centred 2 x 24 months
    variants = [('primary (births/day, centred moving average)', base),
                ('monthly counts, no day adjustment', raw),
                ('births/day, robust STL trend', metrics_from_index(df, lr - stl.trend)),
                ('births/day, STL seasonal component (smoothed over years)',
                 metrics_from_index(df, stl.seasonal)),
                ('births/day, centred 24-month moving average', metrics_from_index(df, lr - trend24)),
                ('births/day, not detrended', metrics_from_index(df, lr))]
    for lab, m in variants:
        lf, _ = bsa.logistic_fit(m.index.values, m['phase'].values)
        say(f'      {lab}:')
        say(f'         peak month {one_line(m["peak_month"].astype(float))}')
        if 'not detrended' in lab:          # acrophase and amplitude need an index that is relative to trend
            continue
        say(f'         acrophase  transition mid-point {lf["mid"]:.2f}, central 80% in {lf["duration"]:.1f} years; '
            f'treated as one shift: {one_line(m["phase"])}')
        say(f'         amplitude  {one_line(m["amplitude"])}')
    say(f'      first-harmonic amplitude: {one_line(base["amplitude_h1"])}')
    for tr_ in (0.10, 0.20, 0.25):
        say(f'      trimming {tr_:.2f}: peak month {one_line(base["peak_month"].astype(float), tr_)}')

# ------------------------------------------------------------------ 5
say('\n5. PHOTOPERIOD ACROSS LATITUDES (Reviewer 1, point 4)')
say('   day length elsewhere = the submitted reference-city values + the difference between the two')
say('   latitudes from the sunrise-sunset formula, so the reference city reproduces section 1 exactly')
cities = {'Germany': [('Flensburg', 54.78), ('Berlin', 52.52), ('Frankfurt', 50.11), ('Munich', 48.14),
                      ('Oberstdorf', 47.41)],
          'France': [('Dunkirk', 51.03), ('Paris', 48.86), ('Lyon', 45.76), ('Marseille', 43.30),
                     ('Perpignan', 42.70)]}
for c, df in frames.items():
    obs = pd.read_csv(data_dir / phot_files[c], header=None, names=['y', 'm', 'b', 'phot'])['phot'].values
    ref = dict(cities[c])['Berlin' if c == 'Germany' else 'Paris']
    say(f'   {c}: largest difference between the day-length formula and the Open-Meteo values used in '
        f'the submission: {np.abs(monthly_photoperiod(ref) - obs[:12]).max():.2f} h')
    sb = STL(df['births'].astype(float), period=12, robust=True).fit().seasonal
    for name, lat in cities[c]:
        pm = monthly_photoperiod(lat)
        series = obs + (pm - monthly_photoperiod(ref))[df['month'].values - 1]
        sp = STL(pd.Series(series, index=df.index), period=12, robust=True).fit().seasonal
        d, F, p = photoperiod_scan(sb, sp)
        say(f'      {name:<10} {lat:5.2f} N: June {pm[5]:.1f} h, December {pm[11]:.1f} h (range {pm.max() - pm.min():.1f} h); '
            f'r with reference city = {np.corrcoef(pm, monthly_photoperiod(ref))[0, 1]:.4f}; '
            f'submitted regression scan -> {d:%b %Y}, F = {F:.2f}')

# ------------------------------------------------------------------ 6
say('\n6. HOW MANY SHIFTS IN MEAN PHASE? (minimum segment 10% of the sample)')
for c, df in frames.items():
    met, _ = bsa.annual_metrics(df)
    rows = bsa.number_of_shifts(met['phase'], mmax=4)
    bb, bl = min(rows, key=lambda q: q['bic'])['m'], min(rows, key=lambda q: q['lwz'])['m']
    for q in rows:
        tag = ('  <- BIC' if q['m'] == bb else '') + ('  <- LWZ' if q['m'] == bl else '')
        brk = ', '.join(f'{v}/{str(v + 1)[2:]}' for v in q['after']) or '-'
        say(f'   {c} {q["m"]} shift(s): {brk:<34} BIC {q["bic"]:6.3f}  LWZ {q["lwz"]:6.3f}{tag}')
    for frac in (0.08, 0.15):
        rr = bsa.number_of_shifts(met['phase'], mmax=5, min_frac=frac)
        say(f'      minimum segment {frac:.0%}: BIC -> after {min(rr, key=lambda q: q["bic"])["after"]}; '
            f'LWZ -> after {min(rr, key=lambda q: q["lwz"])["after"]}')

# ------------------------------------------------------------------ 7
say('\n7. COVERAGE OF THE GERMAN SERIES (Reviewer 1, point 5)')
tot = int(frames['Germany'].loc['1950', 'births'].sum())
say(f'   Sum of the 12 monthly values for 1950 in Destatis 12612-0002: {tot:,}')
say('   Published annual live births 1950: former FRG 812,835 + former GDR 303,866 = 1,116,701')
say(f'   -> the series is the whole of Germany (FRG + GDR) in every year: {tot == 812835 + 303866}')

# ------------------------------------------------------------------ 8
say('\n8. PEAK MONTH MARGINS AND SIMILARITY OF THE ACROPHASE TRAJECTORIES')
phases = {}
for c, df in frames.items():
    met, tab = bsa.annual_metrics(df)
    phases[c] = met['phase']
    idx = np.exp(tab) * 100
    srt = np.sort(idx.values, axis=1)
    margin = pd.Series(srt[:, -1] - srt[:, -2], index=idx.index)
    close = margin[margin < 0.5]
    say(f'   {c}: top month leads the runner-up by less than 0.5% of trend in {len(close)} of {len(margin)} years')
    for yr in ((1976, 1979) if c == 'Germany' else (1989, 1990)):
        r = idx.loc[yr].sort_values(ascending=False)
        say(f'      {yr}: {bsa.MON3[r.index[0] - 1]} {r.iloc[0]:.1f}, {bsa.MON3[r.index[1] - 1]} {r.iloc[1]:.1f} '
            f'(margin {r.iloc[0] - r.iloc[1]:.2f})')
for a, b in ((1951, 1970), (1974, 1989), (1990, 2000), (2001, 2023)):
    g, f = phases['Germany'].loc[a:b].mean(), phases['France'].loc[a:b].mean()
    say(f'   mean acrophase {a}-{b}: Germany {bsa.phase_to_date(g)}, France {bsa.phase_to_date(f)} '
        f'(France minus Germany {(f - g) / 12 * 365:+.0f} days)')

# ------------------------------------------------------------------ 9
say('\n9. WHAT A REGRESSION OF BIRTHS ON DAY LENGTH CAN SEE (Reviewer 2, point 2)')
say('   slope of the mean seasonal profile (% of trend) on monthly day length at the reference city')
ref_lat = {'Germany': 52.52, 'France': 48.86}
spans = {'Germany': ((1951, 1970), (1971, 1979), (1980, 1989), (1990, 2003), (2004, 2023)),
         'France': ((1951, 1970), (1971, 1979), (1980, 1989), (1990, 2003), (2004, 2023))}
for c, df in frames.items():
    _, tab = bsa.annual_metrics(df)
    dl = monthly_photoperiod(ref_lat[c])
    for a, b in spans[c]:
        prof = (np.exp(tab.loc[a:b]).mean() - 1) * 100
        say(f'   {c} {a}-{b}: slope {np.polyfit(dl, prof.values, 1)[0]:+.2f}% per hour of day length '
            f'(r = {np.corrcoef(dl, prof.values)[0, 1]:+.2f}); highest month of the mean profile: '
            f'{bsa.MON3[int(prof.idxmax()) - 1]}')
say('   (the regression measures only the part of the seasonal pattern that is in phase with day length:')
say('    a peak that moves from one side of the June solstice to the other can leave the slope unchanged)')

# ------------------------------------------------------------------ 10
say('\n10. PEAK MONTH WITH AND WITHOUT THE ADJUSTMENT FOR MONTH LENGTH')
for c, (a, b) in (('Germany', (1980, 2023)), ('France', (1990, 2023))):
    df = frames[c]
    met, _ = bsa.annual_metrics(df)
    raw = df.pivot(index='year', columns='month', values='births').loc[a:b].idxmax(axis=1)
    cnt = lambda s_: ', '.join(f'{bsa.MON3[m - 1]} {n}' for m, n in s_.value_counts().items())
    say(f'   {c} {a}-{b}: monthly counts as submitted -> {cnt(raw)};  births per day relative to trend -> '
        f'{cnt(met.loc[a:b, "peak_month"])}')

(out_dir / 'reviewer_checks.txt').write_text('\n'.join(LINES) + '\n')
