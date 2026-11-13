#!/usr/bin/env python3
"""
Figure 1 (composite, one page) of the revised manuscript.

  A, B  Double-plotted birthograms (Germany, France) on a common year axis: peak month of each
        year in a darker shade, annual acrophase overlaid, and the principal break in peak month
        (95% CI) in the margin.
  C     Annual acrophase, both countries, with fitted logistic transitions; bars above the data
        give the mid-point of each transition and its 95% CI.
  D     Annual peak-to-trough amplitude (% of trend), with centred 5-year moving averages.

Run birth_seasonality_analysis.py first (it writes the results/ tables this script reads).
Usage:  python figure1.py [data_dir] [results_dir] [output_stem]
Journal style: sans-serif about 8-9 pt at print size, no boxes, no gridlines, no top/right axes.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import birth_seasonality_analysis as bsa

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data')
res_dir = Path(sys.argv[2] if len(sys.argv) > 2 else 'results')
stem = sys.argv[3] if len(sys.argv) > 3 else 'Figure1'

# ------------------------------------------------------------------ style
INK, INK2, AXIS = '#0b0b0b', '#52514e', '#6f6e69'
COL = {'Germany': '#2a78d6', 'France': '#eb6834'}          # colour-blind-safe pair (validated)
TINT = {'Germany': '#7fb0ee', 'France': '#f4a888'}         # lighter steps of the same hues for area fills
MARK = {'Germany': 'o', 'France': '^'}                     # shape repeats the colour coding
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})
MM = 1 / 25.4
Y0, Y1 = 1946, 2024                                        # common year axis

# ------------------------------------------------------------------ data
tests = pd.read_csv(res_dir / 'break_tests.csv').set_index(['country', 'metric'])
fits = pd.read_csv(res_dir / 'phase_transition_fits.csv').set_index('country')
D = {}
for country, fn in bsa.FILES.items():
    df = bsa.load_births(data_dir / fn)
    met, _ = bsa.annual_metrics(df)
    disp = bsa.index_table(df, extrapolate_ends=True)       # all years, for display
    D[country] = dict(met=met, disp=disp)


def label(yr):
    return f'{yr}/{str(yr + 1)[2:]}'


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(174 * MM, 222 * MM))
gs_top = fig.add_gridspec(1, 2, left=0.075, right=0.955, top=0.962, bottom=0.375, wspace=0.27)
gs_bot = fig.add_gridspec(1, 2, left=0.075, right=0.955, top=0.275, bottom=0.048, wspace=0.24)


def birthogram(ax, country, letter):
    disp, met = D[country]['disp'], D[country]['met']
    x = np.arange(-1, 25)
    for yr in range(disp.index.min(), disp.index.max()):
        a, b = disp.loc[yr].values, disp.loc[yr + 1].values
        v = np.r_[(a - a.min()) / (a.max() - a.min()), (b - b.min()) / (b.max() - b.min())]
        v = np.pad(v, 1, mode='edge')                       # full-width first and last bars
        base = yr + 0.46
        ax.fill_between(x, base, base - 0.90 * v, step='mid', color=TINT[country], lw=0, zorder=2)
        for off, seg in ((0, a), (12, b)):                  # peak month of each year, full-strength colour
            m = int(np.argmax(seg))
            ax.fill_between([off + m - 0.5, off + m + 0.5], base, base - 0.90, color=COL[country], lw=0,
                            zorder=2.5)
    ax.axvline(11.5, color='white', lw=1.2, zorder=3)       # gap between year n and year n + 1
    # annual acrophase (months since 1 January; mid-January sits at x = 0)
    yrs = met.index.values
    ax.plot(met['phase'].values - 0.5, yrs, color=INK, lw=0.9, zorder=4, solid_capstyle='round')
    ax.plot(12 + met['phase'].values - 0.5, yrs - 1, color=INK, lw=0.9, zorder=4, solid_capstyle='round')
    # break in peak month and its reported 95% confidence interval, in the right margin
    t = tests.loc[(country, 'peak_month')]
    xb = 24.25
    ax.plot([xb, xb], [t.ci_low_reported + 0.5, t.ci_high_reported + 0.5], color=INK, lw=1.1, clip_on=False,
            solid_capstyle='butt', zorder=5)
    for yy in (t.ci_low_reported + 0.5, t.ci_high_reported + 0.5):
        ax.plot([xb - 0.28, xb + 0.28], [yy, yy], color=INK, lw=0.8, clip_on=False, zorder=5)
    ax.plot([xb], [t.last_year_before + 0.5], marker='<', ms=4.2, color=INK, mec='white', mew=0.5,
            clip_on=False, zorder=6, ls='none')
    ax.text(xb + 0.8, t.last_year_before + 0.5, label(int(t.last_year_before)), va='center', ha='left',
            fontsize=8, color=INK, clip_on=False, rotation=270)
    # axes
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(Y1 - 0.4, Y0 - 0.6)
    ax.set_xticks(range(24))
    ax.set_xticklabels(list('JFMAMJJASOND') * 2, fontsize=7.5)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_yticks(range(1950, 2021, 10))
    ax.set_yticks(range(1950, 2021, 5), minor=True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_bounds(Y0 - 0.5, Y1 - 0.5)
    for xc, s in ((5.5, 'Year n'), (17.5, 'Year n + 1')):
        ax.text(xc, -0.052, s, transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=8.5, color=INK)
    for xa, xb_ in ((-0.4, 11.3), (11.7, 23.4)):
        ax.plot([xa, xb_], [-0.044, -0.044], transform=ax.get_xaxis_transform(), color=AXIS, lw=0.6,
                clip_on=False)
    ax.set_title(f'{letter}', loc='left', fontweight='bold', fontsize=10, color=INK, pad=5, x=-0.085)
    ax.text(0.0, 1.012, country, transform=ax.transAxes, fontsize=9.5, color=INK, va='bottom', ha='left')


axA = fig.add_subplot(gs_top[0, 0])
axB = fig.add_subplot(gs_top[0, 1])
birthogram(axA, 'Germany', 'A')
birthogram(axB, 'France', 'B')
axA.set_ylabel('Year n')

# small key for the birthogram overlays (in the empty rows above the German series); the margin
# bracket (principal break in peak month, 95% CI) is labelled with its date and explained in the legend
ky = 1947.6
axA.fill_between([-0.3, 0.7], ky + 0.75, ky - 0.75, color=COL['Germany'], lw=0, zorder=6)
axA.text(1.1, ky, 'peak month', va='center', ha='left', fontsize=8, color=INK, zorder=6)
axA.plot([7.6, 9.6], [ky, ky], color=INK, lw=0.9, zorder=6)
axA.text(10.0, ky, 'acrophase', va='center', ha='left', fontsize=8, color=INK, zorder=6)


def year_label(mid):
    """1981.25 -> '1981'; 1990.5 -> '1990/91' (between the 1990 and 1991 values)."""
    y, frac = int(np.floor(mid)), mid - np.floor(mid)
    return f'{y}/{str(y + 1)[2:]}' if 0.375 <= frac <= 0.625 else str(int(round(mid)))


def interval(ax, country, mid, lo, hi, ylev, side, cap=0.17):
    """Estimate (marker) and 95% CI (capped bar) above the data, labelled with the estimate."""
    ax.plot([lo, hi], [ylev, ylev], color=COL[country], lw=1.3, solid_capstyle='butt', zorder=5)
    for x in (lo, hi):
        ax.plot([x, x], [ylev - cap, ylev + cap], color=COL[country], lw=1.3, solid_capstyle='butt', zorder=5)
    ax.plot([mid], [ylev], marker=MARK[country], ms=4.2, color=COL[country], mec='white', mew=0.6,
            ls='none', zorder=6)
    xt, ha = (lo - 1.2, 'right') if side == 'left' else (hi + 1.2, 'left')
    ax.text(xt, ylev, year_label(mid), ha=ha, va='center', fontsize=8, color=INK)


def series(ax, country, col):
    met = D[country]['met']
    ax.plot(met.index, met[col], ls='none', marker=MARK[country], ms=2.9, color=COL[country],
            mec='white', mew=0.35, zorder=3)


# ------------------------------------------------------------------ C: phase
axC = fig.add_subplot(gs_bot[0, 0])
tt = np.arange(Y0, Y1 + 0.01, 0.1)
for country in ('Germany', 'France'):
    series(axC, country, 'phase')
    f = fits.loc[country]
    s = f.duration_10_90 / bsa.LN81
    met = D[country]['met']
    tr = tt[(tt >= met.index.min()) & (tt <= met.index.max())]
    axC.plot(tr, f.lower_plateau + (f.upper_plateau - f.lower_plateau) / (1 + np.exp(-(tr - f.midpoint) / s)),
             color=COL[country], lw=1.4, zorder=4)
for country, side in (('Germany', 'left'), ('France', 'right')):
    f = fits.loc[country]
    interval(axC, country, f.midpoint, f.mid_ci_low, f.mid_ci_high, 9.7, side)
axC.text(1988, 10.32, 'Mid-point of transition (95% CI)', ha='center', va='center', fontsize=8, color=INK2)
axC.set_ylim(1.8, 10.7)
axC.set_yticks(range(2, 11))
axC.set_yticklabels([])
axC.set_yticks(np.arange(2.5, 10, 1), minor=True)
axC.set_yticklabels([bsa.MON3[int(m)] for m in np.arange(2.5, 10, 1)], minor=True)
axC.tick_params(axis='y', which='minor', length=0)
axC.spines['left'].set_bounds(2, 10)
axC.set_title('C', loc='left', fontweight='bold', fontsize=10, color=INK, pad=5, x=-0.105)
axC.text(0.0, 1.03, 'Timing of the seasonal peak (acrophase)', transform=axC.transAxes, fontsize=9.5,
         color=INK, va='bottom', ha='left')
handles = [Line2D([], [], color=COL[c], marker=MARK[c], ms=3.6, mec='white', mew=0.4, lw=1.4, label=c)
           for c in ('Germany', 'France')]
axC.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 0.80), frameon=False, fontsize=8.5,
           handlelength=2.0, borderaxespad=0.2, labelcolor=INK, labelspacing=0.35)

# ------------------------------------------------------------------ D: amplitude
axD = fig.add_subplot(gs_bot[0, 1])
for country in ('Germany', 'France'):
    series(axD, country, 'amplitude')
    ma = D[country]['met']['amplitude'].rolling(5, center=True).mean()      # centred 5-year moving average
    axD.plot(ma.index, ma.values, color=COL[country], lw=1.4, zorder=4, solid_capstyle='round')
axD.set_ylim(0, 26.2)
axD.set_yticks(range(0, 26, 5))
axD.spines['left'].set_bounds(0, 25)
axD.set_ylabel('% of trend')
axD.set_title('D', loc='left', fontweight='bold', fontsize=10, color=INK, pad=5, x=-0.105)
axD.text(0.0, 1.03, 'Peak-to-trough amplitude', transform=axD.transAxes, fontsize=9.5, color=INK,
         va='bottom', ha='left')
axD.legend(handles=handles[:2], loc='lower left', bbox_to_anchor=(0.0, 0.0), frameon=False, fontsize=8.5,
           handlelength=2.0, borderaxespad=0.3, labelcolor=INK, labelspacing=0.35)
for ax in (axC, axD):
    ax.set_xlim(Y0 - 1, Y1)
    ax.set_xticks(range(1950, 2021, 10))
    ax.set_xticks(range(1950, 2021, 5), minor=True)
    ax.spines['bottom'].set_bounds(Y0, Y1 - 1)
    ax.set_xlabel('Year')

fig.savefig(f'{stem}.pdf')
fig.savefig(f'{stem}.png', dpi=300)
fig.savefig(f'{stem}_600dpi.png', dpi=600)
from PIL import Image                                       # flatten to RGB for the journal's TIFF
with Image.open(f'{stem}_600dpi.png') as im:
    rgb = Image.new('RGB', im.size, 'white')
    rgb.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
    rgb.save(f'{stem}.tiff', dpi=(600, 600), compression='tiff_lzw')
Path(f'{stem}_600dpi.png').unlink()
print('wrote', stem + '.pdf/.png/.tiff')
