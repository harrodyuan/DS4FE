"""Render the README hero animation: 523 daily SPX IV surfaces with level/skew tickers.

Run from the repo root:  python experiments/iv_surface/make_readme_gif.py
Output: figures/iv_surface_dynamics.gif
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

S = pd.read_parquet('data/iv/surfaces.parquet')
grid = json.load(open('data/iv/surface_grid.json'))
MATS, MONEY = grid['mats_days'], grid['moneyness']
cube = S.values.reshape(len(S), len(MATS), len(MONEY)) * 100.0      # vol points
dates = S.index

atm30 = cube[:, 0, MONEY.index(1.0)]
skew30 = cube[:, 0, MONEY.index(0.95)] - cube[:, 0, MONEY.index(1.05)]

BG, FG, GRIDC = '#0d1117', '#c9d1d9', '#21262d'                     # GitHub dark palette
ACC1, ACC2 = '#58a6ff', '#f78166'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'text.color': FG, 'axes.edgecolor': '#30363d', 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'font.size': 9, 'axes.grid': True,
    'grid.color': GRIDC, 'grid.linewidth': 0.6,
})

M, T = np.meshgrid(MONEY, MATS)
ZMIN, ZMAX = np.floor(cube.min()) - 1, np.ceil(cube.max()) + 1

fig = plt.figure(figsize=(10.8, 4.5), dpi=80)
ax3 = fig.add_axes([-0.04, 0.0, 0.56, 0.94], projection='3d')
axl = fig.add_axes([0.585, 0.575, 0.385, 0.30])
axs = fig.add_axes([0.585, 0.13, 0.385, 0.30])

axl.plot(dates, atm30, color=ACC1, lw=0.9)
axs.plot(dates, skew30, color=ACC2, lw=0.9)
axl.set_ylabel('30d ATM vol (%)', fontsize=8)
axs.set_ylabel('30d skew (vol pts)', fontsize=8)
axl.set_title('level: how expensive options are', fontsize=8.5, color=FG, loc='left')
axs.set_title('skew: the price of crash protection', fontsize=8.5, color=FG, loc='left')
for ax_ in (axl, axs):
    ax_.tick_params(labelsize=7)
dotl, = axl.plot([], [], 'o', color='white', ms=5, zorder=5)
dots, = axs.plot([], [], 'o', color='white', ms=5, zorder=5)
curl = axl.axvline(dates[0], color='white', lw=0.6, alpha=0.45)
curs = axs.axvline(dates[0], color='white', lw=0.6, alpha=0.45)

fig.text(0.045, 0.945, 'SPX implied volatility surface', fontsize=13, weight='bold', color='white')
fig.text(0.045, 0.895, 'rebuilt daily from raw option quotes: put-call parity -> Black-76 -> 7x9 grid',
         fontsize=8, color=FG)
datetxt = fig.text(0.045, 0.82, '', fontsize=11, color=ACC1, family='monospace', weight='bold')


def style3d():
    ax3.set_facecolor(BG)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        try:
            axis._axinfo['grid']['color'] = (1, 1, 1, 0.07)
        except Exception:
            pass
    ax3.set_xlabel('moneyness K/F', fontsize=8, labelpad=2)
    ax3.set_ylabel('tenor (days)', fontsize=8, labelpad=2)
    ax3.set_zlabel('IV (%)', fontsize=8, labelpad=2)
    ax3.tick_params(labelsize=7, pad=0)
    ax3.set_zlim(ZMIN, ZMAX)


def update(i):
    ax3.clear()
    style3d()
    ax3.plot_surface(M, T, cube[i], cmap='magma', vmin=ZMIN, vmax=ZMAX,
                     rstride=1, cstride=1, linewidth=0.25,
                     edgecolor=(1, 1, 1, 0.15), antialiased=True)
    ax3.view_init(elev=24, azim=-58 + 7 * np.sin(2 * np.pi * i / len(dates)))
    datetxt.set_text(dates[i].strftime('%Y-%m-%d'))
    dotl.set_data([dates[i]], [atm30[i]])
    dots.set_data([dates[i]], [skew30[i]])
    curl.set_xdata([dates[i], dates[i]])
    curs.set_xdata([dates[i], dates[i]])


frames = np.arange(0, len(dates), 2)
ani = FuncAnimation(fig, update, frames=frames)
out = 'figures/iv_surface_dynamics.gif'
ani.save(out, writer=PillowWriter(fps=14))
print(f'{out}: {os.path.getsize(out) / 1e6:.1f} MB, {len(frames)} frames')
