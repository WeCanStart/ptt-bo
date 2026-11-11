import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY = '#8a8984'


def style():
    plt.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': '#0b0b0b', 'axes.titlesize': 11,
        'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
        'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 9.5,
        'lines.linewidth': 2.0, 'lines.markersize': 5, 'legend.frameon': False,
        'axes.prop_cycle': matplotlib.cycler(color=PAL),
    })


style()
