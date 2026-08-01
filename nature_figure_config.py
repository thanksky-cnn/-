
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("C:/Windows/Fonts/arial.ttf")
    fm.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
except:
    pass
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "SimHei", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})
# Override pyplot.savefig AND Figure.savefig to force 600 DPI
import builtins
_orig_plt_savefig = plt.savefig
def _forced_savefig(*args, **kwargs):
    kwargs['dpi'] = 600   # force 600 DPI even if caller passes dpi=200
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('facecolor', 'white')
    # Debug: print once
    if not hasattr(_forced_savefig, '_printed'):
        print(f'  [nature-figure] Forcing DPI to 600 (was: {kwargs.get("dpi", "default")})')
        _forced_savefig._printed = True
    return _orig_plt_savefig(*args, **kwargs)
plt.savefig = _forced_savefig

_orig_fig_savefig = plt.Figure.savefig
def _forced_fig_savefig(self, *args, **kwargs):
    kwargs['dpi'] = 600
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('facecolor', 'white')
    return _orig_fig_savefig(self, *args, **kwargs)
plt.Figure.savefig = _forced_fig_savefig
