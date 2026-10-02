"""
Plot style, shared by every plot make_figures.py draws so they read as one set.

  * ONE typeface, Times. `Times New Roman` is used where it is installed;
    `Nimbus Roman` and `STIXGeneral` are metric-compatible stand-ins.
  * ONE text size (`FS`) for every tick, label and legend entry. Plots are
    drawn at the size they are meant to be placed.
  * ONE text colour, black. Colour carries data, never words: a series colour
    appears in the mark and in the legend swatch, and the text beside it stays
    black.
  * Title Case everywhere, through `cap()`, which leaves gene symbols,
    accessions and unit names alone.
  * No text written over a mark, and no titles beyond the one identifier a
    plot needs to be told from its siblings.
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- type ------------------------------------------------------------------
FS = 9                      # the one text size, in points
SERIF = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]

# --- colour ----------------------------------------------------------------
# Eight categorical slots in a fixed order (the order is the colour-vision
# safety mechanism), one sequential hue, one warm/cool diverging pair.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GROUP = {"case": "#eb6834", "control": "#2a78d6"}     # case warm, control cool; the card names the groups
INK = "#000000"                                        # every character in the figure
MUTED, GRID, SURFACE = "#8a8a85", "#d9d9d6", "#ffffff"

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_orange",
    ["#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb",
     "#f0efec", "#fbdcc9", "#f6b795", "#ef8a55", "#eb6834", "#b84a1f"])
for _cm in (SEQ, DIV):
    _cm.set_bad(GRID)

plt.rcParams.update({
    "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "figure.titlesize": FS,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3, "legend.columnspacing": 1.1, "legend.labelspacing": 0.4,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.2, "patch.linewidth": 0.7,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# A token that is a name rather than a word keeps its own casing: gene symbols,
# accessions, units and anything already carrying capitals inside it.
_NAME = re.compile(r"^[A-Za-z]*[A-Z0-9][A-Za-z0-9\-/+.']*$")


def _cap_word(w):
    if not w or _NAME.match(w) and any(c.isupper() or c.isdigit() for c in w[1:] + w[0].upper()) \
            and not w.islower():
        return w
    return "-".join(p[:1].upper() + p[1:] if p else p for p in w.split("-"))


def cap(s):
    """Title Case: every word capitalised, gene symbols and units untouched.
    'genes per nucleus' -> 'Genes Per Nucleus'; 'mRNA' and 'GAPDH' survive."""
    return " ".join(_cap_word(w) for w in str(s).split(" "))


def label(ax, x=None, y=None):
    if x is not None:
        ax.set_xlabel(cap(x))
    if y is not None:
        ax.set_ylabel(cap(y))


def tidy(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def legend(ax, handles=None, **kw):
    kw.setdefault("frameon", False)
    if handles is not None:
        return ax.legend(handles=handles, **kw)
    return ax.legend(**kw)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf", flush=True)
