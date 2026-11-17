"""Figure style shared by all Arborist plots - matches the lab's SC poster (Helsinki Vision 2026):
Inter, dark-slate ink, morphology red, pale dome fill, dashed grey layer line, cell-type palette."""
import matplotlib as mpl

FONT = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
INK = "#1f2430"          # main text
MUTED = "#5b6270"        # secondary text
GRID = "#e4e7ed"         # recessive grid / axes
FRAME = "#c9ced8"

MORPH = "#c0392b"        # morphology accent (poster)
INVIVO = "#2c6fb7"       # in vivo accent (poster)
MERGE = "#6b3fa0"        # combined accent (poster)
CARD = "#fdf6f5"         # morphology card tint (poster)

DOME_FILL = "#e8ebf1"    # SC / sensory zone fill (poster dome)
DOME_LINE = "#7d8594"    # brain surface line
LAYER_LINE = "#9aa1ae"   # dashed layer border (CR border, poster SGS/SO line)
LAYER_TEXT = "#8a91a0"

# neurite colours: dendrite red / axon green, as in the fill projections and the poster labels.
# The green is shifted slightly towards teal so red vs green stays distinguishable for deuteranopes
# (validated: CVD dE 12.4, contrast >= 3:1 on white).
DEND = "#c0392b"
AXON = "#1a9c8a"
SOMA = "#5b6270"
AIS = MERGE

# poster cell-type palette (narrow, stellate, wide, horizontal, stellate 2, horizontal 2, pyramidal)
TYPE_PALETTE = ["#2aa8c9", "#4f63c6", "#3fae8e", "#8e5bbf", "#e0a823", "#e6643a", "#7cbf4a"]
TYPE_LABEL_COLORS = {"narrow-field": "#2aa8c9", "stellate": "#4f63c6", "wide-field": "#3fae8e",
                     "horizontal": "#8e5bbf", "marginal": "#e0a823", "pyramidal": "#7cbf4a"}


def apply_rc():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT + list(mpl.rcParams.get("font.sans-serif", [])),
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": FRAME,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.linewidth": 2.0,
        "svg.fonttype": "none",          # editable text in Illustrator / Inkscape
        "pdf.fonttype": 42,
    })
    # don't warn when Inter is not installed - DejaVu Sans is used then
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
