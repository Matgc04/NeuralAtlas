import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors

def save_master_colorbar(filename, color_map, width_px=256):
    # Target: slim horizontal bar that matches image width
    dpi = 100
    fig_w = width_px / dpi          # inches
    fig_h = 0.32                     # thin strip
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.45, top=0.95)

    cmap = plt.get_cmap(color_map)
    norm = colors.Normalize(vmin=0, vmax=1)

    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        ticks=[0, 0.5, 1.0],
    )
    ax.tick_params(labelsize=7, length=2, pad=1)

    plt.savefig(filename, format="webp", dpi=dpi, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    save_master_colorbar("interpretability-viewer/public/outputs/master_colorbar_jet.webp", "jet")