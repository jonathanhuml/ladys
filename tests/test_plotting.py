from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ladys.plotting import (
    model_color,
    model_label,
    model_marker,
    plot_context,
    plot_style,
    save_figure,
)


def test_plot_style_returns_uniform_rcparams():
    style = plot_style(nrows=2, ncols=2, rel_width=0.8, height_scale=1.4)

    assert style["text.usetex"] is False
    assert "figure.figsize" in style
    assert "axes.prop_cycle" in style
    assert float(style["figure.figsize"][1]) > float(style["figure.figsize"][0]) * 0.5
    assert model_color("gpfa") != model_color("kalman")
    assert model_marker("gpfa") == "^"
    assert model_label("bgpfa") == "bGPFA"
    assert model_label("ilqr_vae") == "iLQR-VAE"


def test_plot_context_can_write_png(tmp_path: Path):
    path = tmp_path / "figure.png"

    with plot_context(nrows=1, ncols=1):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color=model_color("gpfa"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        save_figure(fig, path)
        plt.close(fig)

    assert path.exists()
    assert path.stat().st_size > 0
