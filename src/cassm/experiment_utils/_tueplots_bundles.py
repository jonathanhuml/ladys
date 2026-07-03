from tueplots import figsizes, fonts, fontsizes

LATEX_PREAMBLE_ADDITIONS = r"\usepackage{amsmath, amssymb} \usepackage{bm} \renewcommand{\vec}[1]{{\bm{#1}}} \newcommand{\mat}[1]{{\bm{#1}}} \newcommand{\rvar}[1]{{\mathrm{#1}}} \newcommand{\rvec}[1]{{\bm{\mathrm{#1}}}} \newcommand{\rmat}[1]{{\bm{\mathrm{#1}}}}"


def jmlr(*, rel_width=1.0, nrows=1, ncols=1, family="serif"):
    size = figsizes.jmlr2001(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
    )
    font_config = fonts.jmlr2001_tex(family=family)
    fontsize_config = fontsizes.jmlr2001()
    font_config["text.latex.preamble"] += " " + LATEX_PREAMBLE_ADDITIONS

    return {**font_config, **size, **fontsize_config}


def neurips(*, usetex=True, rel_width=1.0, nrows=1, ncols=1, family="serif"):
    if usetex is True:
        font_config = fonts.neurips2022_tex(family=family)
    elif usetex is False:
        font_config = fonts.neurips2022(family=family)
    size = figsizes.neurips2022(rel_width=rel_width, nrows=nrows, ncols=ncols)
    base = 10
    fontsize_config = {
        "font.size": base - 1,
        "axes.labelsize": base - 1,
        "legend.fontsize": base - 2,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "axes.titlesize": base - 1,
    }
    font_config["text.latex.preamble"] += " " + LATEX_PREAMBLE_ADDITIONS
    return {**font_config, **size, **fontsize_config}


def icml(
    *, usetex=True, rel_width=1.0, nrows=1, ncols=1, family="serif", column="full"
):
    size = figsizes.jmlr2001(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
    )
    if usetex is True:
        font_config = fonts.icml2022_tex(family=family)
    elif usetex is False:
        font_config = fonts.icml2022(family=family)

    # if column == "full":
    #     size = figsizes.icml2022_full(nrows=nrows, ncols=ncols)
    # elif column == "half":
    #     size = figsizes.icml2022_half(nrows=nrows, ncols=ncols)
    # else:
    #     raise ValueError("Unknown column specification.")

    base = 10
    fontsize_config = {
        "font.size": base - 1,
        "axes.labelsize": base - 1,
        "legend.fontsize": base - 2,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "axes.titlesize": base - 1,
    }

    font_config["text.latex.preamble"] += " " + LATEX_PREAMBLE_ADDITIONS

    return {**font_config, **size, **fontsize_config}
