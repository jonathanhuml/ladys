from typing import Iterable, Optional, Union

from matplotlib import animation

from . import config


def saveanim(
    filename: str,
    anim: animation.Animation,
    extension: Optional[Union[str, Iterable[str]]] = None,
    **savefig_kwargs,
) -> None:

    if extension is None:
        extensions = ["gif"]
    elif isinstance(extension, str):
        extensions = [extension]
    else:
        extensions = extension

    for extension in extensions:
        if extension.startswith("."):
            fname_ext = filename + extension
        else:
            fname_ext = f"{filename}.{extension}"

        anim.save(config.experiment_results_path / fname_ext, **savefig_kwargs)
