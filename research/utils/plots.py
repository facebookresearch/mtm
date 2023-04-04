import pathlib
import subprocess

import matplotlib.pyplot as plt


def save_fig(fig: plt.Figure, filename: pathlib.Path, **kwargs) -> None:
    filename = pathlib.Path(filename).expanduser()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, **kwargs)
    print("Saved", filename)
    if filename.suffix == ".pdf":
        try:
            subprocess.call(["pdfcrop", str(filename), str(filename)])
        except FileNotFoundError:
            print("Install LaTeX to crop PDF outputs.")
