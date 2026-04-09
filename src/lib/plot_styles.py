import os
import matplotlib.pyplot as plt
import src.lib.globals as globals

def publication_style() -> None:
    styling_path = os.path.join(globals._lib_directory, "color.mplstyle")
    plt.style.use(styling_path)
    return None
