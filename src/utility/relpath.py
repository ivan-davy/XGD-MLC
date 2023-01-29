from os.path import join, dirname, realpath
from pathlib import Path


def relpath(relative_path):
    return join(realpath(join(dirname(__file__), '../..')), Path(relative_path))
