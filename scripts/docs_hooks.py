"""Serve the root AI reference files with the MkDocs site."""

from pathlib import Path
from shutil import copyfile


def on_post_build(config, **kwargs):
    root = Path(config.config_file_path).parent
    for filename in ("llms.txt", "llms-full.txt"):
        copyfile(root / filename, Path(config["site_dir"]) / filename)
