#!/usr/bin/env python3
"""Build the marketing homepage and MkDocs as one GitHub Pages artifact."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
SITE_URL = "https://danielendler.github.io/datason/"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def redirect_page(target: str) -> str:
    """Keep old bookmarks, query strings, and heading fragments working."""
    escaped = html.escape(target, quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Documentation moved · datason</title>
<link rel="canonical" href="{escaped}">
<meta name="robots" content="noindex">
<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>
</head><body><p>The documentation has moved. <a href="{escaped}">Continue to datason docs</a>.</p>
<noscript><meta http-equiv="refresh" content="0;url={escaped}"></noscript>
</body></html>
"""


def build(output: Path) -> None:
    # This command replaces its output. Restrict it to a dedicated repository
    # directory so a typo cannot remove source files or an unrelated checkout.
    output = output.resolve()
    reserved = {"docs", "website", "scripts", "examples", "tests", "datason", "test", "dist"}
    if output.parent != ROOT or output.name.startswith(".") or output.name in reserved:
        raise SystemExit("Use a dedicated output directory directly inside the repository, such as site/.")
    if output.exists():
        if not (output / ".datason-site").exists() and any(output.iterdir()):
            raise SystemExit(f"Refusing to replace unmarked directory {output}; choose a new output directory.")
        shutil.rmtree(output)
    output.mkdir()
    (output / ".datason-site").touch()
    subprocess.run(  # noqa: S603 — fixed local build command
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(output / "docs")],
        cwd=ROOT,
        check=True,
    )
    source = ROOT / "website"
    for name in ("index.html", "404.html", "examples.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copytree(source / "assets", output / "assets")
    for name in ("llms.txt", "llms-full.txt"):
        shutil.copyfile(ROOT / name, output / name)
    (output / ".nojekyll").touch()

    redirects = 0
    for page in (output / "docs").rglob("index.html"):
        relative = page.relative_to(output / "docs")
        if relative == Path("index.html"):
            continue  # The old documentation home is now the marketing home.
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = SITE_URL + "docs/" + relative.parent.as_posix() + "/"
        destination.write_text(redirect_page(target), encoding="utf-8")
        redirects += 1

    # One sitemap lists the canonical homepage and all canonical docs pages.
    ElementTree.register_namespace("", SITEMAP_NS)
    sitemap = ElementTree.parse(output / "docs" / "sitemap.xml")  # noqa: S314 — local MkDocs output
    homepage = ElementTree.Element(f"{{{SITEMAP_NS}}}url")
    ElementTree.SubElement(homepage, f"{{{SITEMAP_NS}}}loc").text = SITE_URL
    sitemap.getroot().insert(0, homepage)
    sitemap.write(output / "sitemap.xml", encoding="utf-8", xml_declaration=True)
    with gzip.open(output / "sitemap.xml.gz", "wb") as compressed:
        compressed.write((output / "sitemap.xml").read_bytes())
    (output / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n")
    print(f"Built {output}: homepage, docs, and {redirects} legacy documentation redirects.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    build(parser.parse_args().site_dir)


if __name__ == "__main__":
    main()
