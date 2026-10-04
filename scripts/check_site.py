#!/usr/bin/env python3
"""Verify the combined Pages artifact, link anchors, and executable homepage examples."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_URL = "https://danielendler.github.io/datason/"


class Page(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.code: dict[str, str] = {}
        self.current_code: str | None = None
        self.feed(source)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "code" and values.get("id"):
            self.current_code = values["id"]
            self.code[self.current_code] = ""
        for key in ("href", "src"):
            if values.get(key):
                self.links.append(values[key])
        if tag == "meta" and values.get("property") == "og:image":
            self.links.append(values["content"])

    def handle_data(self, data: str) -> None:
        if self.current_code:
            self.code[self.current_code] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "code":
            self.current_code = None


def local_target(site: Path, current: str, link: str) -> tuple[Path, str] | None:
    parsed = urlsplit(urljoin(PUBLIC_URL + current, link))
    if parsed.netloc != urlsplit(PUBLIC_URL).netloc or not parsed.path.startswith("/datason/"):
        return None
    relative = unquote(parsed.path.removeprefix("/datason/"))
    target = site / relative
    if parsed.path.endswith("/"):
        target /= "index.html"
    return target, unquote(parsed.fragment)


def check(site: Path) -> None:
    errors = []
    pages: dict[Path, Page] = {}
    for path in site.rglob("*.html"):
        pages[path] = Page(path.read_text(encoding="utf-8"))

    # Check every homepage link, image, module, and preload, including docs anchors.
    homepage = pages[site / "index.html"]
    for link in homepage.links:
        target = local_target(site, "", link)
        if target is None:
            continue
        path, anchor = target
        if not path.exists():
            errors.append(f"Homepage: missing {link}")
        elif anchor and path in pages and anchor not in pages[path].ids:
            errors.append(f"Homepage: missing heading {link}")

    # CSS font URLs and JS-fetched data are also first-party assets.
    css = (site / "assets/site.css").read_text()
    for link in re.findall(r"url\(['\"]?([^)'\"]+)", css):
        if not (site / "assets" / link).is_file():
            errors.append(f"Missing stylesheet asset: {link}")
    for name in ("llms.txt", "llms-full.txt", ".nojekyll", "404.html", "docs/search/search_index.json"):
        if not (site / name).is_file():
            errors.append(f"Missing deployed file: {name}")

    for doc in (site / "docs").rglob("index.html"):
        relative = doc.relative_to(site / "docs")
        if relative == Path("index.html"):
            continue
        redirect = site / relative
        canonical = PUBLIC_URL + "docs/" + relative.parent.as_posix() + "/"
        if not redirect.is_file() or canonical not in redirect.read_text():
            errors.append(f"Missing legacy redirect: {relative}")
        elif "location.search + location.hash" not in redirect.read_text():
            errors.append(f"Redirect drops bookmark state: {relative}")

    sitemap = ElementTree.parse(site / "sitemap.xml")  # noqa: S314 — local build output
    locations = {node.text for node in sitemap.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")}
    if PUBLIC_URL not in locations or PUBLIC_URL + "docs/" not in locations:
        errors.append("Sitemap must include homepage and docs home")
    if any(location != PUBLIC_URL and not location.startswith(PUBLIC_URL + "docs/") for location in locations):
        errors.append("Sitemap contains a legacy documentation URL")

    examples = json.loads((site / "examples.json").read_text())
    if homepage.code.get("demo-code", "").strip() != examples["api"]["code"]:
        errors.append("The no-JavaScript Python example differs from examples.json")
    if json.loads(homepage.code.get("demo-output", "null")) != examples["api"]["expected"]:
        errors.append("The no-JavaScript JSON output differs from examples.json")
    for name, demo in examples.items():
        # Execute reviewed repository snippets in isolated processes, just as the
        # docs checker does. No visitor input or downloaded code is executed.
        result = subprocess.run(  # noqa: S603 — trusted repository examples
            [sys.executable, "-c", demo["code"]], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        try:
            actual = json.loads(result.stdout)
        except ValueError:
            actual = None
        if result.returncode or actual != demo["expected"]:
            errors.append(f"Example {name} disagrees with the library: {result.stderr or result.stdout}")
        target = local_target(site, "", demo["guide"])
        if target is None or target[0] not in pages or target[1] not in pages[target[0]].ids:
            errors.append(f"Example {name} guide link is broken: {demo['guide']}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Site checks passed: homepage links/assets, {len(locations)} canonical URLs, redirects, and {len(examples)} live examples."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    check(parser.parse_args().site_dir.resolve())


if __name__ == "__main__":
    main()
