# datason marketing website

The public homepage is a hand-built static HTML/CSS/JavaScript page. It explains
datason through three real examples and links into the detailed MkDocs guides.
GitHub Pages publishes **one artifact**: the homepage at `/datason/`, documentation
at `/datason/docs/`, and redirects from previously published documentation URLs.

## Build and preview

From the repository root:

```bash
uv sync --locked --group docs
uv run python scripts/build_site.py
uv run python scripts/check_site.py
python -m http.server --directory site 8000
```

Visit `http://localhost:8000/`. Normal homepage and documentation links work here;
canonical URLs, old-doc redirects, and the custom 404 target the published GitHub
Pages address. For a project-path preview, serve a parent directory with a
`datason` symlink pointing to `site/`, then visit `/datason/`.

The build replaces its generated `site/` directory. If an older standalone
MkDocs build already occupies `site/`, remove that generated output first. The
script refuses to overwrite unmarked, nonempty directories. Read the Docs keeps
its existing documentation-only build; it does not host the marketing homepage.

## What to edit

- `index.html`: narrative, navigation, metadata, and the default example for
  visitors without JavaScript. Match the default code/output to `examples.json`.
- `examples.json`: Python code, verified expected JSON, descriptions, and links
  for each interactive tab. `check_site.py` runs these snippets against datason
  in fresh Python processes and checks the linked documentation headings.
- `assets/site.css` and `assets/site.js`: responsive layout, keyboard-accessible
  tabs, syntax highlighting, and clipboard behavior. No frontend build step,
  external script, analytics, or runtime CDN is required.
- `assets/mark.svg`: original brand mark; keep `docs/assets/mark.svg` in sync.
- `social-card.html`: editable 1200×630 social-image source. Serve `website/`,
  load this page in Chromium at 1200×630, wait for `document.fonts.ready`, then
  save a screenshot to `assets/social-card.png`. Only the PNG is deployed.

Manrope and Space Grotesk are self-hosted Latin variable fonts from Google Fonts.
Their SIL Open Font License notices are in `assets/fonts/`. Browser/system fonts
cover glyphs outside that subset.

Keep claims scoped to the current implementation. The page deliberately says
v2 **alpha**, distinguishes plain JSON from typed storage, and links to fidelity
and trust contracts. When a matching v2 release is published, update the release
label, installation command, and version FAQ together. Do not add unverified
benchmarks, customer logos, or usage figures.

## Validation and deployment

The Website and Docs workflow checks the Python snippets, generated AI reference,
strict MkDocs build, asset/heading links, canonical sitemap, and legacy routes.
Pull requests produce the `documentation` artifact for review. Merging to `main`
publishes the combined site through the existing GitHub Pages environment.

For visual changes, also check narrow phones, tablets, and desktop screens; tab
keyboard navigation; both copy buttons; native FAQ controls; reduced motion;
and the page with JavaScript disabled. Existing bookmarks retain their query
strings and fragments when redirected to `/docs/`.
