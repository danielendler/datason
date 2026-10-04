# Contributing to datason

For usage, start with the [documentation](https://danielendler.github.io/datason/).
For bugs, include a small nonsensitive reproduction, installed version,
Python/library versions, options, expected result, and the complete error in an
[issue](https://github.com/danielendler/datason/issues).

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use a
supported Python version (3.10+; the docs job uses 3.12). From a repository checkout:

```bash
uv sync --locked --group docs --extra numpy --extra pandas --extra pydantic
uv run pytest
uv run ruff check datason/ tests/
uv run ruff format --check datason/ tests/
```

Optional ML libraries are not required for core development. Tests requiring
uninstalled optional libraries skip. Keep changes focused and describe the
behavior and validation in the pull request. The project uses a plugin registry
for type handlers; see [Custom plugins](docs/plugins.md) for the extension contract.

## Documentation changes

Edit the source pages under `docs/` and the GitHub README. Examples should include
their imports, input, dependencies, and a check of the result. Explain output
policies and fidelity boundaries alongside the happy path. Update `mkdocs.yml`
when adding a page so it is discoverable.

```bash
uv run python scripts/check_doc_examples.py
uv run python scripts/sync_docs.py
uv run python scripts/sync_docs.py --check
uv run ruff check scripts/sync_docs.py scripts/docs_hooks.py scripts/check_doc_examples.py examples/langgraph_checkpoint.py
uv run ruff format --check scripts/sync_docs.py scripts/docs_hooks.py scripts/check_doc_examples.py examples/langgraph_checkpoint.py
uv run mkdocs build --strict
uv run mkdocs serve
```

`llms-full.txt` is generated from the guides; edit the guides and regenerate it.
`llms.txt` is the concise curated index. The MkDocs hook serves both files at the
site root. CI validates the generated reference, independent Python snippets,
formatting, and strict site build. Python snippets run as trusted repository code
in fresh processes and temporary directories, including examples in the README.

To check the optional framework example:

```bash
uv run --with langgraph==1.2.12 --with langgraph-checkpoint-sqlite==3.1.1 python examples/langgraph_checkpoint.py
uv run --with langgraph==1.2.12 --with langgraph-checkpoint-sqlite==3.1.1 python scripts/check_doc_examples.py --with-langgraph
```

Docs from `main` deploy to GitHub Pages after the build passes. A pull request
builds a downloadable `documentation` artifact without publishing it. Keep the
installation guidance explicit when development source is ahead of PyPI.
