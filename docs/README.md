# Diffulex Documentation

The documentation is built with Sphinx and MyST Markdown.

From the repository root:

```bash
uv pip install -r docs/requirements.txt
python -m sphinx -W -b html docs docs/_build/html
```

The generated HTML site is written to `docs/_build/html`.

To preview it locally:

```bash
python -m http.server --directory docs/_build/html 8000
```

Then open `http://localhost:8000`.
