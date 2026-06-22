# General information about the project.
project = "Diffulex"
author = "Diffulex Contributors"
copyright = f"2025-2026, {author}"

# Version information.
import re

with open("../pyproject.toml", "r", encoding="utf-8") as f:
    content = f.read()
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        version = match.group(1)
    else:
        version = "0.1.0"  # fallback
release = version

extensions = [
    "sphinx_tabs.tabs",
    "sphinxcontrib.httpdomain",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_reredirects",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autodoc_typehints = "description"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

redirects = {"get_started/try_out": "../index.html#getting-started"}

language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/*libinfo*",
    "**/*version*",
]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"
templates_path = []
html_static_path = ["_static"]
html_css_files = ["custom.css"]
footer_copyright = "© 2025-2026 Diffulex"
footer_note = " "

html_theme_options = {
    "light_logo": "img/logo-v2.png",
    "dark_logo": "img/logo-v2.png",
}

header_links = [
    ("Home", "https://github.com/SJTU-DENG-Lab/Diffulex"),
    ("Github", "https://github.com/SJTU-DENG-Lab/Diffulex"),
]

html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "SJTU-DENG-Lab",
    "github_repo": "Diffulex",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
}
