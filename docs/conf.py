# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'CapibaraGPT v2.0'
copyright = '2025, Anachroni s.coop'
author = 'Anachroni s.coop'

# The full version, including alpha/beta/rc tags
release = '2.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# -- Intersphinx configuration ----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- MyST configuration ----------------------------------------------------
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
    'html_admonition',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
] 