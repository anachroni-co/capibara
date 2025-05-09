"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
# Añadimos la ruta al directorio raíz del proyecto
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'CapibaraGPT'
copyright = '2025, Anachron s.coop'
author = 'Marco Durán'
release = '2.1.6'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_rtd_theme',
]

# Optimizaciones de rendimiento
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
add_module_names = False
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Configuración de autodoc
autodoc_mock_imports = [
    'jax', 'flax', 'optax', 'orbax', 'tensorflow',
    'qiskit', 'cirq', 'pennylane'
]

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'flax': ('https://flax.readthedocs.io/en/latest/', None),
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Optimizaciones adicionales
nitpicky = False  # Desactivar advertencias estrictas
numfig = True  # Numeración automática de figuras
math_number_all = True  # Numerar todas las ecuaciones

# MathJax settings
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" 