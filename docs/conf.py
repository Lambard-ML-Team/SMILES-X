# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SMILES-X'
copyright = '2023, Guillaume Lambard, Ekaterina Gracheva'
author = 'Guillaume Lambard, Ekaterina Gracheva'
release = '2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Tell sphinx that the code is residing outside of the current docs folder
# sys.path.append(os.path.abspath('..')) 
# extensions = ['extname']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/SMILESX_logo.png"
toc_object_entries_show_parents='all'

html_theme_options = {
    "use_issues_button": True,
    "home_page_in_toc": True,
    "repository_url": "https://github.com/Lambard-ML-Team/SMILES-X",
    "repository_provider": "github",
    "use_repository_button": True,
    "use_source_button": True,
    "show_toc_level": 4,
    "path_to_docs": "../SMILESX",
    "show_navbar_depth": 4,
}
