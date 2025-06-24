# Makefile for Sphinx documentation
# CapibaraGPT-v2 Documentation Build System

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for CapibaraGPT-v2
clean:
	rm -rf $(BUILDDIR)/*

install:
	pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

livehtml:
	sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html

# Build all formats
all: html pdf

# Generate API documentation automatically
apidoc:
	sphinx-apidoc -o api ../capibara -f

# Check documentation for issues
check:
	$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck
	$(SPHINXBUILD) -b doctest $(SOURCEDIR) $(BUILDDIR)/doctest

# Generate documentation with coverage report
coverage:
	$(SPHINXBUILD) -b coverage $(SOURCEDIR) $(BUILDDIR)/coverage

# Development build (fast)
dev:
	$(SPHINXBUILD) -b html -E $(SOURCEDIR) $(BUILDDIR)/html

# Production build (with all checks)
prod: clean apidoc html check

# Deploy to GitHub Pages
deploy: prod
	@echo "Documentation built successfully!"
	@echo "Deploy the contents of _build/html to your web server" 