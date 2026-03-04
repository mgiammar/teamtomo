# TeamTomo

TeamTomo is a set of modular Python package for cryo-EM and cryo-ET for the modern scientific computing environment.

This unified repository contains the core TeamTomo data processing functionality under a single umbrella for better maintainability and cross-package development. File I/O packages are distributed separately and can be found under the [organizations repositories](https://github.com/orgs/teamtomo/repositories).

## Getting Started

If you want to contribute to TeamTomo, please look through the [CONTRIBUTING](CONTRIBUTING.md) guide for more info on adding, migrating, or updating packages. If you're just looking to use TeamTomo for your work, follow the installation instructions below.

### Installation for Developers

The development workspace depends on the [uv](https://github.com/astral-sh/uv) tool for Python environment/package management.

First, clone and navigate into the root of the repository:

```bash
git clone https://github.com/teamtomo/teamtomo.git
cd teamtomo
```

Then, create and activate a new virtual environment:

```bash
uv venv
source .venv/bin/activate
```

And finally sync with the repository to install packages

```bash
uv sync --all-extras --all-packages
```

Note that the `--all-extras` and `--all-packages` flags install the development and testing requirements for all sub-packages. More granular install options are possible, if your system requires it.

## List of Packages

TODO