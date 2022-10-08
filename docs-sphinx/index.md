---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

[![](https://raw.githubusercontent.com/scikit-hep/awkward-1.0/main/docs-img/logo/logo-300px.png)](https://github.com/scikit-hep/awkward-1.0)

[![PyPI version](https://badge.fury.io/py/awkward.svg)](https://pypi.org/project/awkward)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/awkward)](https://github.com/conda-forge/awkward-feedstock)
[![Python 3.7‒3.10](https://img.shields.io/badge/python-3.7%E2%80%923.10-blue)](https://www.python.org)
[![BSD-3 Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Continuous integration tests](https://img.shields.io/azure-devops/build/jpivarski/Scikit-HEP/3/main?label=tests)](https://dev.azure.com/jpivarski/Scikit-HEP/_build)

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4341376.svg)](https://doi.org/10.5281/zenodo.4341376)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://awkward-array.readthedocs.io/)
[![Gitter](https://img.shields.io/badge/chat-online-success)](https://gitter.im/Scikit-HEP/awkward-array)

Awkward Array is a library for **nested, variable-sized data**, including arbitrary-length lists, records, mixed types, and missing data, using **NumPy-like idioms**.

Arrays are **dynamically typed**, but operations on them are **compiled and fast**. Their behavior coincides with NumPy when array dimensions are regular and generalizes when they’re not.

Quickstart
==========

Use the left-bar for tutorials (≡ button on mobile), click below for reference documentation, or [here for GitHub](https://github.com/scikit-hep/awkward-1.0). Note that the most complete documentation can be found in the {doc}`api-reference`: each function has a thorough docstring, usually with examples. Documentation for the compiled portion of Awkward Array can be found in the [C++ API Reference](https://awkward-array.readthedocs.io/en/latest/_static/index.html). User Guides can be found in {doc}`user-guide`.

Installation
------------

Awkward Array can be installed [from PyPI](https://pypi.org/project/awkward/) using pip:

```bash
pip install awkward
```

Most users will get a precompiled binary (wheel) for your operating system and Python version. If not, the above attempts to compile from source.

Getting help
------------

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/awkward-1.0/issues).
   * You can vote for issues by adding a "thumbs up" (👍) using the "smile/pick your reaction" menu on the top-right of the issue. See the [prioritized list of open issues](https://github.com/scikit-hep/awkward-1.0/issues?q=is%3Aissue+is%3Aopen+sort%3Areactions-%2B1-desc+reactions%3A%3E0+).
   * If you have a "How do I...?" question, start a [GitHub Discussion](https://github.com/scikit-hep/awkward-1.0/discussions) with category "Q&A".
   * Alternatively, ask about it on [StackOverflow with the [awkward-array] tag](https://stackoverflow.com/questions/tagged/awkward-array). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * To ask questions in real time, try the Gitter [Scikit-HEP/awkward-array](https://gitter.im/Scikit-HEP/awkward-array) chat room.

For developers
--------------

See Awkward Array's [GitHub page](https://github.com/scikit-hep/awkward-1.0) for more on the following.

   * [Installation for developers](https://github.com/scikit-hep/awkward-1.0#installation-for-developers).
   * [Continuous integration](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary) and [continuous deployment](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=4&_a=summary) are hosted by [Azure Pipelines](https://azure.microsoft.com/en-us/services/devops/pipelines/).
   * [Release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) (changelog) is hosted by [ReadTheDocs](https://readthedocs.org).
   * [Roadmap](https://github.com/scikit-hep/awkward-1.0#roadmap) of future releases.
   * [CONTRIBUTING.md](https://github.com/scikit-hep/awkward-1.0/blob/main/CONTRIBUTING.md) for technical information on how to contribute.
   * [Code of conduct](https://scikit-hep.org/code-of-conduct) for how we work together.
   * The [LICENSE](https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE) is BSD-3.

See also [papers and talks about Awkward Array](https://github.com/scikit-hep/awkward-1.0#papers-and-talks-about-awkward-array) and a [table of contributors](https://github.com/scikit-hep/awkward-1.0#acknowledgements).

