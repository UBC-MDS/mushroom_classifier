# Contributing to the Mushroom Classification Project

This document outlines how to propose a change to the Mushroom Classification project.

## Fixing Typos

Small typos or grammatical errors in documentation may be edited directly using the GitHub web interface, provided the changes are made in the **source** files.

-   **YES**: You edit a Python file in the `src/` directory or a markdown file such as `README.md`.
-   **NO**: You edit auto-generated files or rendered outputs like `.html` or `.ipynb_checkpoints`.

## Prerequisites

Before making a substantial pull request (PR), you should: 1. File an issue to describe the problem or enhancement you're addressing. 2. Ensure someone from the team agrees with the proposed change.

If you've identified a bug: - Create an associated issue and provide a minimal reproducible example (`reprex`) to illustrate the bug clearly.

## Pull Request Process

1.  **Branch Creation**: Create a separate Git branch for each pull request (PR). Name the branch descriptively, e.g., `fix-missing-values` or `add-onehotencoder`.
2.  **Style Guide**:
    -   For Python code, follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/).
    -   For documentation, ensure clarity, conciseness, and proper formatting.
3.  **Testing**:
    -   Write unit tests for any new features or bug fixes.
    -   Run all existing tests to ensure your changes do not break functionality.
4.  **Documentation**:
    -   Update relevant sections in the `README.md` or other documentation files if your changes affect project usage.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Attribution

These contributing guidelines were adapted from the [dplyr contributing guidelines](https://github.com/tidyverse/dplyr/blob/master/.github/CONTRIBUTING.md).

------------------------------------------------------------------------

Thank you for contributing to the Mushroom Classification project! Your help makes this project better.
