# AGENTS.md

- MUST pass `uvx ruff format` and `uvx ruff check` without errors or warnings.
- MUST adhere to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python code UNLESS enforced by Ruff.
- MUST add type annotations everywhere and not violate them.
- MUST inline functions with less than 25 lines and used in less than 3 places.
- MUST NOT roll your own; always use existing libraries if it is well-maintained and widely used.

- MUST adhere to [Standard Readme](https://raw.githubusercontent.com/RichardLitt/standard-readme/refs/heads/main/spec.md) for Markdown files.
- MUST NOT introduce how it works in README.md; only describe what it does and how to use it.

- MUST keep `src/stupidbench/task/` byte-identical to what it is scored on. It
  is a payload, not code this repository owns: reformatting it, or fixing a lint
  in it, changes the baseline and voids every result compared against it.
- MUST NOT weaken what keeps a credential in: this repository is public, so its
  artifacts, caches and logs are. A cell gets one token and one way out, and
  nothing leaves a runner that `stupidbench redact` has not been over.
- MUST NOT add a workflow trigger that a pull request can fire while a secret is
  in scope.
