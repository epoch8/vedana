---
title: Contributing Guide
section: Contributing
order: 1
---

# Contributing Guide

Welcome! Vedana is an open-source project, and we love contributions. This document explains how to contribute — from a typo fix to a large architectural feature.

## TL;DR

1. Open an issue before significant work — discuss the design.
2. Fork the repository, create a feature branch.
3. Run `uv sync`, run the tests.
4. Open a PR with a clear description.
5. Wait for review and discussion.

## Ways to contribute

- **Bug reports.** Open an issue tagged `bug`. Describe what you expected, what you got, how to reproduce.
- **Feature requests.** Issue tagged `enhancement`. Describe the use case and the proposed solution.
- **Pull requests.** Code, documentation, tests, usage examples.
- **Documentation.** Typos, inaccuracies, new guides — always welcome.
- **Examples.** An example of your domain / use case as a PR in `apps/` or as a separate repo + a link.
- **Discussions.** Use GitHub Discussions for ideas without creating an issue.

## Before doing major work

Open an issue and describe **what** you want to do and **why**. That:

- avoids duplicate work;
- gets you early design feedback;
- clarifies whether it's on the roadmap.

Without discussion, large PRs can wait for review or get rejected.

## Dev setup

```bash
git clone https://github.com/<your-username>/vedana
cd vedana

uv sync

# bring up infra for tests
docker compose -f apps/vedana/docker-compose.yml up -d db memgraph grist
```

Each package (`libs/jims-core`, `libs/vedana-core`, etc.) has its own `pyproject.toml` and `tests/`.

### Running tests

```bash
# tests of a specific package
cd libs/jims-core
uv run pytest

# or via the workspace
uv run pytest libs/vedana-core/tests
```

Some tests are integration; they require Memgraph and Postgres running.

### Pre-commit hooks

There's a `.pre-commit-config.yaml` at the root. Install:

```bash
uv run pre-commit install
```

The hooks run linters and formatters on every commit.

## Code style

See [Code Style](./contributing/code-style.md). In short:

- **ruff** for linting, line length 120 (see `[tool.ruff]` in `pyproject.toml`).
- **mypy** for type checking, namespace packages, `enable_incomplete_feature = ["NewGenericSyntax"]`.
- Type hints required for the public API.
- Docstrings — for functions whose semantics aren't obvious.

## Branch / commit / PR

### Branch naming

- `feature/...` — new functionality.
- `fix/...` — bug fix.
- `docs/...` — documentation-only changes.
- `refactor/...` — refactoring without behaviour change.
- `chore/...` — infrastructure changes (CI, deps).

### Commit messages

Conventional Commits:

```
feat(vedana-core): add support for custom vector store
fix(jims-api): correct 401 response format
docs: add troubleshooting guide for Memgraph
refactor(vedana-etl): split prepare_nodes into smaller functions
chore: bump litellm to 1.42
```

### PR description

Include:

- a link to the issue, if any;
- what you did and why;
- screenshots / examples, if the UI changes;
- breaking changes, if any;
- what you tested (tests, manual scenarios).

Small focused PRs go through faster than big "everything-at-once" ones.

## What we definitely won't accept

- A PR with no description.
- Changes that break the public API without explanation.
- Large PRs that weren't discussed in an issue.
- Code style violations without a reason.
- Removing tests "because they're in the way".
- Including secrets in commits.

## What's appreciated

- Tests for new code (unit and/or integration).
- Updating the docs when behaviour changes.
- Benchmarks if the PR is about performance.
- Reference answers for evaluation if the PR is about retrieval quality.

## Conduct

Be respectful. Vedana follows the [Contributor Covenant](https://www.contributor-covenant.org/). Constructive criticism is welcome; personal attacks are not.

## License

Vedana is distributed under the **[Apache License 2.0](https://github.com/epoch8/vedana/blob/main/LICENSE)**. By submitting a PR you agree your code is distributed under the same license. Apache 2.0 includes an explicit patent grant from contributors — you grant the project a license to any patents that read on your contribution.

## Contacting the team

- GitHub Issues / Discussions — primary channel.
- vedana.tech — product website.
- For commercial questions — contacts on the Epoch8 site.

## What's next

- [Code Style](./contributing/code-style.md)
- [Testing](./contributing/testing.md)
- [Repository Structure](./contributing/repository-structure.md)
