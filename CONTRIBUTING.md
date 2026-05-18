# Contributing to Vedana

Thanks for being interested in contributing to Vedana! Bug reports, feature requests, documentation fixes, and code contributions are all welcome.

This file is the quick reference. The full guide — with the rationale behind each rule, deeper code-style notes, and testing patterns — lives in [`docs/contributing/`](docs/contributing/).

## TL;DR

1. **Open an issue first** for anything beyond a typo / small fix — to discuss design and avoid duplicate work.
2. Fork the repo, create a topic branch.
3. `uv sync`, hack, run the linters and tests.
4. Open a PR with a clear description.

## Ways to contribute

- **Bug reports** — open an issue using the `Bug report` template.
- **Feature requests** — `Feature request` template; describe the use case, not only the solution.
- **Documentation** — typos, inaccuracies, new guides. PRs against `docs/` are reviewed on the same footing as code.
- **Code** — see the dev setup below.
- **Examples** — your own domain / use case as a PR in `apps/` or as a separate repo + a link.
- **Discussions** — for open-ended ideas use [GitHub Discussions](https://github.com/epoch8/vedana/discussions) before filing an issue.

## Dev setup

Vedana is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). Python 3.12 is required.

```bash
git clone https://github.com/<your-username>/vedana
cd vedana

uv sync
uv run pre-commit install

# bring up infra for integration tests
docker compose -f apps/vedana/docker-compose.yml up -d db memgraph grist
```

More on local development — including which services are needed for which tests — is in [docs/getting-started/local-development.md](docs/getting-started/local-development.md) and [docs/contributing/testing.md](docs/contributing/testing.md).

### Running tests

```bash
# whole workspace
uv run pytest

# a single package
uv run pytest libs/vedana-core/tests
```

Integration tests need Memgraph and Postgres running; some tests use VCR cassettes for LLM responses (`tests/cassettes/`).

### Linters and type checks

```bash
uv run ruff check .
uv run ruff format .
uv run mypy libs/vedana-core/src libs/jims-core/src
```

CI runs all three on every PR and won't merge until they're green.

## Branch / commit / PR

### Branch naming

- `feature/...` — new functionality
- `fix/...` — bug fix
- `docs/...` — documentation-only changes
- `refactor/...` — refactoring without behaviour change
- `chore/...` — infra (CI, deps, tooling)

### Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) — *recommended, not enforced* for now. Examples:

```
feat(vedana-core): support custom vector store backends
fix(jims-api): correct 401 response format
docs(getting-started): clarify pgvector setup on Supabase
refactor(vedana-etl): split prepare_nodes into smaller functions
chore: bump litellm to 1.42
```

The scopes match package names: `jims-core`, `jims-api`, `jims-telegram`, `jims-widget`, `jims-tui`, `jims-backoffice`, `vedana-core`, `vedana-backoffice`, `vedana-etl`, or `docs` / `ci` / `deps` for cross-cutting changes.

### Pull request

Use the PR template (`.github/PULL_REQUEST_TEMPLATE.md`). Include:

- a link to the issue, if any;
- what changed and why;
- screenshots / examples for UI changes;
- breaking changes called out explicitly;
- a note in [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]` if user-visible.

Smaller, focused PRs go through faster than "everything-at-once" ones.

## Documentation and versioning

- Code and docs change together. If a PR changes behaviour, the matching `docs/` page changes in the same PR.
- The published docs at [vedana.tech/docs](https://vedana.tech/docs/) are built from this repo's `docs/` folder.
- **Versioning policy (target state):**
  - `latest` — built from `main`. Carries a "documents the development version — features may change before release" banner.
  - **Current stable** — the default users land on; no banner.
  - **Previous stable** — maintenance mode (~6 months); banner suggesting upgrade.
  - **Older** — out-of-date banner, dropped from build after a release or two.

## What we won't accept

- PRs with no description or context.
- Breaking public-API changes without explanation / migration notes.
- Large PRs that weren't discussed in an issue first.
- Disabling tests "because they're in the way" — fix them or explain why they're wrong.
- Secrets or real credentials committed (including inside test fixtures or VCR cassettes).

## What's especially appreciated

- Tests for new code (unit and/or integration).
- Docs updates that go with behaviour changes.
- Benchmarks alongside performance PRs.
- Evaluation reference answers alongside retrieval-quality PRs.

## Code of Conduct

Vedana follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Be respectful — constructive criticism is welcome, personal attacks aren't.

## License and contributor agreement

Vedana is licensed under [Apache License 2.0](LICENSE). By opening a PR you agree that your contribution is distributed under the same license. Apache 2.0 includes an explicit patent grant from contributors — you grant the project a license to any patents that read on your contribution.

## Contact

- **Bugs / features / questions:** [GitHub Issues](https://github.com/epoch8/vedana/issues) and [Discussions](https://github.com/epoch8/vedana/discussions).
- **Security issues:** see [SECURITY.md](SECURITY.md).
- **Commercial / enterprise:** [Epoch8](https://epoch8.com).

## Where to read more

- [docs/contributing/contributing.md](docs/contributing/contributing.md) — the full contributing guide.
- [docs/contributing/code-style.md](docs/contributing/code-style.md) — ruff/mypy config, typing, async patterns, logging, SQL/Cypher safety.
- [docs/contributing/testing.md](docs/contributing/testing.md) — unit vs integration vs cassette tests, fixtures, coverage targets.
- [docs/contributing/repository-structure.md](docs/contributing/repository-structure.md) — where things live and why.
