# Security Policy

Thank you for helping keep Vedana and its users safe.

## Reporting a vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately through one of the following channels:

- **Preferred:** GitHub's [private vulnerability reporting](https://github.com/epoch8/vedana/security/advisories/new) (visible only to maintainers).
- **Email:** `<hello@epoch8.co>`.

Please include:

- a description of the issue and its impact;
- the affected version(s) / commit(s);
- steps to reproduce (proof-of-concept code is welcome);
- any suggested mitigation, if you have one.

## What to expect

| Stage | Target time |
| --- | --- |
| Initial acknowledgement of your report | within **3 business days** |
| Triage and confirmation | within **10 business days** |
| Fix and coordinated disclosure | within **90 days** of confirmation, sooner for severe issues |

We will keep you updated as the investigation progresses. If a public advisory is published, we will credit you by name (or anonymously, if you prefer).

## Supported versions

While Vedana is pre-1.0, only the latest released minor version on `main` receives security fixes. Once `1.0` is released, the policy below applies:

| Version | Status | Security fixes |
| --- | --- | --- |
| `main` (development) | Active development | Yes |
| Latest stable (`vX.Y`) | Current stable | Yes |
| Previous stable (`vX.Y-1`) | Maintenance (~6 months after next stable) | Yes, critical / high only |
| Older | End of life | No |

See [CONTRIBUTING.md](CONTRIBUTING.md#documentation-and-versioning) for the documentation-version policy.

## Scope

In scope:

- Vulnerabilities in code inside this repository (`libs/`, `apps/`, ETL pipelines).
- Default Docker / `docker-compose` configurations shipped in `apps/vedana/`.
- Documentation that misleads operators into insecure configurations.

Out of scope (please report upstream instead):

- Vulnerabilities in third-party dependencies — report to the upstream project (Memgraph, pgvector, LiteLLM, Reflex, Datapipe, Grist, etc.).
- Issues in self-hosted deployments that are not reproducible against the default configuration.
- Theoretical attacks against the LLM itself (prompt injection on a generic level) that aren't tied to a specific weakness in Vedana's pipeline. (Targeted prompt injection that bypasses Vedana's tool-call constraints or leaks data **is** in scope.)

## Safe-harbour for researchers

If you make a good-faith effort to comply with this policy, we will not pursue legal action and will work with you to understand and remediate the issue. Please give us reasonable time to fix the issue before any public disclosure.

## Hardening tips for operators

See [docs/operations/security.md](docs/operations/security.md) for the operator-side checklist (TLS termination, secret management, network isolation between Postgres / Memgraph / LLM provider, Grist access control, etc.).
