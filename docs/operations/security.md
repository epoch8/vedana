---
title: Security
section: Operations
order: 5
---

# Security

> **Status: draft / skeleton.** This page is a checklist of security topics that need to be filled in before a public production launch. Each section lists the questions to answer and links to the relevant code. Sections still labelled "TODO" have not been validated against production reality yet — talk to the Vedana team before relying on them.

This document covers the security surface of a Vedana deployment: what data is stored, how it flows between components, who can read it, and how that surface is locked down. It is written for operators ("how do I run Vedana safely?") and integrators ("what guarantees can I give my users?").

## Threat model

What Vedana protects against:

- **TODO** — list the attacker profiles you actually care about (curious internal user / external attacker on the network / compromised LLM provider account / malicious chat user).
- **TODO** — list the assets (raw domain data in Grist / embeddings / thread events with user messages / LLM API keys).

Out of scope:

- **TODO** — note what Vedana does NOT defend against (e.g. host-level compromise, supply chain on `pip install`).

## Data classification

| Data                       | Where it lives                            | Sensitivity              | Notes                                                                  |
| -------------------------- | ----------------------------------------- | ------------------------ | ---------------------------------------------------------------------- |
| Domain content (anchors, attribute values) | Grist + Memgraph + pgvector | Depends on the domain    | Same content is replicated into all three stores during ETL.            |
| User messages              | Postgres (`thread_events.event_data`)     | **Often contains PII**   | Stored verbatim, including anything the user typed (names, emails, IDs).|
| Assistant answers          | Postgres (`thread_events.event_data`)     | Depends on the domain    | May echo back data from the graph.                                      |
| LLM/API keys               | `.env` / runtime env                      | Secret                   | Never log them; never check them into Grist.                            |
| Eval golden dataset        | Grist + Postgres (`eval_gds`)             | Same as domain content   | Often hand-curated, may include realistic PII.                          |
| Backoffice cookies/session | Caddy / Reflex                            | Auth material            | See [Backoffice auth](#backoffice-auth).                                |

## PII masking

**Status: TODO.** Vedana does not perform PII detection or masking out of the box. Before launch, decide:

- Should user messages be PII-scrubbed before being written to `thread_events`? Where in the pipeline does that hook live? (Possible places: a `comm.user_message` interceptor in the `ThreadController`; an OTel-based filter on outgoing LLM spans; pre-LLM message rewriting.)
- Should chunks of source documents be scrubbed before being embedded into pgvector?
- What is the redaction strategy — replace with `[REDACTED]`, hash, or drop the record?

Open question for the engineering team:

- [ ] Confirm whether any of the above is implemented in a downstream Epoch8 fork; if yes, link to the implementation here.

## Encryption

### At rest

- **Postgres / pgvector** — encryption depends on the host. Cloud-managed Postgres (Yandex/GCP/AWS RDS/Neon/Supabase) encrypts disks by default; self-hosted Postgres in Docker on a plain disk does not.
- **Memgraph** — same story: cloud-managed Memgraph encrypts disks; the dev container does not.
- **Grist** — Grist documents are SQLite files on disk; not encrypted by default. Self-hosted Grist users should rely on full-disk encryption.

### In transit

- Vedana components talk to each other over loopback by default. In a real deployment:
  - put Caddy / nginx / cloud LB in front of `jims-api`, `jims-widget`, and the backoffice with TLS termination;
  - use TLS-enabled Bolt for Memgraph (`bolt+s://`), TLS-enabled Postgres (`sslmode=require`), and HTTPS for Grist;
  - **TODO** — document the exact env vars and certificate paths used for each.

## Data retention

`thread_events` grows monotonically. There is **no built-in purge job**.

To decide before launch:

- How long do we keep raw user messages? (GDPR / 152-ФЗ typically expect either a defined retention or a user-driven delete.)
- Do we keep `event_data` forever and only purge anonymised metadata after N days, or purge whole rows?
- How do we honour a "delete my data" request — by `contact_id`, by `thread_id`, or by an external user identifier?

Recommended starter implementation (not in the repo yet):

```sql
-- nightly job, retention = 180 days; tune to your policy
DELETE FROM thread_events WHERE created_at < NOW() - INTERVAL '180 days';
```

**Open question for the engineering team:**

- [ ] Is there an established retention policy at Epoch8 for production Vedana installations? Document it here.

## 152-ФЗ / GDPR checklist

For the Russian-law (152-ФЗ) and GDPR-relevant deployments:

- [ ] **Data Processing Agreement (DPA)** — template DPA between the customer and Epoch8 (or whoever runs the Vedana instance). **TODO — link to the template.**
- [ ] **Lawful basis** — is processing covered by consent, contract, or legitimate interest? Document per customer.
- [ ] **Subject access rights** — how a user gets a copy / deletion of their data (see [Data retention](#data-retention)).
- [ ] **Sub-processor list** — at minimum the LLM provider (OpenAI / Anthropic / Vertex / OpenRouter) is a sub-processor. Embeddings model too.
- [ ] **Data residency** — confirm which region your Postgres / Memgraph / Grist live in. For 152-ФЗ this must include Russia.
- [ ] **Breach notification** — internal runbook + customer-facing SLA.

## Audit log for Cypher

Every Cypher query generated by the assistant is currently visible in the trace (see [Observability](../architecture/observability.md)) but is **not** stored as a first-class audit record.

To decide before launch:

- Do we promote `rag.cypher_*` spans to a separate `audit_log` table (Cypher text + thread_id + contact_id + timestamp)?
- Who has read access to that table? (Should be narrower than read access to `thread_events`.)
- Retention policy for the audit log (often longer than `thread_events`).

**TODO** — document the chosen approach once decided.

## Backoffice auth {#backoffice-auth}

The Reflex-based backoffice (`vedana-backoffice`) does not ship with built-in user management. Production deployments **must** front it with one of:

- Caddy + OAuth2 proxy (the default in `apps/vedana/docker-compose.yml` is a plain Caddy without auth — this is dev-only);
- a reverse proxy with mTLS;
- nginx with basic auth (acceptable only for internal tools).

The `jims-api` HTTP API supports:

- a static bearer token via `--api-key` (see [HTTP API → Authentication](../api/http-api.md#auth));
- Authentik OAuth via `--authentik-url` + `--authentik-app-slug` (both flags must be set together).

**TODO** — describe the canonical production deployment for the backoffice (which OAuth provider, which proxy).

## Reviewing test cassettes for secrets

Vedana tests use VCR-style cassettes (`tests/**/cassettes/`) that record real LLM responses. Before publishing the repo or merging an external PR:

- Run `git grep -nE "sk-[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,}|grist_[A-Za-z0-9]{20,}"` (or equivalent) across `tests/`.
- Confirm `pyproject.toml` does not ship any cassette by accident (only the public ones under `tests/data/cassettes/public/`, **TODO — verify the convention with the team**).
- For each cassette referencing an external provider, confirm the captured `Authorization` header was scrubbed.

**TODO** — automate this as a pre-commit / CI step. Until then, the manual review is the gate.

## Open items before launch

Tracked as a checklist:

- [ ] Decide PII masking strategy and where it hooks into the pipeline.
- [ ] Decide retention period and implement a purge job.
- [ ] Publish the DPA template.
- [ ] Decide audit-log scope for Cypher queries.
- [ ] Document the canonical backoffice auth setup.
- [ ] Automate the cassette secret scan in CI.
- [ ] Fill in every "TODO" above.
