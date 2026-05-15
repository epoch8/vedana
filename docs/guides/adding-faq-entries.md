---
title: Adding FAQ Entries
section: Guides
order: 8
---

# Adding FAQ Entries

A step-by-step scenario for adding FAQ entries.

## 1. Decide whether it's an FAQ

Use FAQ when:

- the answer is short, fixed, and shouldn't vary;
- stable wording matters more than "smart" behaviour;
- the questions are templated and known in advance.

Don't use FAQ for:

- answers with specific values from the database ("how much is product X?");
- multi-step reasoning;
- explanations from documents.

## 2. Write good pairs

A good FAQ entry has:

- **question** — the most common phrasing your users will use;
- **answer** — concise, well-written, authoritative.

Good examples:

| question                                      | answer                                                                                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| "What's the return window?"                   | "You can return an item within 14 days of purchase at any Acme store. Details: https://acme.example.com/policy/refund." |
| "How can I contact you?"                      | "Support: support@acme.example.com, +370 5 123 4567. Hours: Mon–Fri 09:00–18:00."                                                          |
| "What do you do with my personal data?"       | "We process data per GDPR. Details — in the privacy policy: https://acme.example.com/privacy."                                          |

Bad examples:

- "MacBook Air price?" — that's a structured query; prices change.
- "Tell me everything about your services" — too generic, matches poorly via vector similarity.
- "Which is better — MacBook or Dell?" — subjective, requires reasoning.

## 3. Fill in the FAQ table in Grist

In **Grist Data doc > `Anchor_faq`** (the `Anchor_` prefix is hard-coded — see `vedana_core/data_provider.py:69`; there's currently no env var to change it):

| question                              | answer                                                                |
| ------------------------------------- | --------------------------------------------------------------------- |
| What's the return window?              | You can return an item within 14 days...                             |
| How can I contact you?                 | Support: ...                                                          |
| ...                                   | ...                                                                   |

## 4. Run ETL

Backoffice → ETL → **Run Selected**. After the run, the questions land in the graph as `faq` nodes with embeddings on the `question` field.

## 5. Verify in chat

Ask a question that should match:

> "Can I return a product?"

The answer should be exactly the body in `answer` (modulo formatting).

In Details: vector_text_search should hit the FAQ node with similarity > threshold.

## 6. Threshold fine-tuning

If FAQ matches too broadly (it answers FAQ for questions that should hit the graph) → raise the threshold for `faq.question` in `Anchor_attributes`.

If FAQ misses similar questions (the user phrases the same thing differently and gets nothing) → lower the threshold.

Starting range: **0.70–0.78** for FAQ entries (see the canonical table in [Tuning Embeddings & Thresholds](./tuning-embeddings.md#starting-values)). Tune via the golden dataset (see [Evaluation](../product/evaluation.md)).

## 7. When to write multiple variants of a question

Sometimes one answer has several canonical phrasings:

- "How do I return an item?"
- "What's the return window?"
- "Can I return an item?"

You can either:

1. Create **separate FAQ rows** with the same `answer` (more reliable, but duplicates).
2. Use one row with the most general phrasing and lower the threshold (risky — collects false positives).

Usually option 1 is better.

## Checklist

- [ ] The content really fits FAQ (not a structured query).
- [ ] The answer is authoritative, current, consistent with other support channels.
- [ ] The entry passed ETL without errors.
- [ ] In chat the FAQ triggers on the expected phrasings.
- [ ] The threshold is tuned — neither too broad nor too narrow.
- [ ] FAQ entries are reviewed and updated regularly (policies and contacts change!).

## Best practices

- **Maintain a small high-quality set.** 50 good FAQs work better than 500 noisy ones.
- **Review regularly.** Outdated FAQs are the main source of user distrust.
- **Link to documents.** FAQ for the canonical short answer + a link to a document for details.
- **Don't duplicate structured data.** Prices, stock, schedules are DB territory, not FAQ.

## What's next

- [FAQ concept](../data-ingestion/faq.md)
- [Tuning Embeddings](./guides/tuning-embeddings.md)
