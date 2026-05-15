---
title: How to Add FAQ Entries
section: "Guides"

---

_This guide assumes you understand what FAQ is and when to use it. If not, read [FAQ Entries](https://vedana.tech/docs/preparing-data-for-vedana/questions-and-answers/) first._

## The FAQ Table

The FAQ table lives in **Grist → Data → FAQ**. It is predefined and ready to use — you only need to fill it in. It has two columns:

|Column|What to put here|
|---|---|
|`question`|A representative phrasing of the question, as your users are likely to ask it|
|`answer`|The exact answer to return — complete, self-contained, and ready to display|

Each row is one FAQ entry. There are no other required fields.

## Writing the Question

The `question` column is used for similarity matching, not exact matching. The system embeds both the stored question and the user's actual question, then compares them. This means you do not need to anticipate every possible phrasing, but you do need to choose a representative one.

Write the question the way your users most naturally ask it. If most users ask "what are your opening hours?" rather than "please provide store operating times", use the first phrasing. The closer your stored question is to the language users actually use, the more reliable the match will be.

A few things to avoid:

**Do not store multiple entries for the same intent.** If you have three FAQ rows all asking about return policies in slightly different words, the system will sometimes match one and sometimes another, and the answers may differ slightly. Pick the best phrasing and use it once.

**Do not make questions too broad.** A question like "tell me about your services" will match almost anything and pull up an answer in contexts where it is not appropriate. Keep questions specific enough that the answer is clearly the right one.

**Do not make questions too narrow.** A question phrased in highly technical internal language will not match how users actually ask. If your users say "refund" and your FAQ says "reimbursement procedure", the match threshold may not be crossed.

## Writing the Answer

The `answer` column contains the text that will be returned to the user. A few requirements:

**Make answers self-contained.** The answer will be returned without any surrounding context from the conversation. "See above for details" or "as mentioned earlier" make no sense when the answer is retrieved in isolation. Write each answer as if it is the only thing the user will see.

**Keep answers appropriately short.** FAQ is designed for short, canonical responses — not multi-paragraph explanations. If an answer requires extensive detail, consider whether a document chunk would be more appropriate. Long FAQ answers are harder to keep up to date and harder for users to scan.

**Do not add information that is not asked for.** If the question is "what are your opening hours?", the answer should be the opening hours — not a paragraph about how helpful your staff is. The answer should match the question's scope exactly.

## Configuring FAQ Behavior

The FAQ response behavior is governed by a prompt template in **Grist → Data Model → Queries**. A default template is included out of the box and instructs the assistant to prefer FAQ answers when relevant, use the stored answer as-is, and avoid supplementing it with invented information.

You do not need to modify this template for basic FAQ to work. If you need to adjust how FAQ answers are formatted — for example, to add a consistent sign-off or to control when FAQ takes precedence over other retrieval — this is where to make that change.

## Maintaining FAQ Over Time

FAQ entries become stale when the underlying facts change and the entries are not updated. A return policy that changed six months ago but still appears in a FAQ entry is worse than having no FAQ entry at all — it gives users confidently wrong information.

A few practices that help:

- Review FAQ entries any time a policy, price, or operational detail changes. Treat FAQ rows as documentation that needs to be kept current.

- Add new entries based on usage. Questions that appear repeatedly in user logs and are not being answered well by other retrieval methods are good candidates for FAQ entries.

- Remove entries that no longer apply rather than leaving them in place. Outdated entries compete with current ones and increase the chance of a bad match.

- Keep the total number of entries manageable. There is no hard limit, but as the FAQ table grows, entries with overlapping intent become harder to maintain and more likely to produce inconsistent matching. A curated set of 20–50 high-quality entries typically outperforms a sprawling set of 200 poorly maintained ones.
