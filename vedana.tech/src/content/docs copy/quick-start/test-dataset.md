---
title: "Overview"
section: "Example Dataset"
order: 1
---

## Test data

Go to Grist, http://localhost:8484/o/docs/ 
Press Sing in

<img src="/images/docs/sign-in.png" alt="Sign In" width="800" class="center-image" />

You'll see three documents:
- Data
- Data Model
- Golden Dataset

<img src="/images/docs/grist.png" alt="Grist" width="800" class="center-image" />

### Data

We took data from LIMIT dataset which looks like this:

```json
{"_id": "Geneva Durben", "text": "Geneva Durben likes Quokkas, Tapirs, Spinach, ..."}
```

And decomposed it into three tables:
- Person (for example, "Geneva Durben")
- Interest (for example, "Quokkas")
- Link table: Person has Interest

<img src="/images/docs/anchor-person.png" alt="Person" width="500" class="center-image" />

<img src="/images/docs/anchor-interest.png" alt="Interest" width="500" class="center-image" />

<img src="/images/docs/link-person-interest.png" alt="Link" width="600" class="center-image" />

### Data Model

Data Model lives also lives in Grist in Data Model document:
- http://localhost:8484/o/docs/j6PTmqgw4caB/Data-Model/ 

Data Model is a description of our data for LLM.

Data Model consists of:
- anchors (entities)
- attributes 
- links

<img src="/images/docs/dm-anchors.png" alt="Link" width="600" class="center-image" />

<img src="/images/docs/dm-attributes.png" alt="Link" width="600" class="center-image" />

<img src="/images/docs/dm-links.png" alt="Link" width="600" class="center-image" />

Data model is passed to LLM context so LLM knows what data it operates with.

### Playbook (example queries)

Queries also live in Data Model document.
Queries are reasoning instructions for LLM.

<img src="/images/docs/dm-queries.png" alt="Link" width="600" class="center-image" />
