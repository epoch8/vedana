---
title: "Test Data"
section: "Example Dataset"
order: 2
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

