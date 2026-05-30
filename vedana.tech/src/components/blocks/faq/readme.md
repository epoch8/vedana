# FAQ

Semantic FAQ accordion component with `schema.org/FAQPage` markup.

Supports:

- structured SEO markup (`FAQPage`, `Question`, `Answer`)
- theme-controlled typography via CSS variables

---

## Architecture

The component is split into:

- `FAQ.astro` → wrapper (`FAQPage`)
- `FAQItem.astro` → single question (`Question` + `Answer`)
- `FAQ.module.css` → component styles

---

## Usage

```astro
---
import FAQ from "@/components/faq/FAQ.astro";
import FAQItem from "@/components/faq/FAQItem.astro";
---

<section class="section">
  <div class="container">
    <h2 class="text-center">FAQ</h2>

    <FAQ>
      <FAQItem question="Is Vedana a chatbot?">
        <p>No. It is a reasoning layer built on structured domain models.</p>
      </FAQItem>

      <FAQItem question="Do we need to restructure all data?">
        <p>No. We start with a focused slice of your domain.</p>
      </FAQItem>
    </FAQ>
  </div>
</section>