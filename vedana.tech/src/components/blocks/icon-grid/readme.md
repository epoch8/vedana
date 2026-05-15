# IconGrid

Responsive icon-based grid component for highlighting benefits, capabilities, or constraints.

Supports:

- flexible layout (2–N items)
- inline SVG icons via slot
- optional title and HTML content
- fully token-driven styling (spacing, color, typography)
- auto-responsive grid (`auto-fit`)
- zero JavaScript

---

## Architecture

The component is split into:

- `IconGrid.astro` → wrapper
- `IconGridItem.astro` → single card with icon
- `IconGrid.module.css` → component styles (token-based)

---

## Props

### IconGrid

| Prop | Type | Default |
|------|------|---------|
| id | `string` | `"icon-grid"` |

---

### IconGridItem

| Prop | Type | Default |
|------|------|---------|
| title | `string` | `undefined` |
| text | `string` (HTML) | `undefined` |

> Icon is passed via `<slot name="icon" />`

---

## Usage

```astro
---
import IconGrid from "@/components/blocks/icon-grid/IconGrid.astro";
import IconGridItem from "@/components/blocks/icon-grid/IconGridItem.astro";
---

<section class="section">
  <div class="container">
    <h2 class="text-center">Why Precision Matters</h2>

    <IconGrid>
      <IconGridItem
        title="Rules & Exceptions Matter"
        text="Policies, regulations, eligibility criteria, and approval workflows include nuanced edge cases."
      >
        <svg
          slot="icon"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M12 2l7 4v6c0 5-3.5 8.5-7 10-3.5-1.5-7-5-7-10V6l7-4z"/>
        </svg>
      </IconGridItem>

      <IconGridItem
        title="Exact Identifiers Required"
        text="SKUs, legal references, and clause numbers must be precise — never hallucinated."
      >
        <svg
          slot="icon"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M20 12V7a2 2 0 0 0-2-2h-5l-5 5 5 5h5a2 2 0 0 0 2-2v-1"/>
          <path d="M8 7h.01"/>
        </svg>
      </IconGridItem>

      <IconGridItem
        title="Errors Are Expensive"
        text="Mistakes carry legal, financial, operational, or safety consequences."
      >
        <svg
          slot="icon"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M12 9v4"/>
          <path d="M12 17h.01"/>
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.72 3h16.92a2 2 0 0 0 1.72-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        </svg>
      </IconGridItem>
    </IconGrid>
  </div>
</section>
```

## Using Lucide Icons

Icons in this component are typically sourced from:

👉 https://lucide.dev

### How to use

1. Open an icon on lucide.dev  
2. Click **“Copy SVG”**  
3. Paste the `<svg>...</svg>` markup directly inside `IconGridItem`  
4. Add `slot="icon"` to the `<svg>` element  

---

## What Is `<path d="...">`?

The `d` attribute inside `<path>` defines the vector shape.

It contains drawing commands (move, line, arc, close path) that describe the icon geometry.

You do **not** need to edit this manually.  
Lucide generates these paths.

Just copy the SVG as-is.

---

## Styling Notes

- Always use `stroke="currentColor"` so the icon inherits color from CSS.
- Recommended `stroke-width`: `1.5` for a more refined appearance.
- Avoid mixing multiple icon libraries to preserve visual consistency.
- Do not use PNG icons — keep everything inline SVG.

---

## Design Notes

- Icon circle background and spacing are controlled via design tokens.
- SVG scales automatically via spacing tokens.
- Intended for marketing and platform pages where clarity and emphasis matter.