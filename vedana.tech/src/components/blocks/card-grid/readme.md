# CardGrid

Responsive card grid component for structured content blocks.

Supports:

* flexible layout (2–N items)
* optional title, image, and HTML content
* optional full-card link behavior
* optional footer button
* semantic `schema.org/ItemList` markup
* theme-controlled styling via design tokens
* auto-responsive grid (`auto-fit`)

---

## Architecture

The component is split into:

* `CardGrid.astro` → wrapper (`ItemList`)
* `CardGridItem.astro` → single card (`ListItem`)
* `CardGrid.module.css` → component styles

---

## Props

### CardGrid

| Prop | Type     | Default     |
| ---- | -------- | ----------- |
| id   | `string` | `undefined` |

---

### CardGridItem

| Prop        | Type            | Default        |
| ----------- | --------------- | -------------- |
| title       | `string`        | `undefined`    |
| image       | `string`        | `undefined`    |
| imageAlt    | `string`        | `""`           |
| text        | `string` (HTML) | `undefined`    |
| href        | `string`        | `undefined`    |
| buttonLabel | `string`        | `"Learn more"` |
| position    | `number`        | `undefined`    |

---

## Behavior

* If `href` is provided, the entire card becomes clickable.
* If `href` is not provided, the card renders as a static block.
* The footer button is rendered only when `href` exists.
* Cards are layout-agnostic (no `max-width`, no outer padding).
* Layout width and spacing are controlled externally via `.container` and `.section`.

---

## Usage

```astro
---
import CardGrid from "@/components/blocks/card-grid/CardGrid.astro";
import CardGridItem from "@/components/blocks/card-grid/CardGridItem.astro";
---

<section class="section">
  <div class="container">
    <CardGrid id="approach">

      <CardGridItem
        position={1}
        title="Generic LLM Chatbots"
        image="/images/llm.png"
        imageAlt="Abstract neural network with chaotic glowing nodes representing probabilistic AI behavior"
        text={`
          <ul>
            <li>Retrieve something that looks relevant</li>
            <li>Produce plausible outputs under uncertainty</li>
          </ul>
        `}
      />

      <CardGridItem
        position={2}
        title="Vedana's Approach"
        image="/images/graph.png"
        imageAlt="Structured knowledge graph with verified interconnected nodes representing deterministic reasoning"
        text={`
          <ul>
            <li>AI is the interface — not the brain</li>
            <li>Structured knowledge and deterministic logic</li>
            <li>Grounded, verifiable reasoning</li>
          </ul>
        `}
      />

    </CardGrid>
  </div>
</section>
```

---

## Layout Contract

CardGrid follows the global layout contract:

* Components do not define `max-width`
* Components do not define horizontal pa
