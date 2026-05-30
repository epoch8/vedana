# Layout Contract

## 1. Responsibility Split

### Components

Components:

* **MUST NOT** define `max-width`
* **MUST NOT** define horizontal `padding`
* **MUST NOT** center themselves with `margin: 0 auto`
* **MUST NOT** control page-level width

Components are responsible **only for their internal structure and styling**.

They must remain reusable and layout-agnostic.

---

## 2. Container Owns Width

Page width and horizontal spacing are controlled exclusively by `.container`.

```css
.container {
  max-width: var(--container-wide);
  margin: 0 auto;
  padding: 0 var(--container-padding-x);
}
```

No component should override this behavior.

---

## 3. Correct Usage Pattern

All blocks must be used like this:

```html
<section class="section">
  <div class="container">
    <IconGrid />
  </div>
</section>
```

Where:

* `<section>` controls vertical rhythm
* `.container` controls width and horizontal spacing
* `IconGrid` controls only its internal layout

---

## 4. Why This Matters

If a component defines `max-width` or horizontal padding:

* It becomes harder to reuse
* It breaks page rhythm
* It cannot adapt to different layout contexts
* Spacing logic gets duplicated
* The system becomes inconsistent over time

Strict separation prevents layout entropy.

---

## 5. Allowed Exceptions

Internal width constraints are allowed only when:

* Limiting copy width inside a hero
* Constraining text inside a card
* Intentionally narrowing content for readability

These constraints must apply only to internal elements, not to the component wrapper itself.

---

## 6. Vertical Rhythm

Vertical spacing must be controlled by:

* `section`
* spacing tokens (`--space-*`)
* layout wrappers

Components should not introduce arbitrary vertical margins that affect page rhythm.

---

## Core Principle

> Components define structure.
> Container defines width.
> Section defines vertical rhythm.
