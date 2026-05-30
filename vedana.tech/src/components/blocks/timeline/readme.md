# Timeline

Semantic timeline component for structured process steps with `schema.org/HowTo` markup.

Supports:

- layout: `horizontal` (default) / `vertical`
- lines: `true` / `false`  

---

## Props

| Prop         | Type                          | Default      |
|--------------|------------------------------|--------------|
| orientation  | `"horizontal" \| "vertical"` | `"horizontal"` |
| lines        | `boolean`                    | `true`       |

---

## Usage

### Horizontal (default)

```astro
---
import Timeline from "@/components/timeline/Timeline.astro";
import TimelineItem from "@/components/timeline/TimelineItem.astro";
---

<Timeline>
  <TimelineItem index="1" title="Discovery">
    Analyze domain and data sources.
  </TimelineItem>

  <TimelineItem index="2" title="Modeling">
    Build graph and reasoning rules.
  </TimelineItem>

  <TimelineItem index="3" title="Launch">
    Internal demo of working assistant.
  </TimelineItem>
</Timeline>
```

### Vertical without lines (parameters usage example)

```astro
---
import Timeline from "@/components/timeline/Timeline.astro";
import TimelineItem from "@/components/timeline/TimelineItem.astro";
---
<Timeline orientation="vertical" lines={false}>
  <TimelineItem index="1" title="Step One">
    Description.
  </TimelineItem>

  <TimelineItem index="2" title="Step Two">
    Description.
  </TimelineItem>
</Timeline>
```