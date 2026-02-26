# JIMS DeepChat Widget

[DeepChat](https://deepchat.dev/)-based embeddable chat widget for JimsApp backends. 

## Quick start

**1. Run the widget server** (point it at your JimsApp):

```bash
uv run jims-widget --app my_app:app --port 8090 --cors-origins "*"
```

**2. Embed on any page** — add one script tag:

```html
<script
  src="https://YOUR_HOST/static/jims-widget.js"
  data-server="https://YOUR_HOST"
></script>
```

Optional attributes on the `<script>` tag:

| Attribute        | Required | Default           | Description                          |
|------------------|----------|-------------------|-------------------------------------|
| `data-server`    | yes      | —                 | Widget backend origin (same host as script) |
| `data-contact-id`| no       | (anonymous)       | Persistent visitor identifier       |
| `data-thread-id` | no       | (new thread)      | Resume an existing thread           |
| `data-position`  | no       | `bottom-right`    | `bottom-right` \| `bottom-left`     |
| `data-open`      | no       | `false`           | `true` to start with panel open     |
| `data-title`     | no       | Chat assistant    | Header title text                   |
| `data-accent`    | no       | `#4f46e5`         | Accent colour (hex)                 |

## Use as a library

Mount the widget app in your own FastAPI (or ASGI) app:

```python
from jims_widget import create_widget_app

widget_app = create_widget_app(my_jims_app, cors_origins=["https://example.com"])
# mount or include widget_app as needed
```

A demo page is served at `/` when running the standalone server.
