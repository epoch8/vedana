/**
 * JIMS Chat Widget — embeddable snippet.
 *
 * Usage:
 *   <script
 *     src="https://YOUR_HOST/static/jims-widget.js"
 *     data-server="https://YOUR_HOST"
 *     data-contact-id="optional-visitor-id"
 *   ></script>
 *
 * Attributes (on the <script> tag):
 *   data-server      — Widget backend origin (required)
 *   data-contact-id  — persistent visitor identifier (optional)
 *   data-thread-id   — resume an existing thread (optional)
 *   data-intro-message — initial AI greeting shown on load (optional)
 *   data-position    — "bottom-right" (default) | "bottom-left"
 *   data-open        — "true" to start expanded
 *   data-title       — chat header title (default "AI Assistant")
 *   data-accent      — accent hex colour (default "#4f46e5")
 */
(function () {
  "use strict";

  const scriptTag = document.currentScript;
  const cfg = {
    server: scriptTag.getAttribute("data-server") || "",
    contactId: scriptTag.getAttribute("data-contact-id") || "",
    threadId: scriptTag.getAttribute("data-thread-id") || "",
    introMessage: scriptTag.getAttribute("data-intro-message") || "",
    position: scriptTag.getAttribute("data-position") || "bottom-right",
    open: scriptTag.getAttribute("data-open") === "true",
    title: scriptTag.getAttribute("data-title") || "AI Assistant",
    accent: scriptTag.getAttribute("data-accent") || "#4f46e5",
  };

  if (!cfg.server) {
    console.error("[jims-widget] data-server attribute is required.");
    return;
  }

  const DEEP_CHAT_CDN =
    "https://cdn.jsdelivr.net/npm/deep-chat@2.4.1/dist/deepChat.bundle.js";

  /* ---- styles ---- */
  const style = document.createElement("style");
  style.textContent = `
    #jims-widget-fab {
      position: fixed;
      ${cfg.position === "bottom-left" ? "left" : "right"}: 24px;
      bottom: 24px;
      width: 56px; height: 56px;
      border-radius: 50%;
      background: ${cfg.accent};
      color: #fff;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 99999;
      transition: transform .2s, opacity .2s;
    }
    #jims-widget-fab:hover { transform: scale(1.08); }
    #jims-widget-fab svg { width: 26px; height: 26px; fill: currentColor; }

    #jims-widget-panel {
      position: fixed;
      ${cfg.position === "bottom-left" ? "left" : "right"}: 24px;
      bottom: 96px;
      width: 400px;
      height: 560px;
      max-height: calc(100vh - 120px);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 12px 40px rgba(0,0,0,0.18);
      z-index: 99999;
      display: flex;
      flex-direction: column;
      background: #fff;
      transition: opacity .25s, transform .25s;
    }
    #jims-widget-panel[data-closed] {
      opacity: 0;
      pointer-events: none;
      transform: translateY(20px) scale(0.96);
    }

    #jims-widget-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 16px;
      background: ${cfg.accent};
      color: #fff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: .95rem;
      font-weight: 600;
    }
    #jims-widget-header button {
      background: none; border: none; color: #fff; cursor: pointer;
      font-size: 1.3rem; line-height: 1; padding: 0 2px;
    }

    #jims-widget-panel deep-chat {
      flex: 1 1 auto;
      min-height: 0;
      width: 100%;
      height: 100%;
      max-width: 100%;
      border-radius: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    @media (max-width: 480px) {
      #jims-widget-panel {
        width: calc(100vw - 16px);
        height: calc(100vh - 80px);
        ${cfg.position === "bottom-left" ? "left" : "right"}: 8px;
        bottom: 72px;
        border-radius: 12px;
      }
    }
  `;
  document.head.appendChild(style);

  /* ---- FAB button ---- */
  const fab = document.createElement("button");
  fab.id = "jims-widget-fab";
  fab.setAttribute("aria-label", "Open chat");
  fab.innerHTML = `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.2L4 17.2V4h16v12z"/></svg>`;
  document.body.appendChild(fab);

  /* ---- Panel ---- */
  const panel = document.createElement("div");
  panel.id = "jims-widget-panel";
  if (!cfg.open) panel.setAttribute("data-closed", "");

  const header = document.createElement("div");
  header.id = "jims-widget-header";
  header.innerHTML = `<span>${cfg.title}</span><button aria-label="Close chat">&times;</button>`;
  panel.appendChild(header);

  const chatEl = document.createElement("deep-chat");
  chatEl.style.cssText = `
    flex: 1 1 auto;
    min-height: 0;
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    --deep-chat-send-btn-bg: ${cfg.accent};
    --deep-chat-user-bubble-bg: ${cfg.accent};
    --deep-chat-user-bubble-text: #fff;
    --deep-chat-ai-bubble-bg: #f0f0f5;
    --deep-chat-ai-bubble-text: #1a1a2e;
  `;
  panel.appendChild(chatEl);
  document.body.appendChild(panel);

  /* ---- Toggle logic ---- */
  function toggle() {
    const closed = panel.hasAttribute("data-closed");
    if (closed) {
      panel.removeAttribute("data-closed");
    } else {
      panel.setAttribute("data-closed", "");
    }
  }
  fab.addEventListener("click", toggle);
  header.querySelector("button").addEventListener("click", toggle);

  /* ---- Load DeepChat and configure ---- */
  const script = document.createElement("script");
  script.type = "module";
  script.textContent = `
    import "${DEEP_CHAT_CDN}";

    try {
      const { linkify } = await import("https://esm.sh/remarkable@2.0.1/linkify");
      window.remarkable_plugins = [{ plugin: linkify }];
    } catch (e) {
      console.warn("[jims-widget] linkify unavailable:", e);
    }

    const chat = document.querySelector("#jims-widget-panel deep-chat");

    const server = ${JSON.stringify(cfg.server)};
    const wsProto = server.startsWith("https") ? "wss:" : "ws:";
    const host = server.replace(/^https?:\\/\\//, "");

    let wsUrl = wsProto + "//" + host + "/ws/chat";
    const params = [];
    if (${JSON.stringify(cfg.contactId)}) params.push("contact_id=" + encodeURIComponent(${JSON.stringify(cfg.contactId)}));
    if (${JSON.stringify(cfg.threadId)}) params.push("thread_id=" + encodeURIComponent(${JSON.stringify(cfg.threadId)}));
    if (params.length) wsUrl += "?" + params.join("&");

    chat.textInput = { placeholder: { text: "Type a message…" } };

    chat.avatars = {
      ai: {
        src: "https://api.dicebear.com/9.x/bottts/svg?seed=JIMS",
        styles: { avatar: { width: "26px", height: "26px" } },
      },
      user: {
        src: "https://api.dicebear.com/9.x/initials/svg?seed=U",
        styles: { avatar: { width: "26px", height: "26px" } },
      },
    };

    chat.messageStyles = {
      default: {
        shared: {
          bubble: {
            borderRadius: "12px",
            padding: "10px 14px",
            maxWidth: "85%",
            fontSize: "0.9rem",
          },
        },
      },
      html: {
        shared: {
          bubble: {
            backgroundColor: "unset",
            padding: "6px 8px",
            maxWidth: "100%",
            width: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          },
        },
      },
    };

    // Pipeline buttons: bind click via htmlClassUtilities (Deep Chat shadow DOM).
    // https://deepchat.dev/docs/messages/HTML/#htmlclassutilities
    chat.htmlClassUtilities = {
      "jims-widget-button-stack": {
        styles: {
          default: {
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "10px",
            width: "100%",
            boxSizing: "border-box",
          },
        },
      },
      "jims-widget-button-row": {
        styles: {
          default: {
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: "8px",
            width: "100%",
          },
        },
      },
      "jims-widget-callback-btn": {
        events: {
          click: (event) => {
            const t = event.target;
            const el =
              t && typeof t.closest === "function"
                ? t.closest(".jims-widget-callback-btn")
                : event.currentTarget;
            const payload = el && el.dataset ? el.dataset.jimsCallback : "";
            const buttonText = el && typeof el.textContent === "string" ? el.textContent.trim() : "";
            if (payload) {
              event.preventDefault();
              event.stopPropagation();
              chat.submitUserMessage({
                text: buttonText || payload,
                custom: { jimsCallback: payload },
              });
            }
          },
        },
        styles: {
          default: {
            cursor: "pointer",
            minWidth: "12rem",
            maxWidth: "100%",
            padding: "10px 18px",
            fontWeight: "600",
          },
          hover: { filter: "brightness(0.95)" },
        },
      },
    };

    chat.remarkable = { linkTarget: "_blank" };

    const introMsg = ${JSON.stringify(cfg.introMessage)};
    if (introMsg) {
      chat.introMessage = { text: introMsg };
    }

    chat.displayLoadingBubble = true;
    chat.scrollButton = true;

    chat.requestInterceptor = (details) => {
      const body = typeof details.body === "string" ? JSON.parse(details.body) : details.body;
      const messages = body.messages || [];
      const last = messages[messages.length - 1];
      const fallbackText = last ? (last.text || "") : "";
      const callbackText =
        last &&
        last.custom &&
        typeof last.custom === "object" &&
        typeof last.custom.jimsCallback === "string"
          ? last.custom.jimsCallback
          : "";
      const outgoingText = callbackText || fallbackText;
      return {
        ...details,
        body: JSON.stringify({ messages: [{ role: "user", text: outgoingText }] }),
      };
    };

    chat.responseInterceptor = (response) => {
      try {
        if (response == null) return { error: "Empty server response" };

        let payload = response;
        if (typeof payload === "string") {
          try { payload = JSON.parse(payload); } catch (err) { return { text: payload }; }
        }

        if (Array.isArray(payload)) return payload;
        if (typeof payload !== "object") return { text: String(payload) };
        if (payload.error) return { error: String(payload.error) };
        if (payload.text != null || payload.html != null || payload.files) {
          const out = {};
          if (payload.text != null) out.text = String(payload.text);
          if (payload.html != null) out.html = String(payload.html);
          if (payload.files) out.files = payload.files;
          return out;
        }
        return response;
      } catch (err) {
        return { error: "Failed to process server response" };
      }
    };

    chat.connect = { websocket: true, url: wsUrl };
  `;
  document.body.appendChild(script);
})();
