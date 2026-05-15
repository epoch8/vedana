// src/components/MiniDemo.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button, ConfigProvider } from "antd";
import styles from "./MiniDemo.module.css";

const theme = {
  token: {
    colorPrimary: "#2563eb",
    borderRadius: 8,
    fontFamily: "inherit",
  },
};

/* =======================
   Demo data & configuration
======================= */

const PRESETS = {
  p1: {
    label: "Show all Philips lamps",
    left: {
      lines: [
        { text: "Here are some Philips lamps I found:" },
        { text: "" },
        { text: "• Philips Hue Go" },
        { text: "• Philips LED 9W E27" },
        { text: "• Philips Tube 18W" },
        { text: "• Philips LED GU10 5W" },
        { text: "• Philips SceneSwitch A60" },
        { text: "" },
        { chunks: [{ text: "This may not be a full list.", mark: "red" }] },
      ],
      why: [
        'Matched text for "Philips" and "lamp"; did not use brand ID or category graph.',
        "Can miss items, include duplicates, and lacks stock or socket filters.",
      ],
    },
    right: {
      lines: [
        {
          chunks: [
            { text: "There are " },
            { text: "128 Philips lamps", mark: "green" },
            { text: " in the catalog." },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "By type:", mark: "green" },
            { text: " bulbs (76), tubes (22), fixtures (30)." },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "By socket:", mark: "green" },
            { text: " E27 (54), GU10 (18), E14 (12), other (44)." },
          ],
        },
        { text: "" },
        { chunks: [{ text: "Here are top 3 bestsellers:", mark: "green" }] },
        { text: "1) Philips Hue Go" },
        { text: "2) Philips LED 9W E27" },
        { text: "3) Master LEDspot GU10 4.2W" },
        { text: "" },
        { text: "You can clarify lamp and socket type to adjust your search." },
      ],
      why: [
        "Used brand=Philips AND category∈{bulb, tube, fixture} with exact brand ID.",
        "Joined with inventory and attributes (socket) to build counts and filters.",
        "Sources: Products, Inventory, Vendors.",
      ],
    },
  },
  p3: {
    label: "Which accessories are compatible with iPhone 14 Pro?",
    left: {
      lines: [
        { text: "Possible accessories:" },
        { text: "" },
        { text: "• Cases for iPhone 13/14" },
        { text: "• Standard 18W USB chargers" },
        { text: "• Many MagSafe mounts" },
        { text: "" },
        {
          text: "You can try chargers and cases labeled for iPhone 14 as they often fit the Pro model.",
        },
        {
          chunks: [
            {
              text: "This list is generic and may not match iPhone 14 Pro.",
              mark: "red",
            },
          ],
        },
      ],
      why: [
        "Assumed fit based on similar wording in product texts; no check against device model IDs or fit tests.",
        "No reason or proof of compatibility.",
      ],
    },
    right: {
      lines: [
        {
          chunks: [
            { text: "Cases:", mark: "green" },
            {
              text: " Only models mapped to A2890/A2892 form factor (e.g., Case_XP-14P-Mag, Case_Slim-14P).",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Power:", mark: "green" },
            {
              text: " USB-C PD ≥20W with Lightning/USB-C cable; MagSafe OK (tested Magsafe_Charger_MF1).",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Mounts/optics:", mark: "green" },
            {
              text: " MagSafe-compatible only; clip-only mounts are excluded.",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Not compatible:", mark: "green" },
            { text: " Cases for iPhone 13 Pro (camera housing differs)." },
          ],
        },
      ],
      why: [
        "Used link table sku_compat (device↔accessory) plus specs and QA fit tests.",
        "Sources: Compat, Specs, QA_tests (FT-14P-021, FT-14P-033).",
      ],
    },
  },
  p4: {
    label: "Payment terms for vendor ACME",
    left: {
      lines: [
        {
          text: "Standard terms are often Net 30. Some vendors may offer early-payment discounts.",
        },
        { text: "Please check the current contract for details." },
        {
          chunks: [{ text: "This is generic text and may not match ACME.", mark: "red" }],
        },
      ],
      why: [
        "Returned generic policy text; mixed vendor names and ignored amendments.",
        "No effective dates or links to the actual documents.",
      ],
    },
    right: {
      lines: [
        {
          chunks: [
            { text: "Vendor:", mark: "green" },
            { text: " ACME Ltd. (VID 48321)." },
          ],
        },
        {
          chunks: [
            { text: "Contract:", mark: "green" },
            {
              text: " MSA-2024-017, effective 2024-05-01; latest amendment A2 dated 2025-02-10.",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Terms:", mark: "green" },
            {
              text: " Net 30 EOM; 2/10 net 30; currency EUR; late fee 0.5%/month.",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Invoicing:", mark: "green" },
            {
              text: " Invoices via Coupa; PO required; VAT EU123456; IBAN DE….",
            },
          ],
        },
        { text: "" },
        {
          chunks: [
            { text: "Approvals:", mark: "green" },
            { text: " Invoices > €50k require CFO approval." },
          ],
        },
      ],
      why: [
        "Followed vendor → contract → amendment → terms with date filters to pick active clauses.",
        "Sources: Contracts, Amendments, Vendor_registry, Procurement_policy.",
      ],
    },
  },
};

const DEMO_VISIBLE_QUERIES = [
  { key: "p1", text: "Show all Philips lamps" },
  { key: "p3", text: "Which accessories are compatible with iPhone 14 Pro?" },
  { key: "p4", text: "Payment terms for vendor ACME" },
];

/* =======================
   In-view trigger (once)
======================= */

function useInViewOnce(options) {
  const ref = useRef(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (!ref.current || inView) return;

    const obs = new IntersectionObserver(
      (ents) => {
        ents.forEach((e) => {
          if (e.isIntersecting) {
            setInView(true);
            obs.disconnect();
          }
        });
      },
      options || { rootMargin: "-80px 0px" }
    );

    obs.observe(ref.current);
    return () => obs.disconnect();
  }, [inView, options]);

  return [ref, inView];
}

/* =======================
   Chat Avatar
======================= */

function ChatAvatar() {
  return (
    <span className={styles["chat-avatar"]}>
      <span className={styles["chat-avatar-dot"]} />
    </span>
  );
}

/* =======================
   Typewriter
======================= */

function normalizeLines(lines) {
  return (lines || []).map((ln) => {
    if (typeof ln === "string") return { kind: "text", text: ln };
    if (ln && Array.isArray(ln.chunks)) return { kind: "chunks", chunks: ln.chunks };
    if (ln && typeof ln.text === "string") return { kind: "text", text: ln.text };
    return { kind: "text", text: String(ln ?? "") };
  });
}

function countTotalChars(ln) {
  if (ln.kind === "text") return ln.text.length;
  return ln.chunks.reduce((acc, c) => acc + (c?.text?.length || 0), 0);
}

function Typewriter({
  lines,
  speed = 50,
  startKey = 0,
  showCursor = true,
  className = "",
  paintMarks = true,
  onDone,
}) {
  const prefersReduced = useMemo(() => {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  const flat = useMemo(() => normalizeLines(lines), [lines]);
  const totals = useMemo(() => flat.map(countTotalChars), [flat]);

  const [progress, setProgress] = useState(flat.map(() => 0));
  const [activeLine, setActiveLine] = useState(0);
  const doneCallbackFired = useRef(false);

  // reset on startKey/lines
  useEffect(() => {
    setProgress(flat.map(() => (prefersReduced ? Infinity : 0)));
    setActiveLine(0);
    doneCallbackFired.current = false;
  }, [startKey, flat, prefersReduced]);

  // typing loop
  useEffect(() => {
    if (prefersReduced) return;
    if (activeLine >= flat.length) return;

    const totalChars = totals[activeLine] ?? 0;

    // skip truly empty lines immediately (no cursor artefacts)
    if (totalChars === 0) {
      setActiveLine((x) => Math.min(x + 1, flat.length));
      return;
    }

    const id = setInterval(() => {
      setProgress((prev) => {
        const next = [...prev];
        const curr = next[activeLine];

        if (curr < totalChars) {
          next[activeLine] = curr + 1;
          return next;
        }

        // move to next line
        setActiveLine((x) => Math.min(x + 1, flat.length));
        clearInterval(id);
        return next;
      });
    }, speed);

    return () => clearInterval(id);
  }, [activeLine, totals, speed, prefersReduced, flat.length]);

  // done callback
  useEffect(() => {
    const isDone = prefersReduced || activeLine >= flat.length;
    if (!doneCallbackFired.current && isDone) {
      doneCallbackFired.current = true;
      if (typeof onDone === "function") onDone();
    }
  }, [activeLine, prefersReduced, flat.length, onDone]);

  function renderChunks(ln, count) {
    let left = count;

    return ln.chunks
      .map((c, i) => {
        const txt = c?.text || "";
        if (!txt) return null;

        const take = Math.max(0, Math.min(left, txt.length));
        if (take <= 0) return null;

        const visible = txt.slice(0, take);
        left -= take;

        if (c.mark === "red" && paintMarks) {
          return (
            <span key={i} className={styles["mark-red"]}>
              {visible}
            </span>
          );
        }
        if (c.mark === "green" && paintMarks) {
          return (
            <span key={i} className={styles["mark-green"]}>
              {visible}
            </span>
          );
        }
        if (c.mark && paintMarks) {
          return <strong key={i}>{visible}</strong>;
        }
        return <React.Fragment key={i}>{visible}</React.Fragment>;
      })
      .filter(Boolean);
  }

  function renderLine(ln, idx) {
    const total = totals[idx] ?? 0;
    if (total === 0) return null;

    const done = prefersReduced || progress[idx] === Infinity || progress[idx] >= total;
    const count = prefersReduced ? total : progress[idx];

    // cursor only when there is *some* visible text on the active line
    const showCursorHere = showCursor && !done && idx === activeLine && count > 0;
    const cursor = showCursorHere ? <span className={styles.cursor} /> : null;

    if (ln.kind === "text") {
      const content = ln.text.slice(0, count);
      if (!content && !cursor) return null;
      return (
        <p key={idx}>
          {content}
          {cursor}
        </p>
      );
    }

    const nodes = renderChunks(ln, count);
    if (nodes.length === 0 && !cursor) return null;

    return (
      <p key={idx}>
        {nodes}
        {cursor}
      </p>
    );
  }

  return (
    <div className={`${styles["typewriter-output"]} ${className}`} aria-live="polite">
      {flat.map((ln, i) => renderLine(ln, i))}
    </div>
  );
}

/* =======================
   Mini Demo
======================= */

export default function MiniDemo() {
  const presets = useMemo(() => PRESETS, []);
  const [pickedKey, setPickedKey] = useState(null);
  const [opened, setOpened] = useState(false);
  const [runKey, setRunKey] = useState(0);

  const data = presets[pickedKey || "p1"];

  const [ref, inView] = useInViewOnce({ rootMargin: "-120px 0px" });

  // start typing only when opened + in view
  const shouldStart = opened && inView ? runKey : -1;

  // show "Why this" only when BOTH typewriters are done
  const [leftDone, setLeftDone] = useState(false);
  const [rightDone, setRightDone] = useState(false);
  const showWhy = opened && inView && leftDone && rightDone;

  useEffect(() => {
    if (!opened) return;
    // restart typing -> reset done flags
    setLeftDone(false);
    setRightDone(false);
    setRunKey((k) => k + 1);
  }, [pickedKey, opened]);

  function onPick(k) {
    setPickedKey(k);
    if (!opened) setOpened(true);
  }

  return (
    <ConfigProvider theme={theme}>
      <section id="demo" className={styles["demo-section"]} ref={ref}>
        <div className={styles["demo-inner"]}>
          <h2 className={styles["demo-title"]}>Which response do you prefer?</h2>
          <p className={styles["demo-subtitle"]}>Pick a query and compare outputs.</p>

          <div className={styles["query-buttons"]}>
            {DEMO_VISIBLE_QUERIES.map(({ key: k, text }) => {
              const active = opened && pickedKey === k;
              return (
                <Button
                  key={k}
                  shape="round"
                  type={active ? "primary" : "default"}
                  onClick={() => onPick(k)}
                  aria-pressed={active}
                >
                  {text}
                </Button>
              );
            })}
          </div>

          {opened && (
            <>
              <div className={styles["response-grid"]}>
                <div className={styles["response-panel"]}>
                  <div className={styles["response-label"]}>Response 1</div>
                  <div className={styles["response-header"]}>
                    <ChatAvatar />
                    <h3>RAG output</h3>
                    <span className={styles["response-subtext"]}>Text similarity</span>
                  </div>

                  <div className={styles["response-content"]}>
                    <Typewriter
                      lines={data.left.lines}
                      startKey={shouldStart}
                      speed={16}
                      showCursor={true}
                      paintMarks={true}
                      onDone={() => setLeftDone(true)}
                    />
                  </div>
                </div>

                <div className={styles["response-panel"]}>
                  <div className={styles["response-label"]}>Response 2</div>
                  <div className={styles["response-header"]}>
                    <ChatAvatar />
                    <h3>With Semantic RAG</h3>
                    <span className={styles["response-subtext"]}>Data model + rules</span>
                  </div>

                  <div className={styles["response-content"]}>
                    <Typewriter
                      lines={data.right.lines}
                      startKey={shouldStart}
                      speed={16}
                      showCursor={true}
                      paintMarks={true}
                      onDone={() => setRightDone(true)}
                    />
                  </div>
                </div>
              </div>

              {showWhy && (
                <div className={styles["why-grid"]}>
                  <div className={styles["why-panel"]}>
                    <p className={styles["why-title"]}>Why this</p>
                    <ul className={styles["why-list"]}>
                      {data.left.why.map((line, i) => (
                        <li key={i}>{line}</li>
                      ))}
                    </ul>
                  </div>
                  <div className={styles["why-panel"]}>
                    <p className={styles["why-title"]}>Why this</p>
                    <ul className={styles["why-list"]}>
                      {data.right.why.map((line, i) => (
                        <li key={i}>{line}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </section>
    </ConfigProvider>
  );
}


