import { useEffect, useRef } from "react";

type Props = {
  href?: string;
};

const TAU = Math.PI * 2;

/* =========================
   PATH GENERATOR
========================= */

function cloverPath({
  a = 120,
  k = 2,
  steps = 120,
  cx = 200,
  cy = 200
}: {
  a?: number;
  k?: number;
  steps?: number;
  cx?: number;
  cy?: number;
}) {
  const pts: [number, number][] = [];

  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * TAU;

    const r = a * Math.cos(k * t);

    const x = cx + r * Math.cos(t);
    const y = cy + r * Math.sin(t);

    pts.push([x, y]);
  }

  return pts.map(([x, y], i) => `${i ? "L" : "M"} ${x} ${y}`).join(" ") + " Z";
}

/* =========================
   EASING
========================= */

function easeInOutCubic(t: number) {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

/* =========================
   COMPONENT
========================= */

export default function BrandLogo({ href = "/" }: Props) {
  const pathRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    const path = pathRef.current;
    if (!path) return;

    let raf = 0;
    let last = 0;
    let phase = 0;

    const speed = 0.13; // медленнее

    const kClover = 2;
    const kCircle = 0;

    // сколько времени висим в крайних точках
    const pause = 0.12;

    const tick = (now: number) => {
      if (!last) last = now;

      const dt = (now - last) / 1000;
      last = now;

      phase += dt * speed;

      const cycle = phase % 1;

      let t;

      if (cycle < pause) {
        // пауза на клевере
        t = 0;
      } else if (cycle < 0.5) {
        // morph к кругу
        const local = (cycle - pause) / (0.5 - pause);
        t = easeInOutCubic(local);
      } else if (cycle < 0.5 + pause) {
        // пауза на круге
        t = 1;
      } else {
        // morph обратно
        const local = (cycle - (0.5 + pause)) / (0.5 - pause);
        t = 1 - easeInOutCubic(local);
      }

      const k = lerp(kClover, kCircle, t);

      path.setAttribute("d", cloverPath({ k }));

      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <a href={href} className="brand-logo">
      <svg className="brand-icon" viewBox="0 0 400 400" aria-hidden="true">
        <path
          ref={pathRef}
          fill="none"
          stroke="currentColor"
          strokeWidth={27}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </a>
  );
}