// src/content/vedana/seo/seo.ts

import type { SEOConfig } from "@/lib/seo/types";

export const SEOBase: SEOConfig = {
  title: "Vedana — Graph-first AI for Enterprise Decisions",
  description:
    "Structured reasoning infrastructure for enterprises. Build knowledge graphs and answer constrained business questions reliably.",

  siteUrl: "https://vedana.tech",

  og: {
    type: "website",
    image: "/og/vedana-default.png",
    imageAlt: "Vedana — Graph-first AI",
  },

  twitter: {
    card: "summary_large_image",
    image: "/og/vedana-default.png",
  },
};