import { defineConfig } from "astro/config";
import react from "@astrojs/react";

import path from "node:path";

import remarkRelativeDocLinks from "./src/lib/docs/remark-relative-doc-links";

export default defineConfig({
  site: "https://vedana.tech",

  output: "static",

  base: "/",

  integrations: [
    react(),
  ],

  markdown: {
    remarkPlugins: [
      remarkRelativeDocLinks,
    ],
  },

  build: {
    assets: "_astro",
  },

  vite: {
    resolve: {
      alias: {
        "@": path.resolve("./src"),

        "@components": path.resolve("./src/components"),

        "@content": path.resolve("./src/content"),

        "@styles": path.resolve("./src/styles"),
      },
    },
  },
});