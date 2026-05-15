import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import fs from "node:fs";
import path from "node:path";

import {
  LATEST_ID,
  loadVersionsManifest,
  versionPathSegment,
} from "../lib/docs/versions";

/* ========================================
   ID / SLUG / SECTION HELPERS
======================================== */

/**
 * folder → section
 * 01-overview → Overview
 */
function deriveSection(id: string): string {
  const dir = id.split("/")[0] ?? "";

  return dir
    .replace(/^\d+-/, "")
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/**
 * id → slug
 * 01-overview/02-what-is-vedana.md → overview/what-is-vedana
 */
function deriveSlug(id: string): string {
  return id
    .replace(/\.(md|mdx)$/, "")
    .split("/")
    .map((part) => part.replace(/^\d+-/, ""))
    .join("/");
}

/* ========================================
   COLLECTION
========================================

   The docs collection contains entries for every published version
   of the documentation:

   - "latest" entries come from src/content/docs/**.
   - Each snapshotted version "vX.Y.Z" comes from
     .generated/docs/vX.Y.Z/**, which the build script
     scripts/build-versions.mjs materializes from git tags.

   Every entry carries a `version` field on its data so that the
   routes and navigation can filter/group accordingly. For non-latest
   versions, the entry id and slug are prefixed with the version
   segment so the routes resolve to /docs/vX.Y.Z/<slug>.
======================================== */

const REPO_ROOT_FROM_CWD = ".";

const docsCollection = defineCollection({
  schema: z.object({
    title: z.string(),
    slug: z.string().optional(),

    section: z.string().optional(),
    order: z.number().optional(),

    next: z.string().optional(),
    previous: z.string().optional(),

    version: z.string().optional(),
  }),

  loader: {
    name: "docs",
    load: async (ctx: any) => {
      ctx.store.clear();

      const manifest = loadVersionsManifest();

      const collected: Array<{
        versionId: string;
        entries: any[];
      }> = [];

      for (const version of manifest.versions) {
        const base = resolveVersionContentBase(version.id);

        if (!fs.existsSync(path.resolve(REPO_ROOT_FROM_CWD, base))) {
          if (version.id === LATEST_ID) {
            // Latest source missing is a hard error; everything else is best-effort.
            throw new Error(
              `Missing docs source for latest at ${base}`,
            );
          }
          continue;
        }

        ctx.store.clear();

        await glob({
          pattern: "**/*.{md,mdx}",
          base,
        }).load(ctx);

        const entries = [...ctx.store.values()];
        collected.push({ versionId: version.id, entries });
      }

      // Re-emit all entries with version-aware ids/slugs.

      ctx.store.clear();

      for (const { versionId, entries } of collected) {
        const segment = versionPathSegment(versionId);

        for (const entry of entries) {
          const { digest, ...rest } = entry;

          const originalId: string = entry.id;
          const slug = deriveSlug(originalId);
          const versionedSlug = segment ? `${segment}/${slug}` : slug;

          ctx.store.set({
            ...rest,
            id: versionedSlug,
            data: {
              ...entry.data,
              section:
                entry.data.section ?? deriveSection(originalId),
              order: entry.data.order ?? 0,
              version: versionId,
            },
          });
        }
      }
    },
  },
});

/* ========================================
   HELPERS
======================================== */

function resolveVersionContentBase(versionId: string): string {
  if (versionId === LATEST_ID) {
    return "./src/content/docs";
  }

  return `./.generated/docs/${versionId}`;
}

export const collections = {
  docs: docsCollection,
};
