#!/usr/bin/env node
/* eslint-disable no-console */

/* ========================================
   build-versions.mjs

   Materializes versioned docs snapshots from this repo's git tags
   into vedana.tech/.generated/docs/<version>/ and writes the
   manifest at vedana.tech/.generated/versions.json.

   The Astro content collection in src/content/config.ts picks these
   up automatically; nothing else in the site needs to change to add
   or remove a published version — just (re)run this script before
   `astro build`.

   Configuration via environment variables:

     DOCS_TAG_PATTERN   git tag glob to consider, default "v[0-9]*.[0-9]*.[0-9]*"
     DOCS_MAX_VERSIONS  cap on the number of versioned snapshots (0 = unlimited)
     DOCS_STABLE        explicit stable version id (default: highest matching tag)
     DOCS_DOCS_PATH     path inside the tag holding docs content,
                        default "vedana.tech/src/content/docs"
     DOCS_LATEST_LABEL  label for the "latest" entry, default "latest"

   The script is intentionally idempotent: it wipes .generated/docs
   before each run so stale versions cannot leak into the build.
======================================== */

import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SITE_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(SITE_ROOT, "..");

const TAG_PATTERN = process.env.DOCS_TAG_PATTERN || "v[0-9]*.[0-9]*.[0-9]*";
const MAX_VERSIONS = parseInt(process.env.DOCS_MAX_VERSIONS || "0", 10);
const FORCED_STABLE = process.env.DOCS_STABLE || "";
const DOCS_PATH_IN_TAG =
  process.env.DOCS_DOCS_PATH || "vedana.tech/src/content/docs";
const LATEST_LABEL = process.env.DOCS_LATEST_LABEL || "latest";

const GENERATED_DIR = path.join(SITE_ROOT, ".generated");
const DOCS_OUT_DIR = path.join(GENERATED_DIR, "docs");
const MANIFEST_PATH = path.join(GENERATED_DIR, "versions.json");

/* ========================================
   MAIN
======================================== */

function main() {
  ensureGitRepo();
  resetGeneratedDir();

  const allTags = listMatchingTags(TAG_PATTERN);
  const sortedTags = sortTagsDescending(allTags);

  const trimmedTags = MAX_VERSIONS > 0
    ? sortedTags.slice(0, MAX_VERSIONS)
    : sortedTags;

  const exported = [];

  for (const tag of trimmedTags) {
    const ok = exportDocsFromTag(tag);
    if (ok) {
      exported.push(tag);
      console.log(`[versions] exported ${tag}`);
    } else {
      console.log(`[versions] skipped ${tag} (no docs at ${DOCS_PATH_IN_TAG})`);
    }
  }

  const stableId = pickStable(exported, FORCED_STABLE);
  const manifest = buildManifest(exported, stableId);

  fs.mkdirSync(GENERATED_DIR, { recursive: true });
  fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2) + "\n");

  console.log(
    `[versions] wrote ${MANIFEST_PATH} with ${manifest.versions.length} version(s); stable=${manifest.stableId}`,
  );
}

/* ========================================
   GIT HELPERS
======================================== */

function ensureGitRepo() {
  try {
    execFileSync("git", ["rev-parse", "--is-inside-work-tree"], {
      cwd: REPO_ROOT,
      stdio: "ignore",
    });
  } catch {
    throw new Error(
      `Not a git repository at ${REPO_ROOT}; cannot enumerate version tags.`,
    );
  }
}

function listMatchingTags(pattern) {
  const out = execFileSync(
    "git",
    ["tag", "--list", pattern],
    { cwd: REPO_ROOT, encoding: "utf8" },
  );

  return out
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

function tagExistsWithPath(tag, p) {
  const result = spawnSync(
    "git",
    ["ls-tree", "-r", "--name-only", tag, "--", p],
    { cwd: REPO_ROOT, encoding: "utf8" },
  );

  if (result.status !== 0) return false;
  return result.stdout.trim().length > 0;
}

function exportDocsFromTag(tag) {
  if (!tagExistsWithPath(tag, DOCS_PATH_IN_TAG)) return false;

  const outDir = path.join(DOCS_OUT_DIR, tag);
  fs.mkdirSync(outDir, { recursive: true });

  // git archive | tar -x. The archive lists files under
  // <DOCS_PATH_IN_TAG>/...; we strip the leading path with --strip-components
  // so files land directly under .generated/docs/<tag>/.
  const stripDepth = DOCS_PATH_IN_TAG.split("/").length;

  const archive = spawnSync(
    "git",
    ["archive", "--format=tar", tag, "--", DOCS_PATH_IN_TAG],
    { cwd: REPO_ROOT, encoding: "buffer" },
  );

  if (archive.status !== 0) {
    fs.rmSync(outDir, { recursive: true, force: true });
    return false;
  }

  const extract = spawnSync(
    "tar",
    ["-x", "-C", outDir, `--strip-components=${stripDepth}`],
    { input: archive.stdout },
  );

  if (extract.status !== 0) {
    fs.rmSync(outDir, { recursive: true, force: true });
    return false;
  }

  // Sanity check the extraction produced at least one markdown file.
  if (!hasMarkdownFiles(outDir)) {
    fs.rmSync(outDir, { recursive: true, force: true });
    return false;
  }

  return true;
}

function hasMarkdownFiles(dir) {
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    for (const entry of fs.readdirSync(cur, { withFileTypes: true })) {
      const p = path.join(cur, entry.name);
      if (entry.isDirectory()) stack.push(p);
      else if (/\.(md|mdx)$/i.test(entry.name)) return true;
    }
  }
  return false;
}

/* ========================================
   MANIFEST
======================================== */

function pickStable(exportedTags, forced) {
  if (forced && exportedTags.includes(forced)) return forced;
  if (exportedTags.length === 0) return "latest";
  return exportedTags[0];
}

function buildManifest(exportedTags, stableId) {
  const versions = [
    {
      id: "latest",
      label: LATEST_LABEL,
      ref: null,
    },
    ...exportedTags.map((tag) => ({
      id: tag,
      label: tag,
      ref: tag,
    })),
  ];

  return {
    latestId: "latest",
    stableId,
    versions,
  };
}

/* ========================================
   FS / SORTING
======================================== */

function resetGeneratedDir() {
  fs.rmSync(DOCS_OUT_DIR, { recursive: true, force: true });
  fs.mkdirSync(DOCS_OUT_DIR, { recursive: true });
}

/**
 * Sort tags descending by semver-ish comparison (e.g. v0.10.0 > v0.9.0).
 * Falls back to lexicographic for unparseable suffixes.
 */
function sortTagsDescending(tags) {
  return [...tags].sort((a, b) => compareTagDesc(a, b));
}

function compareTagDesc(a, b) {
  const pa = parseSemverish(a);
  const pb = parseSemverish(b);

  for (let i = 0; i < 3; i++) {
    if (pa.parts[i] !== pb.parts[i]) return pb.parts[i] - pa.parts[i];
  }

  // Release > pre-release; otherwise lex on suffix.
  if (pa.suffix === pb.suffix) return 0;
  if (!pa.suffix) return -1;
  if (!pb.suffix) return 1;
  return pb.suffix.localeCompare(pa.suffix);
}

function parseSemverish(tag) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)(?:[-+](.+))?$/.exec(tag);
  if (!m) return { parts: [0, 0, 0], suffix: tag };
  return {
    parts: [parseInt(m[1], 10), parseInt(m[2], 10), parseInt(m[3], 10)],
    suffix: m[4] || "",
  };
}

/* ========================================
   RUN
======================================== */

try {
  main();
} catch (err) {
  console.error("[versions] fatal:", err.message || err);
  process.exit(1);
}
