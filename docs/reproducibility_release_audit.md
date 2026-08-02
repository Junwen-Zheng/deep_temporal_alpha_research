# Reproducibility and release-readiness audit

## Purpose

This stage verifies that the repository is internally consistent before the
version tag and GitHub release are created.

It does not rerun model training. The frozen model and evaluation outputs have
already been reproduced or cross-checked within their respective milestones.

## Frozen-report rebuild order

The deterministic Milestone 22 report inventories all committed research
sources through Milestone 21.

The report is therefore rebuilt twice before release-audit files are created.
Both rebuilds must reproduce the committed HTML, findings, inventory, and
report-manifest hashes exactly.

After the release-audit files are introduced, the report is treated as a
frozen artifact and is verified against:

- its committed report manifest;
- the two-build release record;
- its committed SHA-256.

Regenerating the report after adding new release-audit evidence would expand
its source universe and would require a deliberate report-version update.

## Automated checks

The release audit verifies:

1. the complete pytest suite;
2. Ruff;
3. dependency consistency;
4. Python compilation;
5. Git whitespace checks;
6. that working-tree changes are restricted to Milestone 23 files;
7. byte-identical report rebuilding;
8. the report-manifest hash;
9. historical source and evidence hashes recorded by project manifests;
10. release-inventory ordering and uniqueness;
11. repository file-size limits;
12. absence of symlinks;
13. UTF-8 text decoding;
14. LF-only line endings;
15. absence of machine-specific user paths;
16. required package-version capture.

## Historical manifest validation

Every evidence JSON file matching the project manifest naming convention is
inspected for:

- path and SHA-256 records;
- source-file hash mappings;
- evidence-file hash mappings;
- sibling path and hash fields.

Every discovered reference must exist locally and match its recorded SHA-256.

## Historical source snapshots

Run manifests record the exact source configuration used when each experiment
was produced. A configuration file may be extended by later milestones, so its
current working-tree hash can legitimately differ from an earlier manifest.

The audit first compares each reference with the current file. When a current
source hash differs, it searches committed Git history for a blob at the same
repository path with the exact recorded SHA-256.

A historical reference passes only when the exact bytes are present either in
the current repository state or in committed Git history. Missing hashes and
hashes that cannot be recovered from Git remain failures. Historical manifests
are never rewritten to match newer configuration files.

## Local hash-reference inference

Explicit `path` and `sha256` records, source-file mappings, and evidence-file
mappings are always treated as local references.

For generic sibling fields such as `config` and `config_sha256`, the audit
infers a local file only when the value resembles a repository path. Bare
source labels and remote URLs are not treated as local files. This prevents
metadata descriptions from being misclassified while preserving strict
verification of actual repository artifacts.

## Release inventory

The release inventory covers all tracked files plus pending Milestone 23 files.

The following aggregate audit outputs are excluded to prevent recursive
self-hashing:

- the audit table;
- the release inventory itself;
- the release manifest.

Their hashes are instead recorded directly in the release manifest.

## Environment record

The audit captures:

- Python implementation and version;
- operating-system family;
- machine architecture;
- versions of NumPy, pandas, PyArrow, scikit-learn, LightGBM, PyTorch,
  pytest, and Ruff.

No generation timestamp or machine-specific filesystem path is embedded.

## Release boundary

A passing audit establishes repository consistency and reproducibility of the
committed evidence chain. It does not establish live tradability, profitability,
or freedom from fixed-universe and execution-model limitations.
