# Apex A100 reviewer archives

Download the six ZIP assets from the [A100 evidence release](https://github.com/Aacashh/Apex/releases/tag/a100-evidence-2026-10-23). The names encode method, hardware, study, and campaign completion date. The comparator archive uses the latest completion date among its source campaigns. The packaging session spans 2026-10-22–23.

These are curated results-only archives. Original per-run files and available receipts are copied byte-for-byte into a consistent review layout. The historical records retain their original internal experiment identifiers. Source code, model weights, raw datasets, active-session credentials, and transport bundles are not included. The executable Apex implementation is in this repository.

## Archive contents

| ZIP | Unique run records | Purpose |
|---|---:|---|
| `Apex_A100_E1-E3_Primary_2026-10-18.zip` | 75 | Apex primary matrix: four datasets, 13 settings, 16,000 rounds |
| `Apex_A100_E4_Ablations_2026-10-18.zip` | 105 | 85 new ablation runs plus 20 reused A0 references; 110 analysis rows include five identity aliases |
| `Apex_A100_E5_Reward_2026-10-18.zip` | 40 | Matched R0–R3 reward/probe study: 10,000 rounds |
| `Apex_A100_E7_Feature-Shift_2026-10-18.zip` | 10 | CIFAR-10 feature-shift controls: 2,000 rounds |
| `Apex_A100_Independent-Convergence_2026-10-20.zip` | 40 | Apex, FedAvg, MMR, and FedCor on C4/D2, with independent audit curves; includes controlled CPU selector timing |
| `Apex_A100_Primary-Comparators_2026-10-19.zip` | 600 | The eight primary comparator methods, 75 runs each, used alongside the primary Apex matrix |

The primary comparator archive contains FedAvg, MMR, PoC, Oort, FedCor, FedCS, TiFL, and FedAEB. Their recorded adaptations and environments are retained in the result metadata. The standalone training package implements Apex only. The independent archive contains comparator observations to keep that paired study reviewable.

Do not add the E4 A0 references to the primary sample size, or count the five C0/V1 identity aliases as new executions. The independent-convergence campaign is a separate study; its same-seed curves must not be pooled with primary curves. Controlled CPU timing records in that archive are not A100 round-time measurements.

## Internal directory structure

Each ZIP has one top-level folder matching its filename without `.zip`:

```text
Apex_A100_<study>_<date>/
├── index.json
├── PACKAGE_AUDIT.json
├── SHA256SUMS.txt
├── configuration/
│   └── standalone.yaml
├── audit/
│   └── <original campaign audit records>.json
└── runs/
    └── <original-run-id>/
        ├── result.json
        ├── results.json
        ├── results_receipt.json
        ├── config_effective.json
        ├── provenance.json
        ├── metrics.jsonl
        ├── selection_history.jsonl
        ├── selector_diagnostics.jsonl
        ├── communication_accounting.jsonl
        ├── partition_indices.json
        ├── partition_summary.json
        ├── split_indices.json
        ├── environment_trace.json
        ├── outcomes.json
        ├── timings.json
        ├── completion.json
        └── <other recorded result metadata>
```

This tree is a union of recorded file types, rather than a promise that every file exists in every historical run. Earlier comparator records use `results.json`, with configuration/environment embedded, and may have no separate receipt. Later records use `result.json`, `config_effective.json`, and `results_receipt.json`. Available run files are preserved without filling historical gaps with invented records. The primary-comparator archive has no standalone configuration or campaign-audit folder because its rows originate from multiple comparator campaigns.

The independent-convergence archive additionally has:

```text
Apex_A100_Independent-Convergence_2026-10-20/
├── analysis/
│   ├── independent_summary.csv
│   ├── paired_auc_contrasts.csv
│   ├── selector_support_table.md
│   └── primary_query_refresh_work.csv
├── selector-timing/
│   └── <recorded timing outputs>.json
└── runs/<original-run-id>/
    └── independent_audit.json
```

`index.json` identifies each run's study setting, method label, seed, variant, endpoint round, original run ID, source hash, snapshot hash, and archive-relative directory. Its alias list maps every analysis row to a run record. `metrics.jsonl` contains the feedback trajectory; `independent_audit.json` contains the separate held-out trajectory when available. Final official-test metrics are in the final result record. Read these endpoints separately.

## Verification

[ARCHIVE_INDEX.json](ARCHIVE_INDEX.json) records archive sizes, hashes, counts, and packaging checks. [ARCHIVE_SHA256SUMS.txt](ARCHIVE_SHA256SUMS.txt) is the compact outer checksum list. After downloading all ZIPs into one directory, run this standard-library-only command from the repository:

```bash
python scripts/verify_archives.py /path/to/downloaded/archives
```

It checks the outer SHA-256, ZIP CRCs, every internal member hash, and checksum-index coverage. `PACKAGE_AUDIT.json` records how many original receipt file hashes were independently rechecked during packaging. Files without an original receipt still receive fresh archive-member hashes; those hashes attest to this package, not to an unavailable historical receipt.

Original receipt paths can refer to their previous campaign root. The basename and run ID map those records to `runs/<original-run-id>/`. Original completion records can also reference scientific/recovery checkpoints that were deliberately not exported. Use the archive verifier above for these results-only assets. The executable runner's `apex verify` command applies to its full local outputs, including retained checkpoints.

## Completion boundaries

E1–E3 training completed all 75 jobs and 16,000 rounds, with 1,425 downloaded run-file hashes checked against receipts. Its additional gradient-diagnostic campaign stopped with 87 states recorded; this archive does not claim a complete gradient diagnostic. That limitation is preserved in `audit/independent_completion.json`.

E4, the combined E5/E7 campaign, and independent convergence each have original completion audits. The combined E5/E7 audit appears in both split archives and describes all 50 jobs; each archive's `index.json` gives its own subset. Completed-run evidence is retained without converting partial diagnostics or same-seed repeat studies into stronger claims.

The archives contain no resumable tensors. Historical experiment environments remain recorded in their own provenance. The extraction and local validation of this repository do not constitute a new A100 reproduction campaign.
