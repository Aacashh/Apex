# Reproducing Apex

## Data and preprocessing

The official training split is divided once into 90% client-training, 5% server-feedback, and 5% tuning data using a deterministic stratified split with seed 260928. For CIFAR this is 45,000/2,500/2,500; for MNIST and Fashion-MNIST it is 54,000/3,000/3,000. The official 10,000-example test set is accessed for metrics at completion, never for selection, reward, or stopping. Primary runs record tuning accuracy at predetermined checkpoints; the final reported test is the final model, not a tuning-selected model.

Independent-convergence runs reuse the tuning allocation as an audit split, evaluate it every five rounds from round zero, disable tuning-based checkpoint selection, and restore model/RNG state after observation. Audit values do not enter the selector or reward. Keep this study separate from primary feedback-based curves.

Torchvision stores files under the supplied root:

```text
data/
├── cifar-10-batches-py/
├── cifar-100-python/
├── FashionMNIST/raw/
└── MNIST/raw/
```

Raw arrays and labels are fingerprinted. Preparation validates official train/test counts. Downloading is opt-in through `prepare --download`; later execution reads the prepared data without downloading.

| Dataset | Per-channel normalization mean | Standard deviation | Training augmentation |
|---|---|---|---|
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) | Random 32-pixel crop with padding 4; random horizontal flip |
| CIFAR-100 | (0.5071, 0.4867, 0.4408) | (0.2675, 0.2565, 0.2761) | Same crop and flip |
| Fashion-MNIST | (0.2860) | (0.3530) | None |
| MNIST | (0.1307) | (0.3081) | None |

Evaluation has no random crop/flip. CIFAR ResNet18 uses a 3 × 3 stride-one first convolution, removes the initial max-pool, and has a dataset-sized output layer. The grayscale CNN has convolutions 1→10 and 10→20 with kernel 5, max pooling, then fully connected layers 320→50→10. No pretrained weights are downloaded.

IID partitions shuffle and distribute samples across clients. Non-IID partitions draw classwise Dirichlet proportions with the configured concentration. An empty client receives one sample from the largest donor client; repairs are recorded. Every run uses deterministic initial model, partition, environment, selector, and local-training streams derived from its seed.

## Primary matrix

| Cell | Dataset | Partition / alpha | N | K | Rounds | Seeds |
|---|---|---|---:|---:|---:|---|
| C0 | CIFAR-10 | Dirichlet / 0.3 | 50 | 10 | 200 | S10 |
| C1 | CIFAR-10 | Dirichlet / 0.1 | 50 | 10 | 200 | S5 |
| C2 | CIFAR-10 | Dirichlet / 0.6 | 50 | 10 | 200 | S5 |
| C3 | CIFAR-10 | IID | 50 | 10 | 200 | S5 |
| C4 | CIFAR-10 | Dirichlet / 0.1 | 100 | 5 | 300 | S10 |
| D1 | CIFAR-100 | Dirichlet / 0.3 | 50 | 10 | 300 | S5 |
| D2 | Fashion-MNIST | Dirichlet / 0.3 | 50 | 10 | 200 | S5 |
| D3 | MNIST | Dirichlet / 0.3 | 50 | 10 | 200 | S5 |
| S1 | CIFAR-10 | Dirichlet / 0.3 | 100 | 10 | 200 | S5 |
| S2 | CIFAR-10 | Dirichlet / 0.3 | 200 | 10 | 200 | S5 |
| S3 | CIFAR-10 | Dirichlet / 0.3 | 100 | 20 | 200 | S5 |
| S4 | CIFAR-10 | Dirichlet / 0.3 | 200 | 40 | 200 | S5 |
| S5-bridge | CIFAR-10 | Dirichlet / 0.3 | 200 | 20 | 200 | S5 |

S5 = 42, 123, 456, 789, 1024. S10 adds 2025, 3141, 4096, 5678, 6789. Model parameters remain FP32. All use one local epoch, batch size 128, SGD learning rate 0.01, zero momentum/weight decay, and no scheduler. Feedback evaluation batch size is 512.

The study YAMLs explicitly use `cuda_bf16` for CIFAR and `reference` for grayscale. The reference profile performs FP32 training/evaluation with torchvision augmentation. The CIFAR BF16 profile keeps model/optimizer state in FP32, uses BF16 training and feedback with a separately seeded CUDA augmentation generator, and evaluates final test data in FP32. These are distinct numerical/RNG profiles. Identical seeds across them do not imply identical results. `cuda_bf16_compile` is an optional additional profile and is not the archived default.

The standalone package changes module names and source identities. Reproducing exact historical trajectories also depends on the recorded software, GPU, transforms, settings, and RNG profile. Local extraction tests are not a new A100 campaign or a claim that all historical metrics have been regenerated.

## Ablations

| Variant | Intervention |
|---|---|
| A0 | Complete Apex |
| V1 | Constant recency scale 5 |
| V2 | Phase hysteresis disabled |
| V3 | Heterogeneity scaling disabled |
| V4 | Reward smoothing and variance-floor regularization disabled |
| V5 | Confidence-dependent blend disabled |
| V6 | Same raw weights (0.5, 0.3, 0.2) in every phase; heterogeneity scaling still applies |
| V7 | Diversity disabled and its weight reassigned to utility |
| V8 | Sampling blend set to zero; the remaining random draw still consumes RNG |

V1 is mathematically identical to A0 when N/K=5. The manifest records such cases as aliases instead of rerunning them when both variants are requested. The E4 YAML contains 85 interventions: V2–V8 on C0, V1–V8 on C4, and V3 on C2/C3. It omits the existing A0 references and the redundant C0/V1 intervention. The historical E4 archive contains 85 newly executed jobs plus 20 reused A0 references and five identity aliases. To reproduce that comparison, run E4 and use the matching S5 A0 jobs from E1–E3.

## Reward and feature controls

E5 uses C0/C4 with the five standard seeds, 500 feedback examples, and a matched 10% local holdout for each client with at least two examples. Holdouts are excluded from local training. Credits are individualized in this study and consumed directly, without another division by K:

| Reward | Per-client credit |
|---|---|
| R0 | Global feedback accuracy change divided by K |
| R1 | Clipped relative feedback loss reduction divided by K |
| R2 | Clipped relative pre/post local-model loss reduction on that client's holdout, divided by K |
| R3 | Clipped difference between leave-one-client-out cohort loss and full-cohort loss, normalized by preceding feedback loss and divided by K |

Missing singleton local probes do not generate an observation. R1–R3 are clipped to [-1,1] before division by K. C0 runs for 200 rounds and C4 for 300, totaling 10,000 rounds across 40 jobs. The primary protocol uses the full feedback split and no local probes; compare E5 variants within their matched allocation.

E7 assigns half the clients to a grayscale view of CIFAR-10 while retaining the label partition. Its two controls include or omit label histograms in the proxy. It reports the ordinary test and a separately constructed half-color/half-grayscale test. It does not establish general feature-shift robustness.

## Integrity, recovery, and outputs

Preparation writes the source hash and snapshot hash into each manifest job. Execution rejects a source change after preparation and rejects changed raw data. Reprepare into a new output directory after source/configuration changes.

```text
outputs/example/
├── manifest.lock.json
├── manifest.<digest>.lock.json
├── snapshots/<digest>.pt
├── snapshots/<digest>.pt.sha256.json
├── job_ledger.json
└── runs/<run-id>/
    ├── config_effective.json
    ├── provenance.json
    ├── partition_indices.json
    ├── split_indices.json
    ├── partition_summary.json
    ├── environment_trace.json
    ├── metrics.jsonl
    ├── selection_history.jsonl
    ├── selector_diagnostics.jsonl
    ├── communication_accounting.jsonl
    ├── independent_audit.json
    ├── checkpoints/
    ├── latest_recovery.json
    ├── tuning_selected.json
    ├── completion.json
    └── status.json
```

`independent_audit.json` is present only when configured. Recovery checkpoints are saved every five rounds or five minutes, plus at clean interruption/completion. Scientific checkpoints use the predetermined protocol schedule. Recovery verifies hashes, configuration, source, snapshot, and hardware/software identity, restores model/client/selector/RNG state, and truncates logs to the last verified boundary. Incomplete rounds can be recomputed. `--allow-hardware-change` records an explicit environment change; it does not promise numerical equivalence.

`python -m apex verify MANIFEST` checks all locked jobs, artifact hashes, finite final metrics, and complete round sequences. Retain snapshots and checkpoints if future recovery is needed. The reviewer archives contain compact results and receipts rather than training tensors and cannot resume training.

Communication fields are declared logical payload accounting. Channel speeds and duration indices are synthetic environment variables. Neither measures real distributed bandwidth or power. Shared-GPU round timing and controlled CPU selector timing answer different questions and must not be pooled.
