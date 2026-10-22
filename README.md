# Apex

**Adaptive Phase-aware EXploration for federated client selection.** Apex selects a cohort using learning feedback, diversity of client proxies, and time since participation. It trains no neural selection policy. This repository contains the selector, dataset preparation, local training, weighted model aggregation, checkpoint recovery, and reproducible experiment configurations.

Author and maintainer: [Aacashh](https://github.com/Aacashh) (Aakash Sharma).

## Install

Use Python 3.11. From this repository:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

The pinned dependencies are PyTorch 2.6.0, torchvision 0.21.0, NumPy 2.2.6, SciPy 1.15.3, and PyYAML 6.0.2. A CUDA-capable PyTorch build and compatible NVIDIA driver are needed for GPU execution. The reference engine works on CPU or CUDA; the explicit BF16 engine requires a GPU with native BF16 support, such as an A100.

## Run with the datasets

| Dataset | Official train / test | Model | Configuration |
|---|---:|---|---|
| CIFAR-10 | 50,000 / 10,000 | ResNet18 adapted to 32 × 32 images | `configs/cifar10.yaml` |
| CIFAR-100 | 50,000 / 10,000 | ResNet18 adapted to 32 × 32 images | `configs/cifar100.yaml` |
| Fashion-MNIST | 60,000 / 10,000 | Two-convolution CNN | `configs/fashion-mnist.yaml` |
| MNIST | 60,000 / 10,000 | Two-convolution CNN | `configs/mnist.yaml` |

Data are loaded through torchvision. Existing datasets can be supplied with `--data-root`; use `--download` during preparation to obtain missing official files. Dataset files and trained weights are excluded from Git.

For example, to run Fashion-MNIST:

```bash
python -m apex validate configs/fashion-mnist.yaml
python -m apex prepare configs/fashion-mnist.yaml --output outputs/fashion-mnist --data-root data --download
python -m apex run-all outputs/fashion-mnist/manifest.lock.json --device cpu --data-root data
python -m apex verify outputs/fashion-mnist/manifest.lock.json
```

For CIFAR-10 on an A100:

```bash
python -m apex prepare configs/cifar10.yaml --output outputs/cifar10 --data-root data --download
python -m apex run-all outputs/cifar10/manifest.lock.json --device cuda --data-root data
python -m apex verify outputs/cifar10/manifest.lock.json
```

Replace the configuration and output directory to use CIFAR-100 or MNIST. The four simple configurations use seed 42 and the reference FP32 engine. For the recorded A100 numerical profile, use the study configurations below. Dataset transforms, split policy, precision profiles, and the exact commands are detailed in [the reproduction guide](docs/REPRODUCIBILITY.md).

`validate` and `dry-run` check a configuration and report job/round counts without loading data or training. `prepare` creates deterministic partitions, initial models, data fingerprints, and a source-locked manifest. `run-all` executes jobs sequentially and resumes jobs with recovery pointers. It skips completed jobs after verification. It does not allocate or bill a cloud session.

To run or resume one job, copy its `run_id` from the locked manifest:

```bash
python -m apex run outputs/cifar10/manifest.lock.json --run-id RUN_ID --device cuda --data-root data
python -m apex resume outputs/cifar10/manifest.lock.json --run-id RUN_ID --device cuda --data-root data
```

## Method and mathematics

Let $`N`$ be the number of clients, $`K`$ the cohort size, and $`S_t`$ the selected clients at round $`t`$. Client $`i`$ has $`n_i`$ training examples. The training objective is

```math
\min_w F(w)=\sum_{i=1}^{N}\frac{n_i}{\sum_j n_j}F_i(w),\qquad
F_i(w)=\frac{1}{n_i}\sum_{z\in D_i}\ell(w;z).
```

Each selected client starts from the current global model and performs local SGD. Floating-point model state is aggregated in ascending client-ID order:

```math
w_{t+1}=\sum_{i\in S_t}\frac{n_i}{\sum_{j\in S_t}n_j}w_{t,i}.
```

Integer buffers are copied from the lowest-ID selected client. Models start from random initialization; each participation gets a fresh SGD optimizer.

### 1. Client context and diversity proxy

Client context combines min–max normalized last local training loss, gradient norm, inverse estimated duration, and data size:

```math
u_i=0.4\widetilde\ell_i+0.2\widetilde g_i+
0.2\widetilde s_i+0.2\widetilde n_i.
```

Loss and gradient normalization use clients with an existing reward observation. Unobserved clients receive the neutral value $`0.5`$ for these two features; a constant observed feature normalizes to zero. Losses and gradient norms come from previous participations and can become stale. Estimated duration is an abstract compute/channel trace, rather than a measured network latency.

Let $`h_i`$ be the label-count vector and $`\widehat h_i=h_i/\|h_i\|_2`$. The complete proxy is normalized after concatenation:

```math
v_i=\frac{[\widetilde\ell_i,\widetilde g_i,\widehat h_i]}
{\|[\widetilde\ell_i,\widetilde g_i,\widehat h_i]\|_2}.
```

A zero vector stays zero. With selected cohort $`S`$, the diversity bonus is

```math
d_i(S)=\begin{cases}
0,&S=\varnothing,\\
\mathrm{clip}\left(1-\max_{j\in S}v_i^\top v_j,0,2\right),&S\ne\varnothing.
\end{cases}
```

The maximum similarity implements distance to the **nearest** selected proxy. Label histograms describe label mixture; this proxy is not the full gradient direction.

### 2. Reward observations and Thompson-style sampling

For the primary reward, the feedback-set accuracy improvement is $`r_t=A_{t+1}-A_t`$, measured as a fraction. Every selected client receives the same cohort credit $`q_{i,t}=r_t/K`$. Credits update the selector at the next selection call, so the final round has no subsequent posterior update.

Credits are smoothed with $`\rho=0.3`$:

```math
e_{i,m}=\rho q_{i,m}+(1-\rho)e_{i,m-1},\qquad e_{i,1}=q_{i,1}.
```

For observation count $`m`$, running mean and regularized variance follow the implemented recurrence:

```math
\delta=e_{i,m}-\mu_{i,m-1},\qquad
\mu_{i,m}=\mu_{i,m-1}+\delta/m,
```

```math
V_{i,m}=\max\left\{
\frac{(m-1)V_{i,m-1}+\delta(e_{i,m}-\mu_{i,m})}{m},
\frac{0.1}{\sqrt m}\right\}.
```

The initial mean is zero and initial variance is one. For $`m\ge2`$, draw $`\theta_i\sim\mathcal N(\mu_{i,m},V_{i,m}/m)`$. Before two observations, draw from $`\mathrm{Beta}(a_i,b_i)`$, initially $`(1,1)`$. A positive smoothed credit increments $`a_i`$ by $`\min(e_{i,m},1)`$; a nonpositive credit increments $`b_i`$ by $`\min(|e_{i,m}|,1)`$.

The adaptive blend is

```math
\gamma_i=0.3\left(1-\frac{1}{1+m_i}\right),\qquad
z_i=(1-\gamma_i)u_i+\gamma_i\theta_i.
```

The shared aggregate reward couples clients. These are heuristic exploration statistics, not independently identified causal contributions or an exact independent-arm Bayesian posterior. The variance floor is included in the next recurrence, rather than maintained as a separate unbiased sample variance.

### 3. Phase detection and heterogeneity

Apex tracks the unweighted mean of positive last training losses for the preceding selected cohort. With window $`W=5`$, let $`\bar L_{\rm recent}`$ and $`\bar L_{\rm older}`$ be recent and preceding window means:

```math
\eta_t=\frac{\bar L_{\rm older}-\bar L_{\rm recent}}{\bar L_{\rm older}+10^{-12}},\qquad
c_t=\frac{\mathrm{std}(L_{\rm recent})}{\bar L_{\rm recent}+10^{-12}}.
```

The raw phase is critical when $`\eta_t>0.05`$ and $`c_t>0.10`$; otherwise transition when $`\eta_t>0.01`$; otherwise exploitation. Fewer than five observations retain the critical phase. Between five and nine observations, the older window is the first five observations and may overlap the recent window. Phase changes require the implemented minimum age of three selection calls and cannot jump directly between critical and exploitation.

| Phase | Utility/sampling weight $`a`$ | Diversity weight $`b`$ | Recency weight $`c`$ |
|---|---:|---:|---:|
| Critical | 0.20 | 0.60 | 0.20 |
| Transition | 0.50 | 0.30 | 0.20 |
| Exploitation | 0.70 | 0.15 | 0.15 |

Label heterogeneity is estimated once from up to 200 deterministically spaced client pairs. For probability histograms $`p_i,p_j`$, using natural logarithms,

```math
\mathrm{JSD}(p_i,p_j)=\tfrac12\mathrm{KL}(p_i\|M)+\tfrac12\mathrm{KL}(p_j\|M),
\qquad M=(p_i+p_j)/2,
```

```math
H=\min\left(\frac{\mathrm{mean}_{(i,j)}\sqrt{\mathrm{JSD}(p_i,p_j)}}{0.6},1\right).
```

Missing usable histogram pairs give $`H=0.5`$. The effective weights are $`b'=bH`$, $`a'=a+b-b'`$, and $`c'=c`$. The cached estimate assumes static client label mixtures.

### 4. Recency and greedy cohort construction

Let $`\Delta_i(t)`$ be rounds since selection, or $`t+1`$ for an unseen client. The recency term and selection score are

```math
R_i(t)=\frac{\Delta_i(t)}{\Delta_i(t)+\max(N/K,3)},\qquad
Q_i(S)=a'z_i+b'd_i(S)+cR_i(t).
```

Starting with an empty cohort, repeatedly add the remaining client with the largest $`Q_i(S)`$ until $`K`$ clients are selected. The sampling draw is made once per candidate per round. If no client has a positive loss yet, sample clients without replacement using data-size weights. No hard time budget is enabled in the reproducible runner.

For proxy dimension $`d=C+2`$, the warm selection loop is $`O(NK^2d+Nd)`$ because distances to every selected proxy are recomputed for each remaining candidate at each greedy step. It retains $`O(Nd+N+Kd)`$ working state. Initial heterogeneity estimation can additionally scan $`O(N^2)`$ pair indices and evaluates at most 200 histogram distances. Training and evaluation costs are separate.

## Reproduce the study settings

| Configuration | Apex scope |
|---|---|
| `configs/studies/e1-e3.yaml` | 75 jobs, 16,000 rounds, 13 settings across four datasets |
| `configs/studies/e4-ablations.yaml` | 85 jobs, 21,000 rounds, named component ablations |
| `configs/studies/e5-reward.yaml` | 40 jobs, 10,000 rounds, matched reward/probe comparisons |
| `configs/studies/e7-feature-shift.yaml` | 10 jobs, 2,000 rounds, color/grayscale with and without histogram features |
| `configs/studies/independent-convergence.yaml` | 10 Apex jobs, 2,500 rounds, independent audit curves on C4 and D2 |

Use the same `validate`, `prepare`, `run-all`, and `verify` sequence with a study configuration. The standard five seeds are 42, 123, 456, 789, and 1024. C0 and C4 in the primary study additionally use 2025, 3141, 4096, 5678, and 6789. Preserve the exact seed blocks when comparing archived outcomes.

See [reproduction details](docs/REPRODUCIBILITY.md) for the cell matrix, ablations, reward formulas, and numerical profiles. Historical A100 archives and their **internal directory structures** are documented separately in [the reviewer archive guide](docs/REVIEWER_ARCHIVES.md). Archived results are observations from their recorded source/environment; the standalone namespace produces its own source identity.

## Outputs and interpretation

Each run writes configuration/provenance, split and partition indices, round metrics, selected IDs, selector diagnostics, communication accounting, recovery/scientific checkpoints, and a completion record. Verification checks saved hashes and round sequences. Independent-audit runs additionally write a curve from the held-out audit split while restoring model buffers, modes, and RNG state after each observation.

Apex favors a trade-off among learning progress, diversity, and participation recency. Competitive performance in some settings does not imply universal accuracy or fairness superiority. Recency is a soft preference and does not guarantee bounded waiting. Label histograms do not establish feature-distribution similarity, privacy, or gradient alignment. Selector latency, support work, complete round time, modeled communication, and energy must be assessed separately. The implementation provides no general nonconvex convergence or bandit regret guarantee.

## Code map

| File | Responsibility |
|---|---|
| `apex/selector.py` | Apex scoring, phase state, reward statistics, and greedy selection |
| `apex/client.py` | Client metadata |
| `apex/runner.py` | Local training orchestration, reward feedback, checkpoints, and logs |
| `apex/protocol.py` | Deterministic splits, partitions, seeding, training, and integrity helpers |
| `apex/fast_training.py` | Explicit CIFAR BF16 GPU profile |
| `apex/models.py`, `apex/datasets.py` | The two model families and four dataset loaders |
| `apex/config.py`, `apex/__main__.py` | Validated configurations and CLI |

The source contains no inline comments. Method explanations are maintained in this README and the supporting documentation. Dataset licenses remain with their owners; this repository does not add an open-source license grant.
