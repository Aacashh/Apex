from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import shutil
import uuid
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from .client import ClientInfo
from .models import get_model
from .metadata import heterogeneity, histogram

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "CIFAR10": (10, 50000, 10000),
    "CIFAR100": (100, 50000, 10000),
    "MNIST": (10, 60000, 10000),
    "Fashion-MNIST": (10, 60000, 10000),
}
S5 = [42, 123, 456, 789, 1024]
S10 = S5 + [2025, 3141, 4096, 5678, 6789]
DEV = [7, 17, 27]


@contextmanager
def temporary_directory(parent, prefix="apex_tmp_"):
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (prefix + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        if not path.resolve().is_relative_to(parent):
            raise ValueError("Temporary cleanup escaped its intended directory")
        shutil.rmtree(path)


def seed_for(seed, *parts):
    raw = json.dumps([int(seed), *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def save_tensor_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
    os.replace(temp, path)
    atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": file_hash(path)})


def load_tensor_file(path):
    path = Path(path)
    expected = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())["sha256"]
    if file_hash(path) != expected:
        raise ValueError(f"Checksum failed: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng(seed):
    before = rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        restore_rng(before)


def configure_torch(device, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly")


def provenance():
    files = list((ROOT / "apex").glob("*.py")) + [
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
    ]
    hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(files)}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = (None, None)
    return dict(git_commit=commit, git_dirty=dirty, source_hash=digest(hashes), source_files=hashes)


def environment():
    import torchvision

    info = dict(
        python=platform.python_version(),
        os=platform.platform(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        cpu=platform.processor(),
        cpu_threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=False,
        precision="float32",
    )
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            info["driver"] = result.stdout.strip()
        except FileNotFoundError:
            info["driver"] = None
    return info


def stratified_split(labels, seed=260928):
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(y)
    classes = np.unique(y)
    counts = np.array([(y == c).sum() for c in classes])
    target = int(round(n * 0.05))
    exact = counts * target / n
    quota = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - quota), kind="stable")[: target - quota.sum()]:
        quota[index] += 1
    train, feedback, tuning = ([], [], [])
    for c, k in zip(classes, quota):
        ids = rng.permutation(np.flatnonzero(y == c)).tolist()
        if 2 * k >= len(ids):
            raise ValueError("Too few examples to form disjoint stratified validation sets")
        feedback.extend(ids[:k])
        tuning.extend(ids[k : 2 * k])
        train.extend(ids[2 * k :])
    return {"train": sorted(train), "feedback": sorted(feedback), "tuning": sorted(tuning)}


def partition_samples(labels, ids, N, mode, alpha, seed):
    rng = np.random.default_rng(seed)
    bins = {i: [] for i in range(N)}
    if len(ids) < N:
        raise ValueError("Fewer training examples than clients")
    if mode == "iid":
        order = rng.permutation(ids).tolist()
        bins = {i: order[i::N] for i in range(N)}
    else:
        for c in sorted(set(labels)):
            indices = rng.permutation([i for i in ids if labels[i] == c])
            proportions = rng.dirichlet(np.full(N, alpha))
            parts = np.split(indices, (np.cumsum(proportions) * len(indices)).astype(int)[:-1])
            for cid, samples in enumerate(parts):
                bins[cid].extend(samples.tolist())
    repairs = []
    for cid in range(N):
        if not bins[cid]:
            donor = max(range(N), key=lambda i: (len(bins[i]), -i))
            sample = bins[donor].pop()
            bins[cid].append(sample)
            repairs.append(dict(client=cid, donor=donor, sample=sample))
    return (bins, repairs)


def environment_trace(seed, N, T):
    rng = np.random.default_rng(seed)
    speeds = np.maximum(0.1, rng.lognormal(0, 0.5, N))
    channel = rng.uniform(0.5, 1.0, N)
    initial = channel.copy()
    draws = rng.uniform(-0.05, 0.05, (T, N))
    trace = []
    for increments in draws:
        channel = np.clip(channel + increments, 0.1, 1.5)
        trace.append(channel.copy().tolist())
    return {"speeds": speeds.tolist(), "initial_channel": initial.tolist(), "channels": trace}


def scientific_rounds(T):
    return sorted(
        {
            r
            for r in [
                0,
                1,
                10,
                25,
                50,
                100,
                150,
                200,
                *range(250, T + 1, 50),
                T - 20,
                T - 15,
                T - 10,
                T - 5,
                T,
            ]
            if 0 <= r <= T
        }
    )


def phase_schedule(T, seed, shuffled=False):
    names = ["critical"] * int(0.3 * T) + ["transition"] * (int(0.7 * T) - int(0.3 * T))
    names += ["exploitation"] * (T - len(names))
    if shuffled:
        count = T // 3
        critical = round(0.3 * count)
        transition = round(0.4 * count)
        blocks = [["critical"] * 3 for _ in range(critical)]
        blocks += [["transition"] * 3 for _ in range(transition)]
        blocks += [["exploitation"] * 3 for _ in range(count - critical - transition)]
        random.Random(seed).shuffle(blocks)
        names = [p for block in blocks for p in block] + ["exploitation"] * (T % 3)
    return names


def build_snapshot(config, labels, model_factory=None):
    classes = config["num_classes"]
    split = stratified_split(labels, config["split_seed"])
    bins, repairs = partition_samples(
        labels,
        split["train"],
        config["total_clients"],
        config["partition"],
        config["alpha"],
        seed_for(config["seed"], "partition"),
    )
    probes = {cid: [] for cid in bins}
    if config["local_probe"]:
        for cid, ids in bins.items():
            if len(ids) >= 2:
                rng = np.random.default_rng(seed_for(config["seed"], "probe", cid))
                k = min(len(ids) - 1, max(1, int(len(ids) * 0.1)))
                probes[cid] = sorted(rng.choice(ids, k, replace=False).tolist())
                bins[cid] = [i for i in ids if i not in set(probes[cid])]
    clients = [
        ClientInfo(
            cid,
            len(ids),
            label_histogram={c: int(np.sum(np.asarray(labels)[ids] == c)) for c in range(classes)},
        )
        for cid, ids in bins.items()
    ]
    with isolated_rng(seed_for(config["seed"], "model")):
        model = (
            model_factory()
            if model_factory
            else get_model(
                config["model"], config["dataset"], classes, normalization=config["normalization"]
            )
        )
    init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    snapshot = dict(
        split=split,
        partitions=bins,
        probes=probes,
        repairs=repairs,
        labels_hash=digest([int(y) for y in labels]),
        histograms={c.id: c.label_histogram for c in clients},
        environment=environment_trace(
            seed_for(config["seed"], "environment"), len(clients), config["rounds"]
        ),
        initial_model=init,
        initial_model_hash=state_hash(init),
        heterogeneity=heterogeneity(clients, classes, seed_for(config["seed"], "js_pairs")),
    )
    sizes = np.array([c.data_size for c in clients])
    snapshot["partition_summary"] = dict(
        min=int(sizes.min()),
        size_cv=float(sizes.std() / sizes.mean()),
        size_quantiles=np.quantile(sizes, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        labels_per_client=[sum((v > 0 for v in c.label_histogram.values())) for c in clients],
        mean_label_entropy=float(
            np.mean(
                [
                    -sum(
                        (
                            p * np.log2(p)
                            for p in histogram(c.label_histogram, classes) / c.data_size
                            if p > 0
                        )
                    )
                    for c in clients
                ]
            )
        ),
        repairs=len(repairs),
        sparse_probe_clients=sum((not p for p in probes.values())) if config["local_probe"] else 0,
    )
    snapshot["hash"] = digest({k: v for k, v in snapshot.items() if k != "initial_model"})
    return snapshot


def loader(dataset, ids, batch_size, seed=0, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(ids)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, batches, classes, device):
    was_training = model.training
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    loss_sum, count = (0.0, 0)
    try:
        for x, y in batches:
            logits = model(x.to(device))
            loss_sum += float(nn.functional.cross_entropy(logits, y.to(device), reduction="sum"))
            labels, predicted = (y.numpy(), logits.argmax(1).cpu().numpy())
            np.add.at(confusion, (labels, predicted), 1)
            count += len(y)
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Empty evaluation dataset")
    tp = np.diag(confusion)
    support, predicted = (confusion.sum(1), confusion.sum(0))
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(
        2 * tp, support + predicted, out=np.zeros(classes), where=support + predicted > 0
    )
    return dict(
        accuracy=float(tp.sum() / count),
        loss=loss_sum / count,
        macro_f1=float(f1.mean()),
        balanced_accuracy=float(recall.mean()),
        recall=recall.tolist(),
        worst_class_recall=float(recall.min()),
        confusion=confusion.tolist(),
        examples=count,
    )


@contextmanager
def singleton_batchnorm(model, counter):
    handles, frozen = ([], [])

    def before(module, inputs):
        x = inputs[0]
        if module.training and x.numel() // x.shape[1] <= 1:
            module.training = False
            frozen.append(module)
            counter["bn_frozen_batches"] += 1

    def after(module, inputs, output):
        if module in frozen:
            module.training = True
            frozen.remove(module)

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            handles.extend(
                [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
            )
    try:
        yield
    finally:
        for module in frozen:
            module.training = True
        for handle in handles:
            handle.remove()


def local_train(model, dataset, ids, config, round_idx, cid, device, stream="train", engine=None):
    if config.get("training_engine", "reference") != "reference":
        if engine is None:
            raise ValueError("The configured CUDA training engine was not initialized")
        return engine.train_client(model, dataset, ids, config, round_idx, cid, device, stream)
    local = deepcopy(model).to(device).train()
    opt = torch.optim.SGD(
        local.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    stats = dict(loss=0.0, norm_sq=0.0, steps=0, processed=0, bn_frozen_batches=0, loss_sq=0.0)
    with singleton_batchnorm(local, stats):
        for epoch in range(config["local_epochs"]):
            seed = seed_for(config["seed"], stream, round_idx, cid, epoch)
            with isolated_rng(seed):
                batches = loader(dataset, ids, config["batch_size"], seed, True)
                for x, y in batches:
                    opt.zero_grad(set_to_none=True)
                    logits = local(x.to(device))
                    per_example = nn.functional.cross_entropy(
                        logits, y.to(device), reduction="none"
                    )
                    loss = per_example.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite local loss for client {cid}")
                    loss.backward()
                    norm_sq = sum(
                        (
                            float(p.grad.detach().square().sum())
                            for p in local.parameters()
                            if p.grad is not None
                        )
                    )
                    if not np.isfinite(norm_sq):
                        raise FloatingPointError(f"Nonfinite gradient for client {cid}")
                    opt.step()
                    stats["loss"] += float(loss.detach()) * len(y)
                    stats["loss_sq"] += float(per_example.detach().square().sum())
                    stats["norm_sq"] += norm_sq
                    stats["steps"] += 1
                    stats["processed"] += len(y)
    stats["loss"] /= stats["processed"]
    stats["grad_norm"] = (stats.pop("norm_sq") / stats["steps"]) ** 0.5
    stats["oort_utility"] = len(ids) * (stats.pop("loss_sq") / stats["processed"]) ** 0.5
    state = {k: v.detach().cpu().clone() for k, v in local.state_dict().items()}
    return (state, stats)


def participation(clients, round_idx):
    counts = np.array([c.participation_count for c in clients], float)
    sorted_counts = np.sort(counts)
    n = len(counts)
    gini = (
        float(np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_counts) / (n * counts.sum()))
        if counts.sum()
        else 0.0
    )
    waits = [
        round_idx - c.last_selected_round if c.last_selected_round >= 0 else round_idx + 1
        for c in clients
    ]
    p = counts / max(counts.sum(), 1)
    return dict(
        gini=gini,
        never_selected=float(np.mean(counts == 0)),
        distinct=int(np.sum(counts > 0)),
        p95_wait=float(np.quantile(waits, 0.95)),
        max_wait=max(waits),
        entropy=float(-sum((v * np.log(v) for v in p if v > 0))),
        counts=counts.astype(int).tolist(),
    )
