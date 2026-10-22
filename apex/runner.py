from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from . import datasets as dset
from .aggregation import fedavg
from .client import ClientInfo
from .models import get_model
from . import selector
from .config import PROTOCOL, identity_config, resolve
from .protocol import (
    atomic_json,
    build_snapshot,
    canonical,
    configure_torch,
    digest,
    environment,
    evaluate,
    file_hash,
    isolated_rng,
    load_tensor_file,
    loader,
    local_train,
    participation,
    phase_schedule,
    provenance,
    restore_rng,
    rng_state,
    save_tensor_file,
    scientific_rounds,
    seed_for,
    state_hash,
    DATASETS,
)

LOGS = ("metrics", "selection_history", "selector_diagnostics", "communication_accounting")


class GrayView(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        mean = x.new_tensor(dset._CIFAR10_MEAN).view(3, 1, 1)
        std = x.new_tensor(dset._CIFAR10_STD).view(3, 1, 1)
        rgb = x * std + mean
        gray = (0.2989 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).expand_as(rgb)
        return ((gray - mean) / std, y)


class MixedView(Dataset):

    def __init__(self, dataset, seed):
        self.dataset, self.gray = (dataset, GrayView(dataset))
        order = np.random.default_rng(seed).permutation(len(dataset))
        self.indices = set(order[: len(dataset) // 2].tolist())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.gray[index] if index in self.indices else self.dataset[index]


def load_data(name, root=None, download=False):
    train = dset.get_dataset(name, True, root=root, download=download)
    evaluation = deepcopy(train)
    evaluation.transform = dset.get_transforms(name.lower(), train=False)
    test = dset.get_dataset(name, False, root=root, download=download)
    expected = DATASETS[name]
    if len(train) != expected[1] or len(test) != expected[2]:
        raise ValueError("Official dataset counts differ from the frozen protocol")
    return (train, evaluation, test)


def data_fingerprint(dataset):
    data = getattr(dataset, "data", None)
    if data is None:
        raise ValueError("Dataset must expose official raw image data for provenance")
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(data).tobytes())
    h.update(np.asarray(dataset.targets, dtype=np.int64).tobytes())
    return h.hexdigest()


def snapshot_config(config):
    keys = (
        "dataset",
        "model",
        "num_classes",
        "normalization",
        "partition",
        "alpha",
        "total_clients",
        "clients_per_round",
        "rounds",
        "seed",
        "split_seed",
        "local_probe",
        "feedback_size",
        "feature_shift",
    )
    return {k: config[k] for k in keys}


def prepare_campaign(campaign, output, data_root=None, download=False):
    output = Path(output).resolve()
    configure_torch("cpu")
    code = provenance()
    cache, snapshots, jobs = ({}, {}, [])
    for job in campaign["jobs"]:
        config = job["config"]
        if config["dataset"] not in cache:
            cache[config["dataset"]] = load_data(config["dataset"], data_root, download)
        train, evaluation, test = cache[config["dataset"]]
        key = digest({"recipe": snapshot_config(config), "source": code["source_hash"]})
        path = output / "snapshots" / f"{key}.pt"
        if key not in snapshots:
            if path.exists():
                snapshot = load_tensor_file(path)
            else:
                labels = [int(y) for y in train.targets]
                snapshot = build_snapshot(config, labels)
                snapshot["data_hashes"] = dict(
                    train=data_fingerprint(train), test=data_fingerprint(test)
                )
                snapshot["snapshot_config_hash"] = key
                snapshot["hash"] = digest(
                    {k: v for k, v in snapshot.items() if k not in ("initial_model", "hash")}
                )
                save_tensor_file(path, snapshot)
            if snapshot["snapshot_config_hash"] != key or snapshot["data_hashes"][
                "train"
            ] != data_fingerprint(train):
                raise ValueError("Prepared snapshot differs from the requested data")
            snapshots[key] = snapshot
        snapshot = snapshots[key]
        run_id = digest(
            dict(
                scientific=job["scientific_hash"],
                source=code["source_hash"],
                snapshot=snapshot["hash"],
            )
        )
        jobs.append(
            {
                **job,
                "run_id": run_id,
                "snapshot": str(path.relative_to(output)).replace("\\", "/"),
                "snapshot_hash": snapshot["hash"],
                "source_hash": code["source_hash"],
                "run_directory": f"runs/{run_id}",
            }
        )
    locked = {**campaign, "jobs": jobs, "code": code, "status": "prepared"}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"manifest.{digest(locked)}.lock.json", locked)
    atomic_json(output / "manifest.lock.json", locked)
    return locked


class ApexRunner:

    def __init__(
        self,
        config,
        snapshot,
        train,
        evaluation,
        test,
        output,
        device="cpu",
        source=None,
        model_factory=None,
        allow_hardware_change=False,
    ):
        self.config = resolve(config)
        self.snapshot = snapshot
        self.train, self.evaluation, self._test = (train, evaluation, test)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = device
        configure_torch(device)
        self.source = source or provenance()
        self.hardware = environment()
        self.allow_hardware_change = allow_hardware_change
        with isolated_rng(seed_for(config["seed"], "model")):
            self.model = (
                model_factory()
                if model_factory
                else get_model(
                    config["model"],
                    config["dataset"],
                    config["num_classes"],
                    normalization=config["normalization"],
                )
            )
        self.model.load_state_dict(snapshot["initial_model"])
        self.model.to(device)
        self.engine = None
        if self.config["training_engine"] != "reference":
            from .fast_training import CudaTraining

            self.engine = CudaTraining(self.config, train, evaluation, test, device)
            self.model.to(memory_format=torch.channels_last)
        self.hardware.update(
            training_engine=self.config["training_engine"],
            training_compute_precision="bfloat16" if self.engine else "float32",
            feedback_compute_precision="bfloat16" if self.engine else "float32",
            test_compute_precision="float32",
            parameter_precision="float32",
        )
        self.clients = [
            ClientInfo(cid, len(ids), label_histogram=deepcopy(snapshot["histograms"][cid]))
            for cid, ids in snapshot["partitions"].items()
        ]
        self.rng = random.Random(seed_for(config["seed"], "selector"))
        self.auxiliary_rng = random.Random(seed_for(config["seed"], "auxiliary"))
        self.history = dict(state={}, selected=[])
        self.next_round, self.best_tuning = (0, None)
        self.feedback_ids = self.feedback_subset()
        self.science = scientific_rounds(config["rounds"])
        self.feature_clients = (
            set(
                random.Random(seed_for(config["seed"], "feature_shift")).sample(
                    list(range(len(self.clients))), len(self.clients) // 2
                )
            )
            if config["feature_shift"]
            else set()
        )
        self.cost = self.new_cost()
        self.last_feedback = None
        self.last_full_feedback = None
        self._set_runtime()

    @staticmethod
    def new_cost():
        return dict(
            loss_query_clients=0,
            loss_query_examples=0,
            query_broadcast_bytes=0,
            refresh_updates=0,
            refresh_examples=0,
            refresh_steps=0,
            support_seconds=0.0,
            probe_examples=0,
            loco_evaluations=0,
        )

    @property
    def model_bytes(self):
        state = self.model.state_dict()
        descriptor = {k: dict(dtype=str(v.dtype), shape=list(v.shape)) for k, v in state.items()}
        return (
            8
            + len(canonical(descriptor).encode())
            + sum((v.numel() * v.element_size() for v in state.values()))
        )

    def feedback_subset(self):
        ids = self.snapshot["split"]["feedback"]
        size = self.config["feedback_size"]
        if size is None or size >= len(ids):
            return ids
        labels = np.asarray(self.evaluation.targets)
        rng = np.random.default_rng(seed_for(self.config["split_seed"], "feedback_subset"))
        groups = [
            rng.permutation([i for i in ids if labels[i] == c]).tolist()
            for c in range(self.config["num_classes"])
        ]
        selected = []
        while len(selected) < size:
            for group in groups:
                if group and len(selected) < size:
                    selected.append(group.pop())
        return sorted(selected)

    def client_dataset(self, cid, diagnostic=False):
        dataset = self.evaluation if diagnostic else self.train
        return GrayView(dataset) if cid in self.feature_clients else dataset

    def eval_ids(self, dataset, ids, model=None):
        if self.engine is not None:
            return self.engine.evaluate(model or self.model, dataset, ids)
        return evaluate(
            model or self.model,
            loader(dataset, ids, self.config["evaluation_batch_size"]),
            self.config["num_classes"],
            self.device,
        )

    def _set_runtime(self):
        self.history["runtime"] = {}

    def log(self, name, value):
        with (self.output / f"{name}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")

    def export_state(self):
        return dict(
            protocol=PROTOCOL,
            config_hash=digest(identity_config(self.config)),
            source_hash=self.source["source_hash"],
            snapshot_hash=self.snapshot["hash"],
            next_round=self.next_round,
            environment_cursor=self.next_round,
            model={k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            clients=[asdict(c) for c in self.clients],
            history={k: deepcopy(v) for k, v in self.history.items() if k != "runtime"},
            selector_rng=self.rng.getstate(),
            auxiliary_rng=self.auxiliary_rng.getstate(),
            all_rng=rng_state(),
            last_feedback=deepcopy(self.last_feedback),
            last_full_feedback=deepcopy(self.last_full_feedback),
            best_tuning=deepcopy(self.best_tuning),
            hardware=self.hardware,
            log_offsets={
                name: (
                    (self.output / f"{name}.jsonl").stat().st_size
                    if (self.output / f"{name}.jsonl").exists()
                    else 0
                )
                for name in LOGS
            },
            optimizer_policy="reset SGD each participation; no optimizer survives round boundaries",
        )

    def checkpoint(self, recovery=True, scientific=False):
        checkpoint = self.export_state()
        root = self.output / "checkpoints"
        if scientific:
            save_tensor_file(root / f"round_{self.next_round:05d}.pt", checkpoint)
        if recovery:
            path = root / f"recovery_{self.next_round:05d}.pt"
            save_tensor_file(path, checkpoint)
            load_tensor_file(path)
            atomic_json(
                self.output / "latest_recovery.json",
                {"path": str(path.relative_to(self.output)), "round": self.next_round},
            )
            older = sorted(root.glob("recovery_*.pt"))[:-2]
            for item in older:
                item.unlink()
                item.with_suffix(item.suffix + ".sha256.json").unlink(missing_ok=True)

    def resume(self):
        candidates = sorted((self.output / "checkpoints").glob("recovery_*.pt"), reverse=True)
        checkpoint = None
        for path in candidates:
            try:
                checkpoint = load_tensor_file(path)
                break
            except (ValueError, OSError, json.JSONDecodeError):
                continue
        if checkpoint is None:
            raise ValueError("No verified recovery checkpoint")
        for key, expected in (
            ("protocol", PROTOCOL),
            ("config_hash", digest(identity_config(self.config))),
            ("source_hash", self.source["source_hash"]),
            ("snapshot_hash", self.snapshot["hash"]),
        ):
            if checkpoint[key] != expected:
                raise ValueError(f"Resume {key} mismatch")
        recorded = checkpoint["hardware"]
        if any(
            (recorded.get(k) != self.hardware.get(k) for k in ("torch", "cuda", "cudnn", "gpu"))
        ):
            if not self.allow_hardware_change:
                raise ValueError("Resume hardware/software changed; explicit override required")
            atomic_json(
                self.output / "hardware_change.json", dict(before=recorded, after=self.hardware)
            )
        self.model.load_state_dict(checkpoint["model"])
        self.clients = [ClientInfo(**c) for c in checkpoint["clients"]]
        self.history = deepcopy(checkpoint["history"])
        self.next_round = checkpoint["next_round"]
        self.best_tuning = checkpoint["best_tuning"]
        atomic_json(self.output / "tuning_selected.json", self.best_tuning)
        self.last_feedback, self.last_full_feedback = (
            checkpoint["last_feedback"],
            checkpoint["last_full_feedback"],
        )
        self.rng.setstate(checkpoint["selector_rng"])
        self.auxiliary_rng.setstate(checkpoint["auxiliary_rng"])
        restore_rng(checkpoint["all_rng"])
        self._set_runtime()
        for name, offset in checkpoint["log_offsets"].items():
            path = self.output / f"{name}.jsonl"
            if offset and (not path.exists() or path.stat().st_size < offset):
                raise ValueError(f"Missing checkpoint log prefix: {name}")
            if path.exists():
                with path.open("r+b") as stream:
                    stream.truncate(offset)
        (self.output / "completion.json").unlink(missing_ok=True)

    def initialize(self):
        if any(((self.output / f"{name}.jsonl").exists() for name in LOGS)):
            raise ValueError("Existing job must be resumed, never appended as a fresh run")
        atomic_json(self.output / "config_effective.json", self.config)
        atomic_json(
            self.output / "provenance.json",
            {**self.source, "environment": self.hardware, "snapshot_hash": self.snapshot["hash"]},
        )
        atomic_json(
            self.output / "partition_indices.json",
            {
                "train": self.snapshot["partitions"],
                "probes": self.snapshot["probes"],
                "repairs": self.snapshot["repairs"],
            },
        )
        atomic_json(
            self.output / "split_indices.json",
            {**self.snapshot["split"], "feedback_used": self.feedback_ids},
        )
        atomic_json(
            self.output / "partition_summary.json",
            {**self.snapshot["partition_summary"], "heterogeneity": self.snapshot["heterogeneity"]},
        )
        atomic_json(self.output / "environment_trace.json", self.snapshot["environment"])
        (self.output / "initial_model.sha256").write_text(
            self.snapshot["initial_model_hash"] + "\n"
        )
        self.last_feedback = self.eval_ids(self.evaluation, self.feedback_ids)
        self.last_full_feedback = (
            self.last_feedback
            if self.feedback_ids == self.snapshot["split"]["feedback"]
            else self.eval_ids(self.evaluation, self.snapshot["split"]["feedback"])
        )
        self.log(
            "metrics",
            dict(round=0, feedback=self.last_full_feedback, reward_feedback=self.last_feedback),
        )
        self.evaluate_tuning()
        self.independent_audit()
        self.checkpoint(scientific=True)

    def evaluate_tuning(self):
        if self.config.get("independent_audit_every"):
            self.best_tuning = None
            atomic_json(
                self.output / "tuning_selected.json",
                dict(disabled=True, reason="Independent fixed-budget audit"),
            )
            return
        metrics = self.eval_ids(self.evaluation, self.snapshot["split"]["tuning"])
        if self.best_tuning is None or metrics["accuracy"] > self.best_tuning["accuracy"]:
            self.best_tuning = dict(
                round=self.next_round, accuracy=metrics["accuracy"], metrics=metrics
            )
        atomic_json(self.output / "tuning_selected.json", self.best_tuning)

    def independent_audit(self):
        period = self.config.get("independent_audit_every")
        if not period or self.next_round % period:
            return
        path = self.output / "independent_audit.json"
        rows = json.loads(path.read_text())["rows"] if path.exists() else []
        rows = [row for row in rows if row["round"] < self.next_round]
        state = deepcopy(self.model.state_dict())
        modes = [(module, module.training) for module in self.model.modules()]
        random_state = rng_state()
        selector_state, refresh_state = (self.rng.getstate(), self.auxiliary_rng.getstate())
        before = state_hash(state)
        started = time.perf_counter()
        try:
            metrics = self.eval_ids(self.evaluation, self.snapshot["split"]["tuning"])
        finally:
            self.model.load_state_dict(state)
            for module, mode in modes:
                module.training = mode
            restore_rng(random_state)
            self.rng.setstate(selector_state)
            self.auxiliary_rng.setstate(refresh_state)
        assert state_hash(self.model.state_dict()) == before
        rows.append(
            dict(
                round=self.next_round,
                metrics=metrics,
                examples=len(self.snapshot["split"]["tuning"]),
                seconds=time.perf_counter() - started,
            )
        )
        atomic_json(
            path,
            dict(
                split="tuning",
                selection_access=False,
                checkpoint_selection=False,
                period=period,
                rows=rows,
            ),
        )

    def round(self):
        rnd, cfg = (self.next_round, self.config)
        started = time.perf_counter()
        self.cost = self.new_cost()
        stale_before = [
            rnd - c.last_selected_round if c.last_selected_round >= 0 else None
            for c in self.clients
        ]
        for c in self.clients:
            c.compute_speed = self.snapshot["environment"]["speeds"][c.id]
            c.channel_quality = self.snapshot["environment"]["channels"][rnd][c.id]
            c.estimated_duration = (
                c.data_size / c.compute_speed + c.data_size / 1000 / c.channel_quality
            )
        select_start = time.perf_counter()
        select = selector.select_clients
        ids, scores, update = select(
            rnd,
            cfg["clients_per_round"],
            self.clients,
            self.history,
            self.rng,
            **cfg["selector_params"],
        )
        select_seconds = time.perf_counter() - select_start
        if (
            len(ids) != cfg["clients_per_round"]
            or len(set(ids)) != len(ids)
            or any((i < 0 or i >= len(self.clients) for i in ids))
        ):
            raise ValueError("Selector must return exactly K distinct eligible client IDs")
        self.history["state"].update(update or {})
        self.history["selected"].append(ids)
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        states, sizes, local, probes = ([], [], {}, {})
        train_start = time.perf_counter()
        for cid in ids:
            probe_ids = self.snapshot["probes"][cid]
            pre = (
                self.eval_ids(self.client_dataset(cid, True), probe_ids)["loss"]
                if cfg["reward"] == "R2" and probe_ids
                else None
            )
            state, stats = local_train(
                self.model,
                self.client_dataset(cid),
                self.snapshot["partitions"][cid],
                cfg,
                rnd,
                cid,
                self.device,
                engine=self.engine,
            )
            if pre is not None:
                probe_model = deepcopy(self.model)
                probe_model.load_state_dict(state)
                post = self.eval_ids(self.client_dataset(cid, True), probe_ids, probe_model)["loss"]
                probes[cid] = float(np.clip((pre - post) / max(pre, 1e-08), -1, 1)) / len(ids)
                self.cost["probe_examples"] += 2 * len(probe_ids)
            states.append(state)
            sizes.append(self.clients[cid].data_size)
            local[str(cid)] = stats
        train_seconds = time.perf_counter() - train_start
        self.model.load_state_dict(fedavg(states, sizes, client_ids=ids))
        eval_start = time.perf_counter()
        feedback = self.eval_ids(self.evaluation, self.feedback_ids)
        full = (
            feedback
            if self.feedback_ids == self.snapshot["split"]["feedback"]
            else self.eval_ids(self.evaluation, self.snapshot["split"]["feedback"])
        )
        credits = {
            cid: (feedback["accuracy"] - self.last_feedback["accuracy"]) / len(ids) for cid in ids
        }
        if cfg["reward"] == "R1":
            value = float(
                np.clip(
                    (self.last_feedback["loss"] - feedback["loss"])
                    / max(self.last_feedback["loss"], 1e-08),
                    -1,
                    1,
                )
            ) / len(ids)
            credits = dict.fromkeys(ids, value)
        elif cfg["reward"] == "R2":
            credits = {cid: probes.get(cid) for cid in ids}
        elif cfg["reward"] == "R3":
            loco_model = deepcopy(self.model)
            for i, cid in enumerate(ids):
                others = [j for j in range(len(ids)) if j != i]
                sd = (
                    fedavg(
                        [states[j] for j in others],
                        [sizes[j] for j in others],
                        client_ids=[ids[j] for j in others],
                    )
                    if others
                    else before
                )
                loco_model.load_state_dict(sd)
                loss = self.eval_ids(self.evaluation, self.feedback_ids, loco_model)["loss"]
                credits[cid] = float(
                    np.clip(
                        (loss - feedback["loss"]) / max(self.last_feedback["loss"], 1e-08), -1, 1
                    )
                ) / len(ids)
                self.cost["loco_evaluations"] += 1
        feedback_seconds = time.perf_counter() - eval_start
        cohort_loss = sum(
            (local[str(cid)]["loss"] * local[str(cid)]["processed"] for cid in ids)
        ) / sum((local[str(cid)]["processed"] for cid in ids))
        event = dict(
            round=rnd,
            selected=ids,
            credits={str(k): v for k, v in credits.items()},
            cohort_loss=cohort_loss,
            feedback_loss=feedback["loss"],
            local=local,
            durations={str(cid): self.clients[cid].estimated_duration for cid in ids},
        )
        self.last_feedback, self.last_full_feedback = (feedback, full)
        update_start = time.perf_counter()
        self.history["state"].update(
            last_reward=sum((v for v in event["credits"].values() if v is not None)),
            last_credits=event["credits"],
        )
        update_seconds = time.perf_counter() - update_start
        for cid in ids:
            c = self.clients[cid]
            c.last_loss, c.grad_norm = (local[str(cid)]["loss"], local[str(cid)]["grad_norm"])
            c.last_selected_round = rnd
            c.participation_count += 1
        self.next_round += 1
        self.log(
            "metrics",
            dict(
                round=self.next_round,
                feedback=full,
                reward_feedback=feedback,
                cohort_train_loss=cohort_loss,
                participation=participation(self.clients, rnd),
                processed_examples=sum((v["processed"] for v in local.values())),
                local_steps=sum((v["steps"] for v in local.values())),
                abstract_duration_index=max((self.clients[cid].estimated_duration for cid in ids)),
                selection_seconds=select_seconds,
                selection_ms=select_seconds * 1000,
                support_s=self.cost["support_seconds"],
                update_seconds=update_seconds,
                train_seconds=train_seconds,
                feedback_seconds=feedback_seconds,
                round_seconds=time.perf_counter() - started,
                selection_support=self.cost,
            ),
        )
        self.log(
            "selection_history",
            dict(round=self.next_round, ids=ids, sample_sizes=sizes, event=event),
        )
        state_key = "apex_state"
        st = self.history["state"].get(state_key, {})
        diagnostics = st.get("diagnostics", {})
        diagnostics = dict(
            selector_preset=cfg["method"],
            phase=st.get("prev_phase"),
            heterogeneity=st.get("heterogeneity"),
            observation_counts=st.get("ts_n", {}),
        )
        self.log(
            "selector_diagnostics",
            dict(
                round=self.next_round,
                **diagnostics,
                stale_age_at_selection=stale_before,
                missing_dynamic=[c.id for c in self.clients if stale_before[c.id] is None],
                reward_arms=st.get("arms", {}),
                next_phase=st.get("phase", st.get("prev_phase")),
                phase_age=st.get("phase_age"),
                loss_window=st.get("loss_history", [])[
                    -2
                    * cfg["selector_params"].get(
                        "window", cfg["selector_params"].get("W_phase", 5)
                    ) :
                ],
            ),
        )
        metadata = canonical(
            {
                cid: dict(
                    loss=local[str(cid)]["loss"],
                    norm=local[str(cid)]["grad_norm"],
                    duration=self.clients[cid].estimated_duration,
                    size=self.clients[cid].data_size,
                )
                for cid in ids
            }
        ).encode()
        self.log(
            "communication_accounting",
            dict(
                round=self.next_round,
                model_payload_bytes=self.model_bytes,
                download_bytes=self.model_bytes * len(ids),
                upload_bytes=self.model_bytes * len(ids),
                metadata_bytes=len(metadata),
                registration_histogram_bytes=(
                    len(canonical(self.snapshot["histograms"]).encode()) if rnd == 0 else 0
                ),
                query_bytes=self.cost["query_broadcast_bytes"]
                + 8 * self.cost["loss_query_clients"],
                auxiliary_model_bytes=2 * self.model_bytes * self.cost["refresh_updates"],
            ),
        )
        self.independent_audit()
        if self.next_round in self.science:
            self.evaluate_tuning()
            self.checkpoint(recovery=False, scientific=True)

    def run(
        self,
        resume=False,
        max_rounds=None,
        work_seconds=None,
        export_reserve=300,
        round_seconds_bound=60,
    ):
        if (self.output / "completion.json").exists():
            return verify_completion(
                self.output,
                self.config,
                source_hash=self.source["source_hash"],
                snapshot_hash=self.snapshot["hash"],
            )
        if resume:
            self.resume()
        else:
            self.initialize()
        started, last_save = (time.monotonic(), time.monotonic())
        durations = []
        stop = (
            self.config["rounds"]
            if max_rounds is None
            else min(self.config["rounds"], self.next_round + max_rounds)
        )
        try:
            while self.next_round < stop:
                if work_seconds is not None:
                    estimate = max(durations, default=round_seconds_bound) * 1.25
                    if time.monotonic() - started + estimate + export_reserve >= work_seconds:
                        break
                before = time.monotonic()
                self.round()
                durations.append(time.monotonic() - before)
                if self.next_round % 5 == 0 or time.monotonic() - last_save >= 300:
                    self.checkpoint()
                    last_save = time.monotonic()
            self.checkpoint()
            if self.next_round == self.config["rounds"]:
                return self.finish()
            atomic_json(
                self.output / "status.json", dict(status="interrupted", round=self.next_round)
            )
            return dict(status="interrupted", round=self.next_round)
        except BaseException as error:
            atomic_json(
                self.output / "status.json",
                dict(status="failed", round=self.next_round, error=str(error)),
            )
            raise

    def finish(self):
        test_started = time.perf_counter()
        final = self.eval_ids(self._test, range(len(self._test)))
        test_seconds = time.perf_counter() - test_started
        result = dict(
            status="completed",
            round=self.next_round,
            test=final,
            test_evaluation_seconds=test_seconds,
            config_hash=digest(identity_config(self.config)),
            source_hash=self.source["source_hash"],
            model_hash=state_hash(self.model.state_dict()),
            snapshot_hash=self.snapshot["hash"],
            tuning_selected=self.best_tuning,
        )
        if self.config["feature_shift"]:
            result["test_mixed_views"] = self.eval_ids(
                MixedView(self._test, seed_for(self.config["split_seed"], "test_views")),
                range(len(self._test)),
            )
        result["files"] = {
            str(p.relative_to(self.output)).replace("\\", "/"): file_hash(p)
            for p in sorted(self.output.rglob("*"))
            if p.is_file() and p.name not in ("completion.json", "status.json", "worker.active")
        }
        atomic_json(self.output / "completion.json", result)
        atomic_json(self.output / "status.json", dict(status="completed", round=self.next_round))
        return result


def verify_completion(output, config=None, *, source_hash=None, snapshot_hash=None):
    output = Path(output)
    result = json.loads((output / "completion.json").read_text())
    if result["status"] != "completed" or (
        config is not None
        and (
            result["round"] != config["rounds"]
            or result["config_hash"] != digest(identity_config(config))
        )
    ):
        raise ValueError("Incomplete or mismatched run")
    if source_hash is not None and result["source_hash"] != source_hash:
        raise ValueError("Run differs from locked source")
    if snapshot_hash is not None and result["snapshot_hash"] != snapshot_hash:
        raise ValueError("Run differs from locked snapshot")
    required = {f"{name}.jsonl" for name in LOGS} | {
        "config_effective.json",
        "provenance.json",
        "tuning_selected.json",
    }
    required |= {f"checkpoints/round_{rnd:05d}.pt" for rnd in scientific_rounds(result["round"])}
    if not required.issubset(result["files"]):
        raise ValueError("Completion index lacks required logs/checkpoints")
    for name, expected in result["files"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"Artifact checksum failed: {name}")
    for name in LOGS:
        records = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        if [m["round"] for m in records] != list(
            range(0 if name == "metrics" else 1, result["round"] + 1)
        ):
            raise ValueError(f"Missing or duplicate rounds: {name}")
    if not all((np.isfinite(result["test"][key]) for key in ("accuracy", "macro_f1", "loss"))):
        raise ValueError("Invalid final test metrics")
    return result
