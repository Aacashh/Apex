from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from apex import ClientInfo, select_clients
from apex import selector
from apex.config import read_campaign, resolve
from apex.protocol import build_snapshot, configure_torch, state_hash
from apex.runner import ApexRunner, verify_completion


class TinyData(Dataset):
    def __init__(self):
        self.targets = np.arange(120) % 10
        self.data = torch.tensor(
            np.random.default_rng(13).normal(size=(120, 4)), dtype=torch.float32
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])


def model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 10))


class ApexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch("cpu")

    def test_unit_proxy_and_nearest_distance(self):
        np.testing.assert_allclose(
            np.linalg.norm(selector._build_proxy_vector(0.7, 0.2, [1, 2, 3])), 1.0
        )
        self.assertEqual(selector._min_cosine_distance([1, 0], [[1, 0], [0, 1]]), 0.0)
        self.assertEqual(selector._min_cosine_distance([1, 0], [[0, 1]]), 1.0)
        self.assertEqual(selector._min_cosine_distance([1, 0], []), 0.0)
        self.assertEqual(
            selector._observed_norm([0, 2, 4], [False, True, True]), [0.5, 0.0, 2 / (2 + 1e-12)]
        )

    def test_cohort_and_phase_constraints(self):
        clients = [ClientInfo(i, 20 + i, label_histogram={0: 10, 1: 10 + i}) for i in range(12)]
        history = dict(state={}, selected=[])
        rng = random.Random(42)
        for rnd in range(20):
            ids, scores, update = select_clients(rnd, 3, clients, history, rng)
            self.assertEqual(len(ids), 3)
            self.assertEqual(len(set(ids)), 3)
            history["state"].update(update)
            history["selected"].append(ids)
            history["state"]["last_reward"] = 0.01
            for cid in ids:
                clients[cid].last_loss = 1 / (1 + rnd) + cid * 0.01
                clients[cid].grad_norm = 0.3 + cid * 0.02
                clients[cid].participation_count += 1
                clients[cid].last_selected_round = rnd
        phase, age = selector._detect_phase_with_hysteresis(
            [1.0] * 10, 5, 0.05, 0.10, 0.01, "critical", 3
        )
        self.assertEqual(phase, "transition")

    def test_configuration_matrix_and_rejections(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "e1-e3": (75, 16000),
            "e5-reward": (40, 10000),
            "e7-feature-shift": (10, 2000),
            "independent-convergence": (10, 2500),
        }
        for name, (jobs, rounds) in expected.items():
            campaign = read_campaign(root / "configs/studies" / (name + ".yaml"))
            self.assertEqual(len(campaign["jobs"]), jobs)
            self.assertEqual(sum(j["config"]["rounds"] for j in campaign["jobs"]), rounds)
        for values in (
            {"method": "other"},
            {"clients_per_round": 51},
            {"use_test_for_decisions": True},
            {"selector_params": {"gamma": 0.8}},
            {"dataset": "other"},
        ):
            with self.assertRaises(ValueError):
                resolve(values)

    def test_training_resume_and_audit_parity(self):
        data = TinyData()
        cfg = resolve(
            dict(dataset="MNIST", total_clients=4, clients_per_round=2, rounds=6, batch_size=8)
        )
        snapshot = build_snapshot(cfg, data.targets.tolist(), model_factory=model)
        temporary_root = Path(__file__).resolve().parents[1] / "outputs" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temp:
            root = Path(temp)

            def run(name, config, resume=False, max_rounds=None):
                sim = ApexRunner(
                    config,
                    snapshot,
                    data,
                    data,
                    data,
                    root / name,
                    source={"source_hash": "test"},
                    model_factory=model,
                )
                sim.run(resume=resume, max_rounds=max_rounds)
                return sim

            full = run("full", cfg)
            run("resumed", cfg, max_rounds=2)
            resumed = run("resumed", cfg, resume=True)
            audit = run("audit", {**cfg, "independent_audit_every": 1})
            self.assertEqual(full.history["selected"], resumed.history["selected"])
            self.assertEqual(full.history["selected"], audit.history["selected"])
            self.assertEqual(
                state_hash(full.model.state_dict()), state_hash(resumed.model.state_dict())
            )
            self.assertEqual(
                state_hash(full.model.state_dict()), state_hash(audit.model.state_dict())
            )
            verify_completion(
                root / "full", cfg, source_hash="test", snapshot_hash=snapshot["hash"]
            )
            self.assertEqual(
                len(json.loads((root / "audit/independent_audit.json").read_text())["rows"]), 7
            )
            with (root / "full/metrics.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(ValueError):
                verify_completion(root / "full", cfg)


if __name__ == "__main__":
    unittest.main()
