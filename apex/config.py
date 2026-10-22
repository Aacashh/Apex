from copy import deepcopy
import math
from pathlib import Path
import yaml
from .protocol import DATASETS, DEV, S5, S10, digest

PROTOCOL = "apex_standalone_1"
DEFAULTS = dict(
    dataset="CIFAR10",
    model="ResNet18",
    num_classes=10,
    normalization="batchnorm",
    partition="dirichlet",
    alpha=0.3,
    total_clients=50,
    clients_per_round=10,
    rounds=200,
    seed=42,
    split_seed=260928,
    local_epochs=1,
    batch_size=32,
    lr=0.01,
    momentum=0.0,
    weight_decay=0.0,
    precision="float32",
    optimizer="sgd",
    scheduler="constant",
    training_engine="reference",
    evaluation_batch_size=None,
    parallel_clients=0,
    num_workers=0,
    smoke_test_mode=False,
    pretrained=False,
    dp_sigma=0.0,
    dropout_probability=0.0,
    time_budget=None,
    energy_budget=None,
    bytes_budget=None,
    reward="R0",
    local_probe=False,
    feedback_size=None,
    feature_shift=False,
    selector_params={},
    method="apex",
    paper_variant="A0",
    feedback_each_round=True,
    use_test_for_decisions=False,
    minimum_client_samples=1,
    independent_audit_every=None,
)
SELECTOR_DEFAULTS = {
    "W_phase": 5,
    "tau_critical": 0.05,
    "tau_unstable": 0.1,
    "tau_exploit": 0.01,
    "min_dwell": 3,
    "gamma": 0.3,
    "w_loss": 0.4,
    "w_grad": 0.2,
    "w_speed": 0.2,
    "w_data": 0.2,
    "reward_ema_alpha": 0.3,
    "phase_weights_critical": None,
    "phase_weights_transition": None,
    "phase_weights_exploit": None,
    "use_adaptive_recency": True,
    "use_phase_hysteresis": True,
    "use_het_scaling": True,
    "use_posterior_reg": True,
    "use_adaptive_gamma": True,
    "disable_diversity": False,
    "disable_thompson": False,
    "use_histogram": True,
    "individualized_credits": False,
}
ABLATIONS = {
    "A0": {},
    "V1": {"use_adaptive_recency": False},
    "V2": {"use_phase_hysteresis": False},
    "V3": {"use_het_scaling": False},
    "V4": {"use_posterior_reg": False},
    "V5": {"use_adaptive_gamma": False},
    "V6": {
        "phase_weights_critical": (0.5, 0.3, 0.2),
        "phase_weights_transition": (0.5, 0.3, 0.2),
        "phase_weights_exploit": (0.5, 0.3, 0.2),
    },
}
ABLATIONS.update(V7={"disable_diversity": True}, V8={"disable_thompson": True})


def strict_keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown)}")


def resolve(values):
    strict_keys(values, DEFAULTS, "configuration")
    c = {**deepcopy(DEFAULTS), **deepcopy(values)}
    if c["dataset"] not in DATASETS or c["method"] != "apex":
        raise ValueError("Use method apex and one of the four registered datasets")
    c["evaluation_batch_size"] = c["evaluation_batch_size"] or c["batch_size"]
    if "model" not in values:
        c["model"] = "ResNet18" if c["dataset"].startswith("CIFAR") else "CNN-MNIST"
    if "num_classes" not in values:
        c["num_classes"] = DATASETS[c["dataset"]][0]
    if c["num_classes"] != DATASETS[c["dataset"]][0]:
        raise ValueError("Class count differs from the official dataset")
    expected_model = "ResNet18" if c["dataset"].startswith("CIFAR") else "CNN-MNIST"
    if c["model"] != expected_model:
        raise ValueError(f"{c['dataset']} requires {expected_model}")
    for key in (
        "total_clients",
        "clients_per_round",
        "rounds",
        "local_epochs",
        "batch_size",
        "evaluation_batch_size",
    ):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "split_seed"):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if c["clients_per_round"] > c["total_clients"]:
        raise ValueError("K must not exceed N")
    if (
        c["partition"] not in ("iid", "dirichlet")
        or not isinstance(c["alpha"], (float, int))
        or (not math.isfinite(c["alpha"]))
        or (c["alpha"] <= 0)
    ):
        raise ValueError("Use IID or positive Dirichlet alpha")
    if c["normalization"] not in ("batchnorm", "groupnorm"):
        raise ValueError("Unknown normalization")
    for key in ("lr", "momentum", "weight_decay"):
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError(f"Invalid {key}")
    if c["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    required = dict(
        precision="float32",
        optimizer="sgd",
        scheduler="constant",
        parallel_clients=0,
        num_workers=0,
        smoke_test_mode=False,
        pretrained=False,
        dp_sigma=0.0,
        dropout_probability=0.0,
        time_budget=None,
        energy_budget=None,
        bytes_budget=None,
        feedback_each_round=True,
        use_test_for_decisions=False,
        minimum_client_samples=1,
    )
    for key, value in required.items():
        if c[key] != value:
            raise ValueError(f"The reproducible protocol requires {key}={value}")
    for key in ("local_probe", "feature_shift"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("feedback_size", "independent_audit_every"):
        if c[key] is not None and (type(c[key]) is not int or c[key] < 1):
            raise ValueError(f"{key} must be a positive integer or null")
    if c["paper_variant"] not in ABLATIONS:
        raise ValueError("Unknown ablation variant")
    if c["reward"] not in ("R0", "R1", "R2", "R3") or (
        c["reward"] != "R0" and (not c["local_probe"])
    ):
        raise ValueError("Reward studies require the matched local-probe allocation")
    base = {**deepcopy(SELECTOR_DEFAULTS), **deepcopy(ABLATIONS[c["paper_variant"]])}
    strict_keys(c["selector_params"], base, "selector parameters")
    params = {**base, **c["selector_params"]}
    for key in ("phase_weights_critical", "phase_weights_transition", "phase_weights_exploit"):
        if params[key] is not None:
            params[key] = tuple(params[key])
    expected = deepcopy(base)
    expected["individualized_credits"] = c["local_probe"]
    expected["use_histogram"] = params["use_histogram"]
    if type(params["use_histogram"]) is not bool:
        raise ValueError("use_histogram must be boolean")
    if "individualized_credits" not in c["selector_params"]:
        params["individualized_credits"] = c["local_probe"]
    if params != expected:
        raise ValueError("Use the named ablation and reward-study configurations")
    c["selector_params"] = params
    if c["feature_shift"] and c["dataset"] != "CIFAR10":
        raise ValueError("The grayscale feature-shift control requires CIFAR10")
    if c["training_engine"] not in ("reference", "cuda_bf16", "cuda_bf16_compile"):
        raise ValueError("Unknown training engine")
    if c["training_engine"] != "reference" and (
        not c["dataset"].startswith("CIFAR") or c["normalization"] != "batchnorm"
    ):
        raise ValueError("The BF16 engine requires CIFAR ResNet18 with BatchNorm")
    return c


def identity_config(config):
    c = deepcopy(config)
    if c.get("independent_audit_every") is None:
        c.pop("independent_audit_every", None)
    if c["paper_variant"] == "V1" and c["total_clients"] / c["clients_per_round"] == 5:
        c["paper_variant"] = "A0"
        c["selector_params"]["use_adaptive_recency"] = True
    return c


def read_campaign(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    strict_keys(
        data,
        ("protocol", "name", "defaults", "cells", "methods", "seeds", "variants", "discovery"),
        "campaign",
    )
    if data.get("protocol") != PROTOCOL:
        raise ValueError("Wrong Apex protocol")
    strict_keys(data.get("defaults", {}), DEFAULTS, "campaign defaults")
    if not data.get("cells"):
        raise ValueError("Campaign needs explicit cells")
    jobs, identities = ([], {})
    for cell in data["cells"]:
        strict_keys(cell, ("id", "config", "seeds", "methods", "variants"), "cell")
        seeds = cell.get("seeds", data.get("seeds", S5))
        if isinstance(seeds, str):
            seeds = {"DEV": DEV, "S5": S5, "S10": S10}.get(seeds, [])
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Seeds must be explicit and unique")
        if not data.get("discovery", False) and any((seed in DEV for seed in seeds)):
            raise ValueError("Development seeds require discovery: true")
        if cell.get("methods", data.get("methods", ["apex"])) != ["apex"]:
            raise ValueError("This repository runs Apex only")
        for variant in cell.get("variants", data.get("variants", ["A0"])):
            if variant not in ABLATIONS:
                raise ValueError("Unknown ablation")
            for seed in seeds:
                values = {
                    **data.get("defaults", {}),
                    **cell.get("config", {}),
                    "method": "apex",
                    "seed": seed,
                    "paper_variant": variant,
                }
                values["selector_params"] = {
                    **values.get("selector_params", {}),
                    **ABLATIONS[variant],
                }
                c = resolve(values)
                label = dict(cell=cell["id"], variant=variant, method="apex", seed=seed)
                key = digest(identity_config(c))
                if key in identities:
                    identities[key]["aliases"].append(label)
                else:
                    job = dict(scientific_hash=key, config=c, aliases=[label])
                    identities[key] = job
                    jobs.append(job)
    return dict(
        protocol=PROTOCOL,
        name=data.get("name", Path(path).stem),
        discovery=data.get("discovery", False),
        jobs=jobs,
    )
