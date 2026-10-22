from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from . import protocol as p


class CudaTraining:

    def __init__(self, config, train, evaluation, test, device):
        if not str(device).startswith("cuda") or not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise ValueError("The CUDA BF16 profile requires native BF16 GPU support")
        self.config, self.device = (config, device)
        self.train, self.evaluation, self.test = (train, evaluation, test)
        self.compiled = config["training_engine"] == "cuda_bf16_compile"
        self.local = self.forward = None
        self.cache = {}
        for dataset in (train, test):
            raw = (
                torch.as_tensor(np.asarray(dataset.data))
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(device)
            )
            labels = torch.tensor(dataset.targets, dtype=torch.long, device=device)
            self.cache[id(dataset)] = (raw, labels)
        self.cache[id(evaluation)] = self.cache[id(train)]
        from .datasets import _CIFAR10_MEAN, _CIFAR10_STD, _CIFAR100_MEAN, _CIFAR100_STD

        mean, std = (
            (_CIFAR10_MEAN, _CIFAR10_STD)
            if config["dataset"] == "CIFAR10"
            else (_CIFAR100_MEAN, _CIFAR100_STD)
        )
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def batches(self, dataset, ids, batch_size, seed=0, shuffle=False):
        gray_view = type(dataset).__name__ == "GrayView"
        mixed_view = type(dataset).__name__ == "MixedView"
        original = dataset.dataset if gray_view or mixed_view else dataset
        if id(original) not in self.cache:
            raise ValueError("Unsupported dataset view in the CUDA profile")
        raw, labels = self.cache[id(original)]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        order = torch.tensor(list(ids), device=self.device)
        if shuffle:
            order = order[torch.randperm(len(order), generator=generator, device=self.device)]
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            x = raw[selected].float() / 255
            if shuffle:
                n = len(selected)
                x = F.pad(x, (4, 4, 4, 4))
                top = torch.randint(0, 9, (n, 1, 1), generator=generator, device=self.device)
                left = torch.randint(0, 9, (n, 1, 1), generator=generator, device=self.device)
                rows = top + torch.arange(32, device=self.device).view(1, 32, 1)
                cols = left + torch.arange(32, device=self.device).view(1, 1, 32)
                x = x.permute(0, 2, 3, 1)[
                    torch.arange(n, device=self.device).view(n, 1, 1), rows, cols
                ].permute(0, 3, 1, 2)
                flip = torch.rand((n, 1, 1, 1), generator=generator, device=self.device) < 0.5
                x = torch.where(flip, x.flip(-1), x)
            if gray_view or mixed_view:
                gray = (
                    (0.2989 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand_as(x)
                )
                if gray_view:
                    x = gray
                else:
                    mask = torch.tensor(
                        [int(i) in dataset.indices for i in selected.cpu().tolist()],
                        device=self.device,
                    ).view(-1, 1, 1, 1)
                    x = torch.where(mask, gray, x)
            y = labels[selected]
            yield (
                ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last),
                y if shuffle else y.cpu(),
            )

    def evaluate(self, model, dataset, ids):
        batches = self.batches(dataset, ids, self.config["evaluation_batch_size"])
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=dataset is not self.test and getattr(dataset, "dataset", None) is not self.test,
        ):
            return p.evaluate(model, batches, self.config["num_classes"], self.device)

    def train_client(self, model, dataset, ids, config, round_idx, cid, device, stream="train"):
        if self.local is None:
            self.local = deepcopy(model).to(device, memory_format=torch.channels_last).train()
            if self.compiled:
                self.forward = torch.compile(self.local, mode="reduce-overhead", dynamic=False)
        local = self.local
        local.load_state_dict(model.state_dict())
        local.train()
        opt = torch.optim.SGD(
            local.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            foreach=True,
        )
        stats = dict(loss=0.0, norm_sq=0.0, steps=0, processed=0, bn_frozen_batches=0, loss_sq=0.0)
        for epoch in range(config["local_epochs"]):
            seed = p.seed_for(config["seed"], stream, round_idx, cid, epoch)
            with p.isolated_rng(seed):
                for x, y in self.batches(dataset, ids, config["batch_size"], seed, True):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if self.compiled and len(y) == config["batch_size"]:
                            torch.compiler.cudagraph_mark_step_begin()
                            logits = self.forward(x)
                        else:
                            with p.singleton_batchnorm(local, stats):
                                logits = local(x)
                        losses = F.cross_entropy(logits, y, reduction="none")
                        loss = losses.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite local loss for client {cid}")
                    loss.backward()
                    norm_sq = float(
                        torch.stack(
                            torch._foreach_norm(
                                [
                                    param.grad
                                    for param in local.parameters()
                                    if param.grad is not None
                                ]
                            )
                        )
                        .double()
                        .square()
                        .sum()
                    )
                    if not np.isfinite(norm_sq):
                        raise FloatingPointError(f"Nonfinite gradient for client {cid}")
                    opt.step()
                    stats["loss"] += float(loss.detach()) * len(y)
                    stats["loss_sq"] += float(losses.detach().square().sum())
                    stats["norm_sq"] += norm_sq
                    stats["steps"] += 1
                    stats["processed"] += len(y)
        stats["loss"] /= stats["processed"]
        stats["grad_norm"] = (stats.pop("norm_sq") / stats["steps"]) ** 0.5
        stats["oort_utility"] = len(ids) * (stats.pop("loss_sq") / stats["processed"]) ** 0.5
        return (
            {key: value.detach().cpu().clone() for key, value in local.state_dict().items()},
            stats,
        )
