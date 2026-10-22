import argparse
import json
from pathlib import Path
from .config import read_campaign
from .protocol import atomic_json, digest, provenance
from .runner import prepare_campaign, load_data, load_tensor_file, ApexRunner, verify_completion


def run_job(
    lock_path,
    run_id,
    output=None,
    device="cpu",
    data_root=None,
    resume=False,
    work_seconds=None,
    export_reserve=300,
    allow_hardware_change=False,
    round_seconds_bound=60,
):
    path = Path(lock_path).resolve()
    locked = json.loads(path.read_text())
    if locked.get("status") != "prepared":
        raise ValueError("Run needs a prepared frozen manifest")
    current = provenance()
    if current["source_hash"] != locked["code"]["source_hash"]:
        raise ValueError("Source changed since freeze; prepare a new manifest")
    matches = [j for j in locked["jobs"] if j["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Exact run ID required")
    job = matches[0]
    root = Path(output).resolve() if output else path.parent
    run_dir = root / job["run_directory"]
    if (run_dir / "completion.json").exists():
        return verify_completion(
            run_dir,
            job["config"],
            source_hash=job["source_hash"],
            snapshot_hash=job["snapshot_hash"],
        )
    snapshot = load_tensor_file(path.parent / job["snapshot"])
    if snapshot["hash"] != job["snapshot_hash"]:
        raise ValueError("Snapshot hash differs from manifest")
    datasets = load_data(job["config"]["dataset"], data_root)
    from .runner import data_fingerprint

    for name, dataset in (("train", datasets[0]), ("test", datasets[2])):
        if data_fingerprint(dataset) != snapshot["data_hashes"][name]:
            raise ValueError(f"Dataset {name} changed since preparation")
    ledger_path = root / "job_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger[run_id] = dict(
        status="running", scientific_hash=job["scientific_hash"], source_hash=current["source_hash"]
    )
    atomic_json(ledger_path, ledger)
    try:
        runner = ApexRunner(
            job["config"],
            snapshot,
            *datasets,
            run_dir,
            device,
            current,
            allow_hardware_change=allow_hardware_change,
        )
        result = runner.run(
            resume,
            work_seconds=work_seconds,
            export_reserve=export_reserve,
            round_seconds_bound=round_seconds_bound,
        )
        ledger[run_id].update(status=result["status"], round=result["round"])
        atomic_json(ledger_path, ledger)
        return result
    except BaseException as error:
        ledger[run_id].update(status="failed", error=str(error))
        atomic_json(ledger_path, ledger)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Apex federated client selection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "dry-run", "prepare"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        if name == "prepare":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--data-root", type=Path)
            p.add_argument("--download", action="store_true")
    for name in ("run", "resume", "run-all"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        if name != "run-all":
            p.add_argument("--run-id", required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), required=True)
        p.add_argument("--data-root", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--allow-hardware-change", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("validate", "dry-run", "prepare"):
        campaign = read_campaign(args.manifest)
        report = dict(
            jobs=len(campaign["jobs"]),
            aliases=sum((len(j["aliases"]) - 1 for j in campaign["jobs"])),
            round_executions=sum((j["config"]["rounds"] for j in campaign["jobs"])),
            executes_training=False,
            allocates_session=False,
        )
        if args.command == "prepare":
            prepare_campaign(campaign, args.output, args.data_root, args.download)
            report["locked_manifest"] = str(args.output / "manifest.lock.json")
        print(json.dumps(report, indent=2))
    elif args.command == "verify":
        path = args.manifest.resolve()
        locked = json.loads(path.read_text())
        rounds = 0
        for job in locked["jobs"]:
            result = verify_completion(
                path.parent / job["run_directory"],
                job["config"],
                source_hash=job["source_hash"],
                snapshot_hash=job["snapshot_hash"],
            )
            rounds += result["round"]
        print(
            json.dumps(dict(status="verified", jobs=len(locked["jobs"]), rounds=rounds), indent=2)
        )
    else:
        locked = json.loads(args.manifest.read_text())
        ids = [j["run_id"] for j in locked["jobs"]] if args.command == "run-all" else [args.run_id]
        for run_id in ids:
            directory = (
                args.output.resolve() if args.output else args.manifest.resolve().parent
            ) / next((j["run_directory"] for j in locked["jobs"] if j["run_id"] == run_id))
            resume = args.command == "resume" or (
                args.command == "run-all" and (directory / "latest_recovery.json").exists()
            )
            result = run_job(
                args.manifest,
                run_id,
                args.output,
                args.device,
                args.data_root,
                resume,
                allow_hardware_change=args.allow_hardware_change,
            )
            print(
                json.dumps(dict(run_id=run_id, status=result["status"], round=result["round"])),
                flush=True,
            )


if __name__ == "__main__":
    main()
