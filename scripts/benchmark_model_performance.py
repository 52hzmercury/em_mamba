import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)

import torch
from torch.utils.data import DataLoader

try:
    import torchvision

    torchvision.disable_beta_transforms_warning()
except Exception:
    warnings.filterwarnings(
        "ignore",
        message="The torchvision.datapoints and torchvision.transforms.v2 namespaces are still Beta.*",
        category=UserWarning,
    )

from simlvseg.model import get_model
from simlvseg.seg_3d.camus import CAMUSDatasetTest
from simlvseg.seg_3d.dataset import Seg3DDataset
from simlvseg.seg_3d.preprocessing import get_preprocessing_for_training
from simlvseg.utils import set_seed


# Fill these placeholders before running.
DATASETS = {
    "echonet": {
        "kind": "seg3d",
        "data_path": "/workdir1/echo_dataset/EchoNet-Dynamic",
    },
    "pediatric": {
        "kind": "seg3d",
        "data_path": "/workdir1/cn24/data/pediatric_echo/A4C",
    },
    "camus": {
        "kind": "camus",
        "data_path": "/workdir1/cn24/data/CAMUS",
        "patient_list": "camus/database_split/camus_test_filenames.txt",
    },
}


MODELS = [
    {"name": "SwinUnet", "encoder": "swinunet"},
    {"name": "3d u-net", "encoder": "3d_unet"},
    {"name": "UKAN3D", "encoder": "ukan3d"},
    {"name": "SegFormer3D", "encoder": "segformer3d"},
    {"name": "H2Former", "encoder": "h2former"},
    {"name": "SegMamba", "encoder": "segmamba"},
    {"name": "LKM-Unet", "encoder": "lkmunet"},
    {"name": "Log-vmamba", "encoder": "logvmamba"},
    {"name": "ours", "encoder": "mooremamba"},
    {"name": "nnUNet", "encoder": "nnunet"},
    {"name": "EfficientMedNext", "encoder": "efficientmednext"},
]


# # Each model can use a different checkpoint on each dataset.
# # Keep None if you only want architecture-level Params/GFLOPs/FPS/Mem.
# CHECKPOINTS: Dict[str, Dict[str, Optional[str]]] = {
#     model["encoder"]: {
#         "echonet": "TODO:/path/to/checkpoint.ckpt",
#         "pediatric": "TODO:/path/to/checkpoint.ckpt",
#         "camus": "TODO:/path/to/checkpoint.ckpt",
#     }
#     for model in MODELS
# }

CHECKPOINTS = {
    "swinunet": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_471/checkpoints/epoch=0-step=310.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_471/checkpoints/epoch=0-step=310.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_471/checkpoints/epoch=0-step=310.ckpt",
    },
    "3d_unet": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_304/checkpoints/epoch=26-step=25164.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_41/checkpoints/epoch=18-step=5735.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_321/checkpoints/epoch=46-step=470.ckpt",
    },
    "UKAN3D": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_341/checkpoints/epoch=45-step=34316.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_48/checkpoints/epoch=45-step=56465.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_347/checkpoints/epoch=58-step=590.ckpt",
    },
    "SegFormer3D": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_305/checkpoints/epoch=59-step=221935.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_312/checkpoints/epoch=50-step=4436.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_50/checkpoints/epoch=40-step=50880.ckpt",
    },
    "H2Former": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_476/checkpoints/epoch=5-step=930.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_476/checkpoints/epoch=5-step=930.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_476/checkpoints/epoch=5-step=930.ckpt",
    },
    "SegMamba": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_446/checkpoints/epoch=28-step=18009.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_438/checkpoints/epoch=20-step=13020.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_433/checkpoints/epoch=46-step=4089.ckpt",
    },
    "LKM-Unet": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_442/checkpoints/epoch=45-step=21436.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_444/checkpoints/epoch=46-step=29140.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_445/checkpoints/epoch=55-step=2408.ckpt",
    },
    "Log-vmamba": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_449/checkpoints/epoch=41-step=19572.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_450/checkpoints/epoch=52-step=32860.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_452/checkpoints/epoch=47-step=2064.ckpt",
    },
    "efficientmednext": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_478/checkpoints/epoch=45-step=21436.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_479/checkpoints/epoch=45-step=7130.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_477/checkpoints/epoch=17-step=378.ckpt",
    },
    "nnunet": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_470/checkpoints/epoch=29-step=6990.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_466/checkpoints/epoch=46-step=14570.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_469/checkpoints/epoch=36-step=370.ckpt",
    },
    "mooremamba": {
        "echonet": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_253/checkpoints/epoch=42-step=157592.ckpt",
        "pediatric": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_368/checkpoints/epoch=22-step=14260.ckpt",
        "camus": r"/workdir1/cn24/program/SimLVSeg/lightning_logs/version_268/checkpoints/epoch=36-step=3174.ckpt",
    },
}



@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    encoder: str
    checkpoint: str
    input_shape: str
    params_m: float
    gflops: Optional[float]
    fps: float
    mem_mb: Optional[float]
    status: str


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark 3D segmentation model Params/GFLOPs/FPS/Mem on all datasets.")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pinned memory.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--precision",
        choices=("amp", "fp32"),
        default="amp",
        help="Inference precision. Use fp32 for full precision; amp reduces CUDA memory on GPUs like RTX 3090.",
    )
    parser.add_argument(
        "--flops_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used for GFLOPs profiling. auto tries CPU first, then CUDA for CUDA-only ops.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="benchmark_model_performance.csv")
    parser.add_argument("--output_md", type=str, default="benchmark_model_performance.md")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_dataset", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_encoder", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip_missing_checkpoints", action="store_true")
    parser.add_argument("--models", nargs="*", default=None, help="Optional encoder names to benchmark.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to benchmark.")
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.12741163, 0.1279413, 0.12912785),
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(0.19557191, 0.19562256, 0.1965878),
    )
    return parser.parse_args()


def read_lines(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def is_placeholder(path: Optional[str]) -> bool:
    return path is None or path == "" or path.startswith("TODO:")


def build_dataloader(dataset_name: str, cfg: dict, args) -> DataLoader:
    if is_placeholder(cfg["data_path"]):
        raise ValueError(f"{dataset_name} data_path is still a placeholder: {cfg['data_path']}")

    if cfg["kind"] == "seg3d":
        preprocessing = get_preprocessing_for_training(args.frames, args.mean, args.std)
        dataset = Seg3DDataset(
            cfg["data_path"],
            "test",
            args.frames,
            args.period,
            False,
            preprocessing,
            None,
            test=True,
        )
    elif cfg["kind"] == "camus":
        patient_list = cfg.get("patient_list")
        if is_placeholder(patient_list):
            raise ValueError(f"{dataset_name} patient_list is still a placeholder: {patient_list}")
        dataset = CAMUSDatasetTest(
            cfg["data_path"],
            args.frames,
            args.mean,
            args.std,
            read_lines(patient_list),
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {cfg['kind']}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.pin_memory,
    )


def get_first_input(dataloader: DataLoader, device: Optional[torch.device] = None) -> torch.Tensor:
    batch = next(iter(dataloader))
    x = batch[0]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    if device is not None:
        x = x.to(device, non_blocking=True)
    return x


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: Optional[str]):
    if is_placeholder(checkpoint_path):
        return "no checkpoint"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing or unexpected:
        return f"checkpoint loaded, missing={len(missing)}, unexpected={len(unexpected)}"
    return "checkpoint loaded"


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def unwrap_output(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _autocast_context(device: torch.device, precision: str):
    return torch.cuda.amp.autocast(enabled=device.type == "cuda" and precision == "amp")


def _profile_with_thop(model: torch.nn.Module, x: torch.Tensor, device: torch.device, precision: str) -> float:
    from thop import profile

    model.eval()
    with torch.inference_mode():
        with _autocast_context(device, precision):
            flops, _ = profile(model, inputs=(x,), verbose=False)
    return float(flops) / 1e9


def _profile_with_fvcore(model: torch.nn.Module, x: torch.Tensor, device: torch.device, precision: str) -> float:
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    with torch.inference_mode():
        with _autocast_context(device, precision):
            flops = FlopCountAnalysis(model, x).total()
    return float(flops) / 1e9


def compute_gflops(
    encoder: str,
    img_size: Tuple[int, int, int],
    x_cpu: torch.Tensor,
    device: torch.device,
    precision: str,
    flops_device: str,
) -> Optional[float]:
    if flops_device == "cpu" or device.type != "cuda":
        profile_devices = [torch.device("cpu")]
    elif flops_device == "cuda":
        profile_devices = [device]
    else:
        profile_devices = [torch.device("cpu"), device]

    errors = []
    for profile_device in profile_devices:
        x = None
        for profiler_name, profiler_fn in (
            ("thop", _profile_with_thop),
            ("fvcore", _profile_with_fvcore),
        ):
            model = None
            try:
                model = get_model(encoder, img_size=img_size).to(profile_device)
                x = x_cpu.to(profile_device, non_blocking=True)
                gflops = profiler_fn(model, x, profile_device, precision)
                return gflops
            except Exception as exc:
                errors.append(f"{profiler_name}/{profile_device}: {exc}")
            finally:
                del model
                if profile_device.type == "cuda":
                    cleanup_cuda(profile_device)
        del x

    print("GFLOPs unavailable: " + "; ".join(errors))
    return None


def format_result(result: BenchmarkResult) -> str:
    gflops = "NA" if result.gflops is None else f"{result.gflops:.2f}"
    mem = "NA" if result.mem_mb is None else f"{result.mem_mb:.2f}"
    return (
        f"Result {result.dataset}/{result.model}: "
        f"Params={result.params_m:.2f}M, GFLOPs={gflops}G, "
        f"FPS={result.fps:.2f}, Mem={mem}MB, Status={result.status}"
    )


def safe_empty_cache(device: torch.device, quiet: bool = False):
    if device.type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if not quiet:
            print(f"Warning: torch.cuda.empty_cache() failed: {exc}")


def benchmark_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
    precision: str,
) -> Tuple[float, Optional[float]]:
    model.eval()
    is_cuda = device.type == "cuda"
    use_amp = is_cuda and precision == "amp"

    if is_cuda:
        safe_empty_cache(device)
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _ = unwrap_output(model(x))
            if is_cuda:
                torch.cuda.synchronize(device)

            start = time.perf_counter()
            for _ in range(iters):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _ = unwrap_output(model(x))
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
    except RuntimeError as exc:
        if is_cuda and "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"CUDA out of memory during {precision} inference. "
                "Try --frames 16, --iters 3, or --precision amp on a less busy GPU."
            ) from exc
        raise

    fps = (iters * x.shape[0]) / elapsed
    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if is_cuda else None
    return fps, mem_mb


def cleanup_cuda(device: torch.device):
    gc.collect()
    safe_empty_cache(device, quiet=True)


def select_items(items: Iterable[dict], selected_names: Optional[List[str]], key: str) -> List[dict]:
    if selected_names is None:
        return list(items)
    selected = {name.lower() for name in selected_names}
    return [item for item in items if item[key].lower() in selected]


def get_model_cfg(encoder: str) -> dict:
    for model_cfg in MODELS:
        if model_cfg["encoder"].lower() == encoder.lower():
            return model_cfg
    raise ValueError(f"Unknown encoder: {encoder}")


def get_checkpoint_path(encoder: str, dataset_name: str) -> Optional[str]:
    checkpoint_by_dataset = CHECKPOINTS.get(encoder)
    if checkpoint_by_dataset is None:
        model_name = get_model_cfg(encoder)["name"]
        checkpoint_by_dataset = CHECKPOINTS.get(model_name)
    if checkpoint_by_dataset is None:
        checkpoint_by_dataset = CHECKPOINTS.get(encoder.lower())
    if checkpoint_by_dataset is None:
        return None
    return checkpoint_by_dataset.get(dataset_name)


def result_from_dict(data: dict) -> BenchmarkResult:
    return BenchmarkResult(
        dataset=data["dataset"],
        model=data["model"],
        encoder=data["encoder"],
        checkpoint=data["checkpoint"],
        input_shape=data["input_shape"],
        params_m=float(data["params_m"]),
        gflops=None if data["gflops"] is None else float(data["gflops"]),
        fps=float(data["fps"]),
        mem_mb=None if data["mem_mb"] is None else float(data["mem_mb"]),
        status=data["status"],
    )


def write_worker_result(result: BenchmarkResult, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)


def write_csv(results: List[BenchmarkResult], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def write_markdown(results: List[BenchmarkResult], path: str):
    headers = ["Dataset", "Model", "Param(M)", "GFLOPs(G)", "FPS", "Mem(MB)", "Status"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in results:
            gflops = "" if r.gflops is None else f"{r.gflops:.2f}"
            mem = "" if r.mem_mb is None else f"{r.mem_mb:.2f}"
            f.write(
                f"| {r.dataset} | {r.model} | {r.params_m:.2f} | {gflops} | "
                f"{r.fps:.2f} | {mem} | {r.status} |\n"
            )


def run_single_benchmark(args, dataset_name: str, model_cfg: dict) -> BenchmarkResult:
    cfg = DATASETS[dataset_name]
    encoder = model_cfg["encoder"]
    model_name = model_cfg["name"]
    checkpoint_path = get_checkpoint_path(encoder, dataset_name)

    if args.skip_missing_checkpoints and (
        is_placeholder(checkpoint_path) or not os.path.exists(checkpoint_path)
    ):
        return BenchmarkResult(
            dataset=dataset_name,
            model=model_name,
            encoder=encoder,
            checkpoint="" if checkpoint_path is None else checkpoint_path,
            input_shape="",
            params_m=0.0,
            gflops=None,
            fps=0.0,
            mem_mb=None,
            status="skipped: missing checkpoint",
        )

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dataloader = None
    model = None
    x = None
    x_cpu = None
    params_m = 0.0
    gflops = None
    fps = 0.0
    mem_mb = None
    status = "ok"
    input_shape = ""

    try:
        dataloader = build_dataloader(dataset_name, cfg, args)
        x_cpu = get_first_input(dataloader)
        input_shape = str(tuple(x_cpu.shape))

        img_size = (x_cpu.shape[2], x_cpu.shape[3], x_cpu.shape[4])

        model = get_model(encoder, img_size=img_size)
        params_m = count_params_m(model)
        del model
        model = None

        gflops = compute_gflops(
            encoder,
            img_size,
            x_cpu,
            device,
            args.precision,
            args.flops_device,
        )

        # Recreate the model after profiling. THOP/FVCore attach hooks and
        # buffers, and failed profiling can leave modules in a bad state.
        model = get_model(encoder, img_size=img_size)
        checkpoint_status = load_checkpoint_if_available(model, checkpoint_path)
        model = model.to(device)
        x = x_cpu.to(device, non_blocking=True)
        fps, mem_mb = benchmark_inference(model, x, args.warmup, args.iters, device, args.precision)
        status = checkpoint_status
    except Exception as exc:
        status = f"failed: {exc}"
        print(f"  failed: {exc}")
    finally:
        del x
        del x_cpu
        del model
        del dataloader
        cleanup_cuda(device)

    return BenchmarkResult(
        dataset=dataset_name,
        model=model_name,
        encoder=encoder,
        checkpoint="" if checkpoint_path is None else checkpoint_path,
        input_shape=input_shape,
        params_m=params_m,
        gflops=gflops,
        fps=fps,
        mem_mb=mem_mb,
        status=status,
    )


def worker_main(args):
    if args.worker_dataset is None or args.worker_encoder is None or args.worker_output is None:
        raise ValueError("--worker requires --worker_dataset, --worker_encoder, and --worker_output")

    set_seed(args.seed)
    model_cfg = get_model_cfg(args.worker_encoder)
    print(f"Preparing dataset: {args.worker_dataset}")
    print(f"Benchmarking {args.worker_dataset}/{model_cfg['name']} ({model_cfg['encoder']})")
    result = run_single_benchmark(args, args.worker_dataset, model_cfg)
    if result.input_shape:
        print(f"  input shape: {result.input_shape}")
    print(format_result(result))
    write_worker_result(result, args.worker_output)


def build_worker_command(args, dataset_name: str, encoder: str, output_path: str) -> List[str]:
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--worker_dataset",
        dataset_name,
        "--worker_encoder",
        encoder,
        "--worker_output",
        output_path,
        "--device",
        args.device,
        "--frames",
        str(args.frames),
        "--period",
        str(args.period),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--precision",
        args.precision,
        "--flops_device",
        args.flops_device,
        "--seed",
        str(args.seed),
        "--mean",
        *[str(value) for value in args.mean],
        "--std",
        *[str(value) for value in args.std],
    ]
    if args.pin_memory:
        command.append("--pin_memory")
    if args.skip_missing_checkpoints:
        command.append("--skip_missing_checkpoints")
    return command


def run_worker(args, dataset_name: str, model_cfg: dict, output_path: str) -> BenchmarkResult:
    command = build_worker_command(args, dataset_name, model_cfg["encoder"], output_path)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return result_from_dict(json.load(f))

    status = f"failed: worker exited with code {completed.returncode}"
    if completed.stderr.strip():
        status = f"{status}: {completed.stderr.strip().splitlines()[-1]}"
    return BenchmarkResult(
        dataset=dataset_name,
        model=model_cfg["name"],
        encoder=model_cfg["encoder"],
        checkpoint="" if get_checkpoint_path(model_cfg["encoder"], dataset_name) is None else get_checkpoint_path(model_cfg["encoder"], dataset_name),
        input_shape="",
        params_m=0.0,
        gflops=None,
        fps=0.0,
        mem_mb=None,
        status=status,
    )


def main():
    args = parse_args()

    if args.worker:
        worker_main(args)
        return

    set_seed(args.seed)

    selected_models = select_items(MODELS, args.models, "encoder")
    selected_datasets = {
        name: cfg
        for name, cfg in DATASETS.items()
        if args.datasets is None or name.lower() in {d.lower() for d in args.datasets}
    }

    results: List[BenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="benchmark_model_performance_") as tmpdir:
        for dataset_name in selected_datasets:
            for model_cfg in selected_models:
                output_path = os.path.join(tmpdir, f"{dataset_name}_{model_cfg['encoder']}.json")
                results.append(run_worker(args, dataset_name, model_cfg, output_path))

    write_csv(results, args.output_csv)
    write_markdown(results, args.output_md)
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
