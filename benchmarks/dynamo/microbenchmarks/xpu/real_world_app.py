# shows real influence on production inference without using Eager Mode
# model -> torch.compile -> with or without CUDA graphs
# image classification / detection models, plus sd15: one Stable Diffusion 1.5 UNet denoise step
# (diffusion loops). tinyllama: one causal LM forward (BS=1), last-token logits — same shape of
# work as the inner body of token-by-token decode where manual graphs cut launch overhead.
# whisper_enc: Whisper encoder only on fixed-shape log-mel (STT-style; static graph friendly for
# fixed window length, cf. mel-spectrogram / transcription stacks).
#
# VTune (Intel VTune Profiler):
#   With --emit-itt, torch.autograd.profiler.emit_itt() marks each autograd op in the timed
#   section; record_function(...) always adds named regions (eager, graph_replay, ...).
#   Use --fine-grain-itt for per-iteration regions. Enable ITT in the VTune analysis properties.
#
# torch.profiler on XPU requires Kineto (USE_KINETO=ON when building PyTorch). USE_KINETO=OFF
# triggers: "Legacy XPU profiling is not supported. Requires use_kineto=True on XPU devices."

import argparse
import contextlib
import csv
import json
import logging
import os
import sys
import time

import timm
import torch
import torch.autograd.profiler as profiler
import torchvision.models as models


def _profiler_activities_for_device(device: str) -> list[torch.profiler.ProfilerActivity]:
    """Activities for torch.profiler.profile. XPU needs a Kineto-enabled build."""
    pa = torch.profiler.ProfilerActivity
    if device == "cuda":
        return [pa.CPU, pa.CUDA]
    if device == "xpu":
        if not torch.profiler.kineto_available:
            raise SystemExit(
                "torch.profiler on XPU requires Kineto inside this PyTorch build.\n"
                "You used USE_KINETO=OFF; rebuild with USE_KINETO=ON (or unset; default is ON).\n"
                "This is not fixable with pip packages — libtorch must include Kineto."
            )
        supported = set(torch.profiler.supported_activities())
        if pa.XPU not in supported:
            raise SystemExit(
                "ProfilerActivity.XPU is missing from torch.profiler.supported_activities(). "
                "Use an XPU build that compiles Kineto XPU profiling support."
            )
        return [pa.CPU, pa.XPU]
    raise SystemExit(f"--profiler: unsupported --device {device!r}")


def _configure_stdout_logger(name: str) -> logging.Logger:
    """Timestamped INFO logs to stdout; flush after each record."""
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False

    class _FlushStreamHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    handler = _FlushStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    return log


parser = argparse.ArgumentParser(description="Testing CUDA/SYCL Graphs.")
parser.add_argument(
    "--graphs",
    "--graph",
    action="store_true",
    help=("Use device capture/replay graphs (CUDA or XPU)."),
)
parser.add_argument(
    "--profiler",
    "--profile",
    action="store_true",
    dest="profiler",
    help=(
        "Enable torch.profiler in the timed loop (exports table + real_world_app_profiler.csv). "
        "On --device xpu, PyTorch must be built with USE_KINETO=ON."
    ),
)
parser.add_argument("--logs", action="store_true", help="Log if cuda graphs are used or not.")  # alternatively set TORCH_LOGS="inductor,cuda_graphs"
parser.add_argument("--autographs", "--autograph", action="store_true", help="Automatic use of graphs in torch.compile and with kernel fusions")
parser.add_argument("--compile", action="store_true", help="Call compile on model")
parser.add_argument("--iter", type=int, default=3000, help="Number of iterations")
parser.add_argument(
    "--batch",
    type=int,
    default=None,
    metavar="N",
    help=(
        "In the timed section: synchronize every N iterations (graph replay or eager forward). "
        "Each batch runs N steps then syncs; a final partial batch syncs the same way. "
        "Default: N = --iter (one batch, sync only at end)."
    ),
)
parser.add_argument("--device", type=str, choices=['xpu', 'cuda'], default='xpu', help="Which backend to use.")
parser.add_argument(
    "--model",
    type=str,
    choices=[
        "resnet",
        "transformer",
        "retina",
        "vit",
        "sd15",
        "tinyllama",
        "whisper_enc",
    ],
    default="resnet",
    help=(
        "Which model to use. sd15 = one SD 1.5 UNet step (needs diffusers). "
        "tinyllama = TinyLlama 1.1B one forward, last-token logits, BS=1 (needs transformers) — "
        "typical manual-graph target for decode-style loops (Phi/TinyLlama/Llama-small family). "
        "whisper_enc = Whisper encoder on fixed-shape log-mel (needs transformers) — audio/STT-style "
        "static compute for fixed window sizes."
    ),
)
parser.add_argument("--retina-size", type=int, default=224, help="retina input size")
parser.add_argument(
    "--llm-seq-len",
    type=int,
    default=128,
    metavar="T",
    help="For --model tinyllama: sequence length (1, T) int64 token ids). Default: 128.",
)
parser.add_argument(
    "--mel-frames",
    type=int,
    default=3000,
    metavar="F",
    help=(
        "For --model whisper_enc: requested mel time frames. OpenAI Whisper checkpoints require "
        "F = 2 * WhisperConfig.max_source_positions (3000 for openai/whisper-*). If your value "
        "does not match the loaded model, the script uses the config value and logs a message."
    ),
)
parser.add_argument(
    "--vit-size",
    type=int,
    default=16,
    help="ViT spatial size H=W (must be divisible by patch size 16 for vit_*_patch16_*).",
)
parser.add_argument("--skip-consistency-check", action="store_true", help="Skip graph vs eager consistency check when using --graphs.")
parser.add_argument(
    "--native",
    action="store_true",
    help=(
        "XPU only with --graphs: build the SYCL command_graph with "
        "property::graph::enable_native_recording (Level Zero native graph capture). "
        "Requires a SYCL/UR stack that supports it."
    ),
)
parser.add_argument(
    "--fine-grain-itt",
    action="store_true",
    help="Per-iteration record_function regions and labeled sync for VTune (see module docstring).",
)
parser.add_argument(
    "--emit-itt",
    action="store_true",
    help="Wrap timed loops with profiler.emit_itt() (dense per-op ITT for VTune). Default: off.",
)
parser.add_argument(
    "--output-json",
    type=str,
    default=None,
    metavar="PATH",
    help="If set, write latency (and basic run metadata) to this JSON file in addition to the log line.",
)
args = parser.parse_args()

batch_size = args.batch if args.batch is not None else args.iter
if batch_size < 1:
    raise SystemExit("--batch must be >= 1")
if args.model == "tinyllama" and args.llm_seq_len < 1:
    raise SystemExit("--llm-seq-len must be >= 1")
if args.model == "whisper_enc" and args.mel_frames < 1:
    raise SystemExit("--mel-frames must be >= 1")

logger = _configure_stdout_logger("real_world_app")

if args.native and args.device != "xpu":
    raise SystemExit("--native is only supported for --device xpu")
if args.native and not args.graphs:
    raise SystemExit("--native requires --graphs")

if args.graphs and args.device == "cuda" and args.model == "retina":
    logger.warning(
        "RetinaNet eval postprocess_detections (NMS, masking, etc.) uses ops CUDA graph capture does not allow"
    )

_profiler_activities: list[torch.profiler.ProfilerActivity] | None = None
if args.profiler:
    _profiler_activities = _profiler_activities_for_device(args.device)

def _itt_ctx():
    return profiler.emit_itt() if args.emit_itt else contextlib.nullcontext()


def _compare_outputs(ref, graph_out, *, rtol=1e-1, atol=1e-1, name="output"):
    """Compare reference (eager) and graph outputs. Raises AssertionError on mismatch."""
    if isinstance(ref, torch.Tensor):
        ok = (
            torch.allclose(ref, graph_out, rtol=rtol, atol=atol)
            if ref.is_floating_point()
            else torch.equal(ref, graph_out)
        )
        if not ok:
            diff = (ref - graph_out).abs()
            max_abs_diff = diff.max().item()
            denom = ref.abs().clamp(min=1e-8)
            max_rel_diff = (diff / denom).max().item()
            raise AssertionError(
                f"{name}: graph output does not match eager (rtol={rtol}, atol={atol}). "
                f"max_abs_diff={max_abs_diff:.3e}, max_rel_diff={max_rel_diff:.3e}"
            )
        return
    if isinstance(ref, (list, tuple)):
        assert len(ref) == len(graph_out), f"{name}: length mismatch {len(ref)} vs {len(graph_out)}"
        for i, (r, g) in enumerate(zip(ref, graph_out)):
            if isinstance(r, dict):
                assert set(r.keys()) == set(g.keys()), f"{name}[{i}]: dict keys mismatch"
                for k in r:
                    if isinstance(r[k], torch.Tensor):
                        if r[k].is_floating_point():
                            ok = torch.allclose(r[k], g[k], rtol=rtol, atol=atol)
                            if not ok:
                                d = (r[k] - g[k]).abs()
                                max_abs = d.max().item()
                                denom = r[k].abs().clamp(min=1e-8)
                                max_rel = (d / denom).max().item()
                                raise AssertionError(
                                    f"{name}[{i}].{k}: graph does not match eager, "
                                    f"max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
                                )
                        else:
                            assert torch.equal(r[k], g[k]), f"{name}[{i}].{k}: graph does not match eager"
                    else:
                        assert r[k] == g[k], f"{name}[{i}].{k}: graph does not match eager"
            else:
                _compare_outputs(r, g, rtol=rtol, atol=atol, name=f"{name}[{i}]")
        return
    raise TypeError(f"{name}: unsupported output type {type(ref)}")


def _materialize_detection_transform_stats_on_device(model: torch.nn.Module, device: str) -> None:
    """GeneralizedRCNNTransform keeps image_mean / image_std as Python lists. During forward,
    torchvision builds mean/std via torch.as_tensor(..., device=image.device), which can
    trigger a CPU→device copy that CUDA graph capture rejects unless the source is pinned.
    Storing mean/std as tensors already on the target device avoids that copy inside capture."""
    root = getattr(model, "_orig_mod", model)
    t = getattr(root, "transform", None)
    if t is None or not hasattr(t, "image_mean") or not hasattr(t, "image_std"):
        return
    dev = torch.device(device)
    dtype = next(root.parameters()).dtype

    def _tensor_on_dev(x):
        if isinstance(x, torch.Tensor):
            if x.device == dev and x.dtype == dtype:
                return x
            return x.to(device=dev, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=dev)

    t.image_mean = _tensor_on_dev(t.image_mean)
    t.image_std = _tensor_on_dev(t.image_std)


class _StableDiffusion15UNetDenoiseStep(torch.nn.Module):
    """Single UNet forward as in one diffusion timestep (SD 1.5, 512 px -> 64x64 latent)."""

    def __init__(self, unet: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.unet = unet
        self.register_buffer(
            "timestep",
            torch.tensor([500], device=device, dtype=torch.long),
        )
        self.register_buffer(
            "encoder_hidden_states",
            torch.randn(1, 77, 768, device=device, dtype=dtype),
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        out = self.unet(
            sample,
            self.timestep,
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False,
        )
        return out[0]


class _WhisperEncoderMelForward(torch.nn.Module):
    """Whisper encoder on fixed-shape log-mel `input_features` (B, n_mels, n_frames).

    Matches STT stacks that run a static acoustic front-end into the same encoder shape every
    frame/window (real-time transcription style).
    """

    def __init__(self, whisper_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = whisper_encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_features, return_dict=True)
        return out.last_hidden_state


class _CausalLMOneDecodeForward(torch.nn.Module):
    """Single `forward` of a causal LM on fixed-shape `input_ids` (BS=1).

    Returns last-position logits (fp32) — the tensor shape producers typically use for argmax
    sampling in autoregressive decode; same static subgraph many engines capture per step.
    """

    def __init__(self, causal_lm: torch.nn.Module) -> None:
        super().__init__()
        self.lm = causal_lm

    @property
    def config(self):
        return self.lm.config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(
            input_ids,
            use_cache=False,
            attention_mask=None,
            return_dict=True,
        )
        return out.logits[:, -1, :].float()


def _load_tinyllama_decode_benchmark(device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise SystemExit(
            "--model tinyllama requires the `transformers` package. Install with: pip install transformers"
        ) from e

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dev = torch.device(device)
    logger.info("load %s (one decode-style forward, BS=1)", model_id)
    lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    lm = lm.to(dev).eval()
    return _CausalLMOneDecodeForward(lm)


def _load_whisper_encoder_benchmark(device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        from transformers import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "--model whisper_enc requires the `transformers` package. Install with: pip install transformers"
        ) from e

    model_id = "openai/whisper-tiny"
    dev = torch.device(device)
    logger.info(
        "load %s (encoder only, fixed-shape log-mel input_features)", model_id
    )
    wm = WhisperModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    wm = wm.to(dev).eval()
    return _WhisperEncoderMelForward(wm.encoder)


def _load_sd15_unet_benchmark(device: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        from diffusers import UNet2DConditionModel
    except ImportError as e:
        raise SystemExit(
            "--model sd15 requires the `diffusers` package. Install with: pip install diffusers"
        ) from e

    dev = torch.device(device)
    logger.info("load SD 1.5 UNet (runwayml/stable-diffusion-v1-5, subfolder unet)...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        torch_dtype=dtype,
    )
    unet = unet.to(dev).eval()
    return _StableDiffusion15UNetDenoiseStep(unet, dev, dtype)


torch.set_float32_matmul_precision('high')

class Backend:
    def __init__(self):
        if args.device == 'cuda':
            self.synchronize = torch.cuda.synchronize
            self.create_graph = torch.cuda.CUDAGraph
            self.graph = torch.cuda.graph
            self.empty_cache = torch.cuda.empty_cache
        elif args.device == 'xpu':
            assert torch.xpu.is_available()
            self.synchronize = torch.xpu.synchronize
            if args.native:
                self.create_graph = lambda: torch.xpu.XPUGraph(
                    native_recording=True
                )
            else:
                self.create_graph = torch.xpu.XPUGraph
            self.graph = torch.xpu.graph
            self.empty_cache = torch.xpu.empty_cache
        else:
            raise RuntimeError(f"unknown backend {args.device}")


backend = Backend()

if args.logs:
    import torch._logging

    # TORCHINDUCTOR_CUDA_GRAPHS=1 <- force graphs ??
    # https://docs.pytorch.org/docs/stable/generated/torch._logging.set_logs.html
    torch._logging.set_logs(graph_breaks=True)

whisper_n_mels = None
whisper_mel_frames = None
if args.model == 'resnet':
    # CNN - convolutional net
    logger.info("resnet50 model to device")
    model = models.resnet50().to(args.device).eval()
elif args.model == 'transformer':
    # vit: Vision Transformer
    # base: model size
    # patch16; spliting image into 16x16 patches (14x14 patches grid)
    # 224: expected image resolution
    # like in NLP: layernorm, attention, matmul, softmax, matmul, mlp. gelu - many small kernels
    
    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(args.device).eval()
elif args.model == 'retina':
    model = models.detection.retinanet_resnet50_fpn(pretrained=False, weights=None).to(args.device).eval()
elif args.model == 'vit':
    # timm ViT expects (B, C, H, W). Default checkpoint is 224; set img_size to match --vit-size.
    vs = args.vit_size
    if vs % 16 != 0:
        raise SystemExit("--vit-size must be divisible by 16 for vit_*_patch16_*")
    logger.info("create model")
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        img_size=vs,
    )
    logger.info("move model to device")
    model = model.to(args.device)
    logger.info("set model to eval")
    model = model.eval()
elif args.model == "sd15":
    model = _load_sd15_unet_benchmark(args.device, torch.float32)
elif args.model == "tinyllama":
    model = _load_tinyllama_decode_benchmark(args.device, torch.float32)
elif args.model == "whisper_enc":
    model = _load_whisper_encoder_benchmark(args.device, torch.float32)
    enc_cfg = model.encoder.config
    whisper_n_mels = int(enc_cfg.num_mel_bins)
    # HF WhisperModel rejects mel length != 2 * max_source_positions (e.g. 3000 for openai/whisper-*).
    whisper_mel_frames = int(enc_cfg.max_source_positions) * 2
    if args.mel_frames != whisper_mel_frames:
        logger.info(
            "Whisper expects mel length %s (2 * max_source_positions); using that instead of --mel-frames=%s.",
            whisper_mel_frames,
            args.mel_frames,
        )
else:
    raise RuntimeError(f"unknown model {args.model}")

logger.info("model created")

if args.autographs:
    logger.info("compile it with reduce-overhead")
    model = torch.compile(model, mode="reduce-overhead")
elif args.compile:
    logger.info("compile it")
    model = torch.compile(model)
else:
    logger.info("skipping compile")

# look for https://docs.pytorch.org/docs/stable/generated/torch.compile.html
# torch._inductor.list_mode_options()
# see options, in particular
# triton.cudagraphs which will reduce the overhead of python with CUDA graphs

logger.info("fill random")  # one image in a batch 3 channels, 224x224 pixels
if args.model == 'retina':
    size =  args.retina_size
    static_x = torch.randn(3,size,size,device=args.device)
    x = [static_x]
    is_list_input = True
elif args.model == 'vit':
    size = args.vit_size
    x = torch.randn(1, 3, size, size, device=args.device)
    is_list_input = False
elif args.model == "sd15":
    # SD 1.5 @ 512x512: latent (B, 4, 64, 64); same tensor each replay iteration (like fixed-shape denoise).
    x = torch.randn(1, 4, 64, 64, device=args.device, dtype=torch.float32)
    is_list_input = False
elif args.model == "tinyllama":
    t = args.llm_seq_len
    x = torch.randint(
        0,
        model.config.vocab_size,
        (1, t),
        device=args.device,
        dtype=torch.long,
    )
    is_list_input = False
elif args.model == "whisper_enc":
    assert whisper_n_mels is not None and whisper_mel_frames is not None
    x = torch.randn(
        1,
        whisper_n_mels,
        whisper_mel_frames,
        device=args.device,
        dtype=torch.float32,
    )
    is_list_input = False
else:
    x = torch.randn(1, 3, 224, 224, device=args.device)
    is_list_input = False

schedule = torch.profiler.schedule(
    wait=20, warmup=30, active=1000, repeat=1)

prof = None


def _write_profiler_key_averages_csv(profile, csv_path: str) -> None:
    """Export profiler.key_averages() to CSV (same ordering as table(sort_by=self_cuda_time_total))."""
    from torch.autograd.profiler import DeviceType

    events = list(profile.key_averages())
    if not events:
        logger.warning("Profiler key_averages empty; skip CSV %s", csv_path)
        return

    events.sort(key=lambda e: e.self_device_time_total, reverse=True)

    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if evt.device_type == DeviceType.CPU and evt.is_legacy:
            sum_self_device_time_total += evt.self_device_time_total
        elif (
            evt.device_type
            in (
                DeviceType.CUDA,
                DeviceType.PrivateUse1,
                DeviceType.MTIA,
                DeviceType.XPU,
            )
            and not getattr(evt, "is_user_annotation", False)
        ):
            sum_self_device_time_total += evt.self_device_time_total

    def _pct(numerator: int, denom: int) -> float:
        if denom == 0:
            return 0.0 if numerator == 0 else float("nan")
        return round(100.0 * numerator / denom, 2)

    use_device = (events[0].use_device or "device").upper()
    has_device_time = any(e.self_device_time_total > 0 for e in events)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if has_device_time:
            writer.writerow(
                [
                    "Name",
                    "Self CPU %",
                    "Self CPU (us)",
                    "CPU total %",
                    "CPU total (us)",
                    "CPU time avg (us)",
                    f"Self {use_device} (us)",
                    f"Self {use_device} %",
                    f"{use_device} total (us)",
                    f"{use_device} time avg (us)",
                    "# of Calls",
                ]
            )
        else:
            writer.writerow(
                [
                    "Name",
                    "Self CPU %",
                    "Self CPU (us)",
                    "CPU total %",
                    "CPU total (us)",
                    "CPU time avg (us)",
                    "# of Calls",
                ]
            )

        for evt in events:
            cnt = evt.count if evt.count > 0 else 1
            row = [
                evt.key,
                _pct(evt.self_cpu_time_total, sum_self_cpu_time_total),
                evt.self_cpu_time_total,
                (
                    _pct(evt.cpu_time_total, sum_self_cpu_time_total)
                    if not evt.is_async
                    else 0.0
                ),
                evt.cpu_time_total,
                round(evt.cpu_time_total / cnt, 3),
            ]
            if has_device_time:
                row.extend(
                    [
                        evt.self_device_time_total,
                        _pct(
                            evt.self_device_time_total, sum_self_device_time_total
                        ),
                        evt.device_time_total,
                        round(evt.device_time_total / cnt, 3),
                    ]
                )
            row.append(evt.count)
            writer.writerow(row)


with torch.inference_mode():
    logger.info("warmup")
    for _ in range(50):
        model(x)

    logger.info("synchronize after warmup")
    backend.synchronize()
    backend.empty_cache()

    N = args.iter

    if args.graphs:
        if args.model == "retina":
            _materialize_detection_transform_stats_on_device(model, args.device)
        logger.info("prepare graph")
        g = backend.create_graph()

        if is_list_input:
            static_x = [x[0].clone()]
        else:
            static_x = x.clone()

        with backend.graph(g):
            static_y = model(static_x)

        if not args.skip_consistency_check:
            ref_y = model(x)
            g.replay()
            backend.synchronize()
            _compare_outputs(ref_y, static_y, name="graph vs eager")
            logger.info("consistency check passed (graph vs eager)")

        logger.info("graph case: start %s iterations measured", N)
        start = time.time()

        def _one_graph_replay_iter():
            if is_list_input:
                static_x[0].copy_(x[0])
            else:
                static_x.copy_(x)
            g.replay()

        synch_name = "graph_replay_synch"

        if args.profiler:
            with torch.profiler.profile(
                schedule=schedule,
                acc_events=True,
                activities=_profiler_activities,
            ) as prof:
                with _itt_ctx(), profiler.record_function("graph_replay"):
                    remaining = N
                    while remaining > 0:
                        this_batch = min(batch_size, remaining)
                        for _ in range(this_batch):
                            if args.fine_grain_itt:
                                with profiler.record_function("graph_replay_iter"):
                                    _one_graph_replay_iter()
                            else:
                                _one_graph_replay_iter()
                            prof.step()
                        if args.fine_grain_itt:
                            with profiler.record_function(synch_name):
                                backend.synchronize()
                        else:
                            backend.synchronize()
                        remaining -= this_batch
        else:
            with _itt_ctx(), profiler.record_function("graph_replay"):
                remaining = N
                while remaining > 0:
                    this_batch = min(batch_size, remaining)
                    for _ in range(this_batch):
                        if args.fine_grain_itt:
                            with profiler.record_function("graph_replay_iter"):
                                _one_graph_replay_iter()
                        else:
                            _one_graph_replay_iter()
                    if args.fine_grain_itt:
                        with profiler.record_function(synch_name):
                            backend.synchronize()
                    else:
                        backend.synchronize()
                    remaining -= this_batch

    else:

        logger.info("non-graph case, start %s iterations measured", N)
        start = time.time()

        synch_name = "eager_synch"
        if args.profiler:
            with torch.profiler.profile(
                schedule=schedule,
                acc_events=True,
                activities=_profiler_activities,
            ) as prof:
                with _itt_ctx(), profiler.record_function("eager"):
                    remaining = N
                    while remaining > 0:
                        this_batch = min(batch_size, remaining)
                        for _ in range(this_batch):
                            if args.fine_grain_itt:
                                with profiler.record_function("eager_iter"):
                                    model(x)
                            else:
                                model(x)
                            prof.step()
                        if args.fine_grain_itt:
                            with profiler.record_function(synch_name):
                                backend.synchronize()
                        else:
                            backend.synchronize()
                        remaining -= this_batch
        else:
            with _itt_ctx(), profiler.record_function("eager"):
                remaining = N
                while remaining > 0:
                    this_batch = min(batch_size, remaining)
                    for _ in range(this_batch):
                        if args.fine_grain_itt:
                            with profiler.record_function("eager_iter"):
                                model(x)
                        else:
                            model(x)
                    if args.fine_grain_itt:
                        with profiler.record_function(synch_name):
                            backend.synchronize()
                    else:
                        backend.synchronize()
                    remaining -= this_batch

end_time = time.time()

if args.profiler:
    if prof is None:
        raise RuntimeError("internal error: --profiler set but profiler context did not set prof")
    logger.info("\n%s", prof.key_averages().table(sort_by="self_cuda_time_total"))
    _csv_path = os.path.abspath("real_world_app_profiler.csv")
    _write_profiler_key_averages_csv(prof, _csv_path)
    logger.info("Profiler key_averages CSV: %s", _csv_path)

latency_ms = 1000.0 * (end_time - start) / N
logger.info("Latency: %.3f msec", latency_ms)

if args.output_json:
    out_path = os.path.abspath(args.output_json)
    payload = {
        "latency_ms": latency_ms,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info("Latency JSON: %s", out_path)

