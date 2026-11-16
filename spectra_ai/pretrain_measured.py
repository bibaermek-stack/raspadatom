"""Prepare measured LIBS spectra and pretrain a masked-spectrum autoencoder."""
import argparse
import json
from pathlib import Path
import time
import zlib

import numpy as np

GRID = np.linspace(200, 900, 1024)
SOURCES = {
    "chemcam_single": "single-00000-of-00001.parquet",
    "chemcam_stat": "stat-00000-of-00001.parquet",
    "kepes": "test.h5",
}
CACHE = Path(".venv/measured_cache")
OUTPUT = Path("checkpoints/measured_libs.pt")


def partition(key):
    bucket = zlib.crc32(key.encode("utf-8")) % 100
    return 0 if bucket < 90 else (1 if bucket < 95 else 2)


def select_channels(wavelengths, grid):
    order = np.argsort(wavelengths)
    sorted_waves = wavelengths[order]
    right = np.searchsorted(sorted_waves, grid).clip(0, len(order) - 1)
    left = (right - 1).clip(0)
    nearest = np.where(abs(sorted_waves[left] - grid) < abs(sorted_waves[right] - grid), left, right)
    valid = abs(sorted_waves[nearest] - grid) <= 0.5
    return order[nearest[valid]], valid


def deflate_rows(compressed, dtype, width, selected, block_size=1024 * 1024):
    """Decode one HDF5 deflate chunk without retaining all its uncompressed rows."""
    selected = np.asarray(selected)
    order = np.argsort(selected)
    sorted_rows = selected[order]
    result = np.empty((width, len(selected)), dtype=np.float32)
    decoder = zlib.decompressobj()
    stride = dtype.itemsize * width
    offset, tail = 0, b""
    for start in range(0, len(compressed), block_size):
        decoded = tail + decoder.decompress(compressed[start:start + block_size])
        count, _ = divmod(len(decoded), stride)
        stop = count * stride
        first, last = np.searchsorted(sorted_rows, (offset, offset + count))
        if first != last:
            values = np.frombuffer(decoded, dtype=dtype, count=count * width).reshape(count, width)
            result[:, order[first:last]] = values[sorted_rows[first:last] - offset].T
        tail = decoded[stop:]
        offset += count
    if not decoder.eof or tail or offset <= sorted_rows[-1]:
        raise ValueError("Incomplete HDF5 deflate chunk")
    return result


def write_spectra(output, values, valid):
    good = np.isfinite(values).all(axis=1)
    np.nan_to_num(values, copy=False, nan=0, posinf=0, neginf=0)
    np.maximum(values, 0, out=values)
    scale = values.max(axis=1, keepdims=True)
    values /= np.maximum(scale, np.finfo(np.float32).tiny)
    np.sqrt(values, out=values)
    output[:, valid] = values
    return good & (scale[:, 0] > 0)


def prepare_parquet(path, cache, key):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(6)
    parquet = pq.ParquetFile(path)
    columns = [name for name in parquet.schema_arrow.names if name.startswith("wave@")]
    idx, valid = select_channels(np.array([float(name[5:]) for name in columns]), GRID)
    wanted = ["name", *dict.fromkeys(columns[i] for i in idx)]
    lookup = {name: j for j, name in enumerate(wanted)}
    order = [lookup[columns[i]] for i in idx]
    count = parquet.metadata.num_rows
    output = np.lib.format.open_memmap(cache / f"{key}.npy", mode="w+", dtype="float16",
                                      shape=(count, len(GRID)))
    splits = np.empty(count, dtype=np.uint8)
    start = 0
    next_log = 100_000
    for batch in parquet.iter_batches(batch_size=4096, columns=wanted):
        # Arrow projects in schema order, so resolve the names once per batch.
        if list(batch.schema.names) != wanted:
            order = [batch.schema.get_field_index(columns[i]) for i in idx]
        values = np.column_stack([batch.column(i).to_numpy() for i in order])
        stop = start + len(values)
        good = write_spectra(output[start:stop], values, valid)
        names = batch.column(batch.schema.get_field_index("name")).to_pylist()
        splits[start:stop] = [partition(name) if ok and name else 255 for name, ok in zip(names, good)]
        start = stop
        if start >= next_log:
            print(f"{key}: prepared {start:,}/{count:,}", flush=True)
            next_log += 100_000
    if start != count:
        raise ValueError(f"{path} yielded {start} rows, expected {count}")
    output.flush()
    return splits, valid


def prepare_h5(path, cache, key):
    import h5py

    with h5py.File(path) as source:
        wavelengths = next(iter(source["Wavelengths"].values()))[...].ravel()
        idx, valid = select_channels(wavelengths, GRID)
        spectra = list(source["UNKNOWN"].values())
        count = sum(d.shape[1] for d in spectra)
        output = np.lib.format.open_memmap(cache / f"{key}.npy", mode="w+", dtype="float16",
                                          shape=(count, len(GRID)))
        splits = np.empty(count, dtype=np.uint8)
        start = 0
        for dataset in spectra:
            n_waves, n_spectra = dataset.shape
            if n_waves != len(wavelengths):
                raise ValueError("HDF5 wavelength and intensity dimensions differ")
            filters = dataset.id.get_create_plist()
            if (dataset.chunks == dataset.shape and filters.get_nfilters() == 1
                    and filters.get_filter(0)[0] == h5py.h5z.FILTER_DEFLATE):
                skipped, chunk = dataset.id.read_direct_chunk((0, 0))
                if skipped == 0:
                    values = deflate_rows(chunk, dataset.dtype, n_spectra, idx)
                else:
                    values = np.frombuffer(chunk, dtype=dataset.dtype).reshape(dataset.shape)[idx].T.copy()
                del chunk
            else:
                sorted_idx, reverse = np.unique(idx, return_inverse=True)
                values = dataset[sorted_idx, :][reverse].T.astype(np.float32)
            stop = start + n_spectra
            good = write_spectra(output[start:stop], values, valid)
            splits[start:stop] = [
                partition(f"kepes:{dataset.name}:{i // 500}") if ok else 255
                for i, ok in enumerate(good)
            ]
            start = stop
            print(f"{key}: prepared {start:,}/{count:,}", flush=True)
        output.flush()
    return splits, valid


def prepare(data, cache, source_names):
    cache.mkdir(parents=True, exist_ok=True)
    for key in source_names:
        path = data / SOURCES[key]
        print(f"Reading {path}", flush=True)
        read = prepare_h5 if key == "kepes" else prepare_parquet
        splits, valid = read(path, cache, key)
        np.savez(cache / f"{key}.npz", splits=splits, valid=valid)
        print(f"{key}: {len(splits):,} rows; train/validation/test/invalid = "
              f"{[int((splits == i).sum()) for i in (0, 1, 2, 255)]}", flush=True)


def make_model():
    from torch import nn

    width = len(GRID)
    model = nn.Sequential(
        nn.Linear(2 * width, 384), nn.LayerNorm(384), nn.GELU(),
        nn.Linear(384, 192), nn.GELU(),
        nn.Linear(192, 384), nn.GELU(),
        nn.Linear(384, width),
    )
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    return model


def reconstruction(model, spectra, available, hidden):
    import torch

    visible = (available & ~hidden).expand_as(spectra)
    x = torch.where(visible, spectra, 0.0)
    position = torch.arange(len(GRID), device=x.device)
    left = torch.where(visible, position, 0).cummax(dim=1).values
    right = torch.where(visible, position, len(GRID) - 1).flip((1,)).cummin(dim=1).values.flip((1,))
    left_ok, right_ok = visible.gather(1, left), visible.gather(1, right)
    low, high = x.gather(1, left), x.gather(1, right)
    fraction = (position - left) / (right - left).clamp_min(1)
    baseline = torch.where(left_ok & right_ok, torch.lerp(low, high, fraction),
                           torch.where(left_ok, low, high))
    prediction = baseline + model(torch.cat((x, visible.to(x.dtype)), dim=1))
    return prediction, baseline


def save_checkpoint(torch, payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(cache, output, epochs, batch_size=512, resume=None, fast=False):
    import torch
    from math import cos, pi

    if min(epochs, batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    accelerated = fast and device.type == "cuda"
    amp = accelerated and torch.cuda.is_bf16_supported()
    data = {}
    for key in SOURCES:
        with np.load(cache / f"{key}.npz") as info:
            data[key] = (
                np.load(cache / f"{key}.npy", mmap_mode="r"),
                torch.tensor(info["valid"], device=device),
                [np.flatnonzero(info["splits"] == i) for i in (0, 1, 2, 255)],
            )
    counts = {
        key: dict(zip(("train", "validation", "test", "invalid"), map(len, splits)))
        for key, (_, _, splits) in data.items()
    }
    if any(min(counts[key][split] for split in ("train", "validation", "test")) < 1 for key in data):
        raise ValueError(f"Every source needs three nonempty splits: {counts}")

    if accelerated:
        cache_bytes = sum(spectra.nbytes for spectra, _, _ in data.values())
        if torch.cuda.mem_get_info()[0] > cache_bytes + 1700 * 1024**2:
            for key, (spectra, available, splits) in data.items():
                resident = torch.empty(spectra.shape, dtype=torch.float16, device=device)
                for start in range(0, len(spectra), 16384):
                    resident[start:start + 16384].copy_(torch.from_numpy(spectra[start:start + 16384].copy()))
                data[key] = resident, available, splits
            print(f"Cached {cache_bytes / 1024**3:.2f} GiB of spectra on CUDA", flush=True)

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, fused=accelerated)
    block = 8
    start_epoch, best, history = 0, float("inf"), []
    output.parent.mkdir(parents=True, exist_ok=True)
    latest = output.with_suffix(".last.pt")
    print(f"Training on {device}; batch={batch_size}, bfloat16={amp}: {counts}", flush=True)

    def batch(key, rows, rng):
        spectra, available, _ = data[key]
        if isinstance(spectra, torch.Tensor):
            x = spectra.index_select(0, torch.as_tensor(rows, device=device)).float()
        else:
            x = torch.from_numpy(spectra[rows]).to(device=device, dtype=torch.float32)
        hidden = (torch.rand((len(rows), len(GRID) // block), generator=rng, device=device)
                  < 0.20).repeat_interleave(block, dim=1)
        return x, available, hidden & available

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        scores = {}
        for key, (_, _, splits) in data.items():
            rng = torch.Generator(device=device).manual_seed(500 + split)
            total = torch.zeros(5, device=device)
            rows = splits[split]
            for start in range(0, len(rows), batch_size):
                x, available, hidden = batch(key, rows[start:start+batch_size], rng)
                prediction, baseline = reconstruction(model, x, available, hidden)
                residual = prediction.clamp(0, 1) - x
                reference = baseline - x
                total += torch.stack(((residual.square() * hidden).sum(),
                                      (reference.square() * hidden).sum(),
                                      (residual.abs() * hidden).sum(),
                                      (reference.abs() * hidden).sum(), hidden.sum()))
            mse, base_mse, mae, base_mae, n = total.tolist()
            scores[key] = {
                "spectra": len(rows), "masked_bins": int(n),
                "rmse": (mse / n) ** 0.5, "linear_interpolation_rmse": (base_mse / n) ** 0.5,
                "mae": mae / n, "linear_interpolation_mae": base_mae / n,
            }
        return scores

    if resume:
        prior = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(prior["state_dict"])
        start_epoch = prior["epoch"]
        if start_epoch > epochs:
            raise ValueError("Requested epoch precedes the checkpoint")
        if "optimizer" in prior:
            optimizer.load_state_dict(prior["optimizer"])
            history, best = prior["history"], prior["best"]
            if not output.exists():
                raise FileNotFoundError(f"Keep the companion best checkpoint at {output}")
        else:
            report_path = resume.with_suffix(".json")
            if report_path.exists():
                history = json.loads(report_path.read_text(encoding="utf-8"))["history"][:start_epoch]
            baseline = evaluate(1)
            best = float(np.mean([metrics["rmse"] for metrics in baseline.values()]))
            save_checkpoint(torch, prior, output)
        print(f"Resuming after epoch {start_epoch}; validation RMSE to beat {best:.5f}", flush=True)
        del prior

    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.perf_counter()
        ordering = np.random.default_rng([42, epoch])
        rng = torch.Generator(device=device).manual_seed(43 + epoch)
        if fast:
            progress = (epoch - 1) / max(1, epochs - 1)
            optimizer.param_groups[0]["lr"] = 4e-5 + 2.6e-4 * (1 + cos(pi * progress)) / 2
        model.train()
        batches = []
        for key, (_, _, splits) in data.items():
            rows = ordering.permutation(splits[0])
            batches.extend((key, rows[start:start+batch_size]) for start in range(0, len(rows), batch_size))
        sums = torch.zeros(2, device=device)
        for step, index in enumerate(ordering.permutation(len(batches)), 1):
            key, rows = batches[index]
            x, available, hidden = batch(key, rows, rng)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction, _ = reconstruction(model, x, available, hidden)
            error = ((prediction - x).square() * hidden).sum()
            bins = hidden.sum()
            optimizer.zero_grad(set_to_none=True)
            (error / bins.clamp_min(1)).backward()
            optimizer.step()
            sums += torch.stack((error.detach(), bins))
            if step % 500 == 0:
                print(f"Epoch {epoch}/{epochs}: {step}/{len(batches)} batches", flush=True)
        error, bins = sums.tolist()
        val = evaluate(1)
        macro_rmse = float(np.mean([metrics["rmse"] for metrics in val.values()]))
        duration = time.perf_counter() - started
        history.append({"epoch": epoch, "train_rmse": (error / bins) ** 0.5, "seconds": duration,
                        "validation_macro_rmse": macro_rmse, "validation": val})
        print(f"Epoch {epoch}/{epochs}: train RMSE {history[-1]['train_rmse']:.5f}, "
              f"validation macro RMSE {macro_rmse:.5f} | {duration:.1f}s", flush=True)
        payload = {
            "state_dict": model.state_dict(), "wavelength_nm": torch.tensor(GRID),
            "epoch": epoch, "mask_block_bins": block,
            "normalization": "sqrt(max(intensity, 0) / per_spectrum_peak)",
            "source_coverage": {key: available.cpu() for key, (_, available, _) in data.items()},
        }
        if macro_rmse < best:
            best = macro_rmse
            save_checkpoint(torch, payload, output)
        save_checkpoint(torch, {**payload, "optimizer": optimizer.state_dict(),
                                "history": history, "best": best}, latest)

    ck = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    report = {
        "sources": {key: {"file": SOURCES[key], **counts[key]} for key in SOURCES},
        "device": str(device), "selected_epoch": ck["epoch"], "completed_epochs": epochs,
        "batch_size": batch_size, "bfloat16": amp,
        "mask_fraction": 0.2, "history": history, "test": evaluate(2),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": report["sources"], "test": report["test"]}, indent=2), flush=True)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=Path("data"))
    prep.add_argument("--cache", type=Path, default=CACHE)
    prep.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    fit = actions.add_parser("train")
    fit.add_argument("--cache", type=Path, default=CACHE)
    fit.add_argument("--out", type=Path, default=OUTPUT)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.cache, args.sources)
    else:
        train(args.cache, args.out, args.epochs, args.batch_size, args.resume, args.fast)


if __name__ == "__main__":
    main()
