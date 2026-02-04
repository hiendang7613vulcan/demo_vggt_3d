"""
Generate multi-view images using fal-ai/qwen-image-edit-2511-multiple-angles.
All settings are driven by a YAML config file.

Usage:
  python multiview_generate.py --config multiview_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

try:
    import fal_client
except Exception as exc:  # pragma: no cover - runtime import
    raise RuntimeError(
        "fal_client is required. Install with: pip install fal-client"
    ) from exc


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_input(cfg: Dict[str, Any]) -> Dict[str, str]:
    input_cfg = cfg.get("input", {}) or {}
    image_path = (input_cfg.get("image_path") or "").strip()
    image_url = (input_cfg.get("image_url") or "").strip()

    if image_path:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        return {"type": "path", "value": str(path)}

    if image_url:
        return {"type": "url", "value": image_url}

    raise ValueError("Set input.image_path or input.image_url in the config.")


def set_fal_key(cfg: Dict[str, Any]) -> None:
    fal_cfg = cfg.get("fal", {}) or {}
    api_key = (fal_cfg.get("api_key") or "").strip()
    api_env = (fal_cfg.get("api_key_env") or "FAL_KEY").strip()

    if api_key:
        os.environ[api_env] = api_key

    if not os.environ.get(api_env):
        raise EnvironmentError(
            f"Missing API key. Set {api_env} or fal.api_key in the config."
        )


def upload_if_needed(input_info: Dict[str, str]) -> str:
    if input_info["type"] == "url":
        return input_info["value"]
    return fal_client.upload_file(input_info["value"])


def download_to(path: Path, url: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def filename_for(
    pattern: str,
    base: str,
    name: str,
    h: float,
    v: float,
    i: int,
    ext: str,
) -> str:
    return pattern.format(base=base, name=name, h=h, v=v, i=i, ext=ext)


def run_generation(cfg: Dict[str, Any]) -> None:
    set_fal_key(cfg)

    input_info = pick_input(cfg)
    output_cfg = cfg.get("output", {}) or {}
    output_dir = Path(output_cfg.get("dir", "outputs/multiview")).expanduser()
    ensure_dir(output_dir)

    pattern = output_cfg.get("filename_pattern", "{base}_{name}_h{h}_v{v}_{i}.{ext}")
    overwrite = bool(output_cfg.get("overwrite", False))
    write_manifest = bool(output_cfg.get("write_manifest", True))

    fal_cfg = cfg.get("fal", {}) or {}
    model = fal_cfg.get("model", "fal-ai/qwen-image-edit-2511-multiple-angles")
    params = fal_cfg.get("params", {}) or {}

    output_format = (params.get("output_format") or "png").lower()
    num_images = int(params.get("num_images") or 1)
    use_original_when_zero = bool(cfg.get("use_original_when_zero", True))

    angles: List[Dict[str, Any]] = cfg.get("angles", []) or []
    if not angles:
        raise ValueError("Config must include at least one entry in angles.")

    # Resolve input URL once (upload local file only once)
    input_url = upload_if_needed(input_info)

    # Base name for filenames
    if input_info["type"] == "path":
        base = Path(input_info["value"]).stem
    else:
        base = Path(input_info["value"].split("?")[0]).stem or "input"

    manifest: Dict[str, Any] = {
        "input": input_info,
        "output_dir": str(output_dir),
        "model": model,
        "params": params,
        "angles": angles,
        "results": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for angle in angles:
        name = str(angle.get("name") or "view")
        h = float(angle.get("horizontal", 0))
        v = float(angle.get("vertical", 0))

        # If angle is 0/0 and config says to use original, copy/download once.
        if use_original_when_zero and h == 0 and v == 0:
            out_name = filename_for(pattern, base, name, h, v, 0, output_format)
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                manifest["results"].append({"angle": angle, "file": str(out_path), "skipped": True})
                continue

            if input_info["type"] == "path":
                shutil.copyfile(input_info["value"], out_path)
            else:
                download_to(out_path, input_info["value"])

            manifest["results"].append({"angle": angle, "file": str(out_path), "copied": True})
            continue

        # Build request
        request_args = dict(params)
        request_args.update({
            "image_urls": [input_url],
            "horizontal_angle": h,
            "vertical_angle": v,
        })

        result = fal_client.subscribe(model, arguments=request_args, with_logs=True)
        images = result.get("images", [])

        saved: List[str] = []
        for i, img in enumerate(images[:num_images]):
            url = img.get("url")
            if not url:
                continue
            out_name = filename_for(pattern, base, name, h, v, i, output_format)
            out_path = output_dir / out_name
            if out_path.exists() and not overwrite:
                saved.append(str(out_path))
                continue
            download_to(out_path, url)
            saved.append(str(out_path))

        manifest["results"].append({"angle": angle, "files": saved, "api_result": result})

    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if write_manifest:
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-view images from a YAML config.")
    parser.add_argument("--config", default="multiview_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    run_generation(cfg)


if __name__ == "__main__":
    main()
