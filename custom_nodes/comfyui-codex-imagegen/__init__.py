"""Codex Image Generate — ComfyUI custom node.

Generates an image through the local Codex CLI ($imagegen tool, ChatGPT
subscription auth) and returns it as a ComfyUI IMAGE tensor, so cloud
generation can be mixed with local checkpoints inside one workflow.

Requires: `codex` CLI installed and logged in (`codex login`). No API key.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_CODEX_FALLBACKS = sorted(
    glob.glob(str(Path.home() / ".nvm/versions/node/*/bin/codex"))
) + [
    "/opt/homebrew/bin/codex",
    "/usr/local/bin/codex",
]

_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _resolve_codex() -> str:
    found = shutil.which("codex")
    if found:
        return found
    for candidate in _CODEX_FALLBACKS:
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "codex CLI not found. Install it and run `codex login` "
        "(https://developers.openai.com/codex), then restart ComfyUI."
    )


def _collect_images(root: Path, since: float) -> list[Path]:
    hits: list[Path] = []
    for pattern in _IMAGE_EXTS:
        hits.extend(p for p in root.rglob(pattern) if p.stat().st_mtime >= since)
    return sorted(hits, key=lambda p: p.stat().st_mtime)


class CodexImageGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cute robot artist painting on a canvas in a sunny studio, digital art",
                    },
                ),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"],),
                "model": ("STRING", {"default": "gpt-5.5"}),
            },
            "optional": {
                "pose_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/codex"

    def generate(self, prompt: str, aspect_ratio: str, model: str, pose_image=None):
        codex = _resolve_codex()
        gen_root = Path.home() / ".codex" / "generated_images"
        gen_root.mkdir(parents=True, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="comfy-codex-")
        started = time.time()

        text = (prompt or "").strip() or "A high-quality reference image."

        pose_image_path = None
        if pose_image is not None:
            frame = pose_image[0].detach().cpu().numpy()
            frame = np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)
            pose_img = Image.fromarray(frame)
            pose_fd, pose_image_path = tempfile.mkstemp(suffix=".png", prefix="comfy-codex-pose-")
            os.close(pose_fd)
            pose_img.save(pose_image_path)
            full_prompt = (
                "$imagegen Generate a new image that follows the exact body pose "
                "shown in the attached OpenPose skeleton reference image: "
                f"{text}\nAspect ratio: {aspect_ratio}."
            )
        else:
            full_prompt = f"$imagegen {text}\nAspect ratio: {aspect_ratio}."

        args = [
            codex,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-c",
            "sandbox_workspace_write.network_access=true",
            "-C",
            workdir,
            "--add-dir",
            str(gen_root),
            "--model",
            (model or "gpt-5.5").strip(),
        ]
        if pose_image_path:
            # `-i/--image` is a variadic clap arg: without a `--` terminator it
            # greedily swallows the positional PROMPT that follows it too,
            # which makes codex fall back to (empty) stdin for the prompt.
            args += ["-i", pose_image_path, "--"]
        args.append(full_prompt)

        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=600)
        finally:
            if pose_image_path:
                try:
                    os.remove(pose_image_path)
                except OSError:
                    pass
            shutil.rmtree(workdir, ignore_errors=True)

        thread_id = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("type") == "thread.started" and isinstance(obj.get("thread_id"), str):
                thread_id = obj["thread_id"]

        candidates: list[Path] = []
        if thread_id and (gen_root / thread_id).is_dir():
            candidates = _collect_images(gen_root / thread_id, since=0)
        if not candidates and proc.returncode == 0:
            # Last-resort scan of the shared root. Only trusted on a clean
            # exit: on failure a recent image here likely belongs to another
            # concurrently running node, not to this execution.
            candidates = _collect_images(gen_root, since=started)

        if not candidates:
            stderr_tail = (proc.stderr or "").strip()[-400:]
            stdout_tail = (proc.stdout or "").strip()[-400:]
            raise RuntimeError(
                "codex imagegen produced no image "
                f"(exit {proc.returncode}). stderr: {stderr_tail or '<empty>'} "
                f"stdout: {stdout_tail or '<empty>'}"
            )

        img = Image.open(candidates[-1]).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return (tensor,)


NODE_CLASS_MAPPINGS = {"CodexImageGenerate": CodexImageGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"CodexImageGenerate": "Codex Image Generate (CLI)"}
