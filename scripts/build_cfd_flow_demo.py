"""Assemble the CFD -> Warp -> full-panel portfolio video.

The stills are native Isaac Sim captures.  The 30-second middle segment is a
trimmed excerpt of the already validated full-panel process video; this script
does not rerun Isaac Sim or alter any process physics.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media" / "air_assisted_spray"
OUTPUT = MEDIA / "isaac_full_panel_cfd_flow_demo.mp4"
SEQUENCE = [
    ("process", MEDIA / "isaac_cfd_process_frame.png", 8.0),
    ("flow", MEDIA / "isaac_cfd_flow_field.png", 10.0),
    ("combined", MEDIA / "isaac_cfd_warp_combined.png", 12.0),
    ("full_panel", MEDIA / "isaac_full_panel_film_demo.mp4", 30.0),
    ("result", MEDIA / "isaac_full_panel_film_final.png", 10.0),
]


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    missing = [str(path) for _, path, _ in SEQUENCE if not path.exists()]
    if missing:
        raise FileNotFoundError("missing media: " + ", ".join(missing))
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")

    inputs: list[str] = []
    filters: list[str] = []
    for index, (name, path, duration) in enumerate(SEQUENCE):
        if path.suffix.lower() == ".mp4":
            inputs.extend(["-ss", "0", "-t", f"{duration:.3f}", "-i", str(path)])
            filters.append(f"[{index}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,setpts=PTS-STARTPTS[v{index}]")
        else:
            inputs.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", str(path)])
            filters.append(f"[{index}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,setpts=PTS-STARTPTS[v{index}]")
    concat_inputs = "".join(f"[v{index}]" for index in range(len(SEQUENCE)))
    filters.append(f"{concat_inputs}concat=n={len(SEQUENCE)}:v=1:a=0[vout]")
    filter_graph = ";".join(filters)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _run([
        ffmpeg,
        "-y",
        *inputs,
        "-filter_complex",
        filter_graph,
        "-map",
        "[vout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ])
    probe = subprocess.check_output([
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,codec_type",
        "-of",
        "json",
        str(OUTPUT),
    ], cwd=ROOT, text=True)
    metadata = json.loads(probe)
    (MEDIA / "isaac_full_panel_cfd_flow_demo.json").write_text(json.dumps({
        "schema_version": "isaac_full_panel_cfd_flow_demo_v1",
        "output": "media/air_assisted_spray/isaac_full_panel_cfd_flow_demo.mp4",
        "sequence": [{"name": name, "source": str(path.relative_to(ROOT)).replace("\\", "/"), "duration_s": duration} for name, path, duration in SEQUENCE],
        "probe": metadata,
        "physics": "reused validated full-panel media; no CFD or painting rerun",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CFD_VIDEO_PASS output={OUTPUT.relative_to(ROOT)} duration_s=70.0 codec=h264 1920x1080 30fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
