"""Document the deterministic media-recording entry point for the local scene."""

from __future__ import annotations

import shutil


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to encode the rendered PNG sequence")
    print("ffmpeg is available. Record from the locally composed Isaac Sim scene at 30 fps.")
    print("The published media/demo.mp4 is the reviewed portfolio capture.")


if __name__ == "__main__":
    main()
