"""Video of an agent control run: every camera side by side, with the step number, elapsed time and estimated model
cost in the top right corner. Recorded in real time in a background thread. agent_control starts it, and the elapsed
time, at the first model call.

Frames are piped to ffmpeg (H.264, plays in browsers). ffmpeg runs in its own session so that Ctrl+C in the terminal
does not kill it before it has finalized the file. Without ffmpeg, OpenCV's MPEG-4 Part 2 writer ("mp4v") is used
instead; that plays in VLC / mpv but often not in browsers.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


class _FfmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        # fmt: off
        command = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            str(path),
        ]
        # fmt: on
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, start_new_session=True)

    def write(self, frame_bgr: np.ndarray) -> None:
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait(timeout=120)


class _OpenCvWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        assert self._writer.isOpened(), f"could not open a video writer for {path}"

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()


class RunVideoRecorder:
    """Records `get_image(camera_id)` (RGB, or None before a camera's first frame) for every camera at `fps`."""

    def __init__(
        self,
        path: Path,
        camera_ids: Sequence[str],
        get_image: Callable[[str], np.ndarray | None],
        t0: float,
        fps: float = 10.0,
        panel_height: int = 376,
    ):
        self.path = path
        self._camera_ids = list(camera_ids)
        self._get_image = get_image
        self._t0 = t0  # wall clock time the overlay's elapsed time counts from
        self._fps = fps
        self._panel_height = panel_height
        self._panel_widths: dict[str, int] = {}
        self._step: int | None = None
        self._cost_usd = 0.0
        self._writer: _FfmpegWriter | _OpenCvWriter | None = None
        self._frame_size: tuple[int, int] | None = None  # (width, height), fixed by the first frame
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_step(self, step: int) -> None:
        self._step = step

    def set_cost(self, cost_usd: float) -> None:
        self._cost_usd = cost_usd

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop recording and finalize the file. Safe to call more than once."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _run(self) -> None:
        t_start = time.time()
        n_written = 0
        while not self._stop.is_set():
            frame = self._compose()
            if self._writer is None:
                self._open_writer(frame)
            frame = self._fit(frame)
            # Write as many frames as wall clock time calls for, so playback runs in real time even if a tick is late.
            n_due = max(1, int((time.time() - t_start) * self._fps) + 1 - n_written)
            for _ in range(n_due):
                self._writer.write(frame)
            n_written += n_due
            self._stop.wait(max(0.0, t_start + n_written / self._fps - time.time()))

    def _open_writer(self, frame: np.ndarray) -> None:
        # yuv420p needs even dimensions
        height, width = frame.shape[0] + frame.shape[0] % 2, frame.shape[1] + frame.shape[1] % 2
        self._frame_size = (width, height)
        writer_cls = _FfmpegWriter if shutil.which("ffmpeg") else _OpenCvWriter
        self._writer = writer_cls(self.path, width, height, self._fps)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        width, height = self._frame_size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        return frame

    def _compose(self) -> np.ndarray:
        """All cameras side by side at a common height (BGR), each labelled, with the run overlay."""
        panels = []
        for cam_id in self._camera_ids:
            image = self._get_image(cam_id)
            if image is None:
                width = self._panel_widths.get(cam_id, int(self._panel_height * 16 / 9))
                panel = np.zeros((self._panel_height, width, 3), dtype=np.uint8)
            else:
                width = round(image.shape[1] * self._panel_height / image.shape[0])
                panel = cv2.cvtColor(cv2.resize(image, (width, self._panel_height)), cv2.COLOR_RGB2BGR)
            self._panel_widths[cam_id] = width
            cv2.putText(panel, cam_id, (10, 28), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        frame = np.hstack(panels)
        self._draw_overlay(frame)
        return frame

    def _draw_overlay(self, frame: np.ndarray) -> None:
        elapsed = time.time() - self._t0
        lines = [
            f"step {'-' if self._step is None else self._step}",
            f"time {int(elapsed // 60):02d}:{elapsed % 60:04.1f}",
            f"cost ${self._cost_usd:.4f}",
        ]
        scale, thickness, line_height, margin = 0.7, 2, 26, 10
        text_width = max(cv2.getTextSize(line, FONT, scale, thickness)[0][0] for line in lines)
        x0 = frame.shape[1] - text_width - 2 * margin
        box = frame[0 : len(lines) * line_height + margin, x0:]
        box[:] = (0.4 * box).astype(np.uint8)  # darken behind the text so it reads on any image
        for i, line in enumerate(lines):
            line_width = cv2.getTextSize(line, FONT, scale, thickness)[0][0]
            origin = (frame.shape[1] - margin - line_width, (i + 1) * line_height)
            cv2.putText(frame, line, origin, FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)
