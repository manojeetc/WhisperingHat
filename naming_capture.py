"""Bounded local debug writer. No inference, uploads, or speech calls."""
import json
import hashlib
import math
from pathlib import Path
import queue
import tempfile
import threading

import cv2


def json_safe(value):
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


class NamingCapture:
    def __init__(self, directory, limit=5):
        directory = Path(directory).expanduser().resolve()
        if any((parent / ".git").exists() for parent in (directory, *directory.parents)):
            raise ValueError("naming captures must be outside Git")
        if not 1 <= limit <= 20:
            raise ValueError("naming capture limit must be between 1 and 20")
        self.directory, self.limit, self.count = directory, limit, 0
        self.items = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._write, name="naming-capture", daemon=True)
        self.worker.start()

    def submit(self, frame, metadata):
        # Called only after speech dispatch. Never wait for disk or a full queue.
        if self.count >= self.limit or self.items.full():
            return False
        try:
            self.items.put_nowait((frame.copy(), metadata))
            self.count += 1
            return True
        except queue.Full:
            return False
        except Exception as exc:
            print(f"[naming capture] could not queue frame: {exc}", flush=True)
            return False

    def _write(self):
        output = None
        while not self.stop.is_set() or not self.items.empty():
            try:
                frame, metadata = self.items.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                if output is None:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    output = Path(tempfile.mkdtemp(prefix="whisper-naming-", dir=self.directory))
                stem = f"frame-{metadata['frame_number']:06d}"
                image_path, context_path = output / f"{stem}.png", output / f"{stem}.json"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"could not write {image_path}")
                metadata = dict(metadata, image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
                context_path.write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8")
                print(f"[naming capture] image={image_path} context={context_path}", flush=True)
            except Exception as exc:
                print(f"[naming capture] write failed: {exc}", flush=True)
            finally:
                self.items.task_done()

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            print("[naming capture] shutdown with writes pending", flush=True)
