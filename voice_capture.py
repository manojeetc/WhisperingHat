"""Single supervised microphone owner. Native PortAudio calls stay in the child."""
import json
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time


class CaptureCleanupFailed(RuntimeError):
    pass


def _send(pipe, tag, payload):
    pipe.write(struct.pack("!I", len(payload) + 1) + tag + payload)
    pipe.flush()


def _exact(pipe, size):
    data = bytearray()
    while len(data) < size:
        part = pipe.read(size - len(data))
        if not part:
            raise EOFError("capture process closed its audio pipe")
        data.extend(part)
    return bytes(data)


def capture_worker(config):
    import pyaudio
    audio, stream = None, None
    pipe = sys.stdout.buffer
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=config["rate"],
                            input=True, input_device_index=config["index"], frames_per_buffer=1024)
        info = (audio.get_default_input_device_info() if config["index"] is None
                else audio.get_device_info_by_index(config["index"]))
        _send(pipe, b"R", json.dumps(dict(rate=config["rate"], width=2, chunk=1024,
                                        name=info["name"])).encode())
        while True:
            # PyAudio's blocking read has no supported timeout. The parent can
            # terminate/reap this entire owner, including a stuck read/close.
            data = stream.read(1024, exception_on_overflow=False)
            _send(pipe, b"A", struct.pack("!d", time.monotonic()) + data)
    except Exception as error:
        _send(pipe, b"E", f"{type(error).__name__}: {error}".encode())
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if audio is not None:
            audio.terminate()


class CaptureProcess:
    """Bounded PCM queue and reads; reap the old owner before any replacement."""
    def __init__(self, index, rate, check=lambda: None, on_audio=lambda data, stamp: data,
                 command=None, stall_timeout=3.0):
        self.check, self.on_audio, self.stall_timeout = check, on_audio, stall_timeout
        self.frames = queue.Queue(maxsize=8)
        self.ready = threading.Event()
        self.info, self.error, self.closed = None, None, False
        self.process = subprocess.Popen(command or [sys.executable, str(Path(__file__).resolve()),
            "--worker", json.dumps(dict(index=index, rate=rate))], stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=sys.platform != "win32", bufsize=0)
        self.reader = threading.Thread(target=self._receive, name="microphone-pipe", daemon=True)
        self.reader.start()
        try:
            deadline = time.monotonic() + 8
            while not self.ready.wait(.05):
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("microphone open exceeded 8s")
            self.check()
            if self.error:
                raise OSError(self.error)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        try:
            while True:
                length = struct.unpack("!I", _exact(self.process.stdout, 4))[0]
                if not 1 <= length <= 1024 * 1024:
                    raise ValueError("invalid capture packet size")
                packet = _exact(self.process.stdout, length)
                tag, payload = packet[:1], packet[1:]
                if tag == b"R":
                    self.info = json.loads(payload)
                    self.ready.set()
                elif tag == b"E":
                    raise OSError(payload.decode(errors="replace"))
                elif tag == b"A":
                    stamp = struct.unpack("!d", payload[:8])[0]
                    if len(payload) <= 8:
                        raise ValueError("empty microphone packet")
                    try:
                        self.frames.put_nowait((payload[8:], stamp))
                    except queue.Full:
                        try:
                            self.frames.get_nowait()
                        except queue.Empty:
                            pass
                        self.frames.put_nowait((payload[8:], stamp))
                else:
                    raise ValueError("invalid capture packet type")
        except Exception as error:
            self.error = str(error)
            self.ready.set()

    def read(self, size):
        deadline = time.monotonic() + self.stall_timeout
        while True:
            self.check()
            if self.closed:
                raise OSError("capture is closed")
            try:
                data, stamp = self.frames.get(timeout=.05)
                self.check()
                # Do not feed old speaker audio after a slow decoder finishes.
                if time.monotonic() - stamp > .5:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("microphone supplied only stale PCM")
                    continue
                return self.on_audio(data, stamp)
            except queue.Empty:
                if self.error:
                    raise OSError(self.error)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"microphone produced no PCM for {self.stall_timeout:g}s (silence is valid PCM)")

    def close(self):
        if self.closed:
            return
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=.75)
        except subprocess.TimeoutExpired:
            self.process.kill()
            try:
                self.process.wait(timeout=.75)
            except subprocess.TimeoutExpired as error:
                raise CaptureCleanupFailed("capture owner could not be reaped; refusing reopen") from error
        self.reader.join(timeout=.5)
        if self.reader.is_alive():
            raise CaptureCleanupFailed("capture pipe did not reach EOF; refusing reopen")
        self.process.stdout.close()
        self.closed = True
        print(f"[voice capture] owner pid={self.process.pid} reaped; stream released.", flush=True)

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("internal microphone capture worker")
    capture_worker(json.loads(sys.argv[2]))
