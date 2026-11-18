"""Request-only offline captioning/OCR. Heavy runtimes stay in low-priority processes."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import threading
import time
import urllib.request

FLORENCE_REVISION = "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
OCR_ASSETS = {
    "ch_PP-OCRv5_det_mobile.onnx": ("det", "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"),
    "en_PP-OCRv5_rec_mobile.onnx": ("rec", "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8"),
}


def model_directory():
    return Path(os.environ.get("WHISPER_HAT_VISION_MODELS", str(Path.home() / ".cache" / "whisperinghat" / "voice-vision")))


def short_sentence(text):
    text = " ".join(re.sub(r"<[^>]+>", "", text).split())
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if not text or len(text.split()) > 45:
        raise ValueError("caption is empty or not a short sentence")
    return text if text[-1] in ".!?" else text + "."


def ordered_regions(boxes, texts, scores):
    """Retain recognized strings verbatim, in the detector's document reading order."""
    # ponytail: single-column reading order; add layout analysis for columns.
    return [dict(text=str(text), confidence=float(score), box=box.tolist())
            for box, text, score in zip(boxes, texts, scores)]


def ocr_reliable(regions, shape=None):
    # Unlike Tesseract's confidence scale, PP-OCR scores are in [0, 1].
    if not regions or not all(r["text"].strip() and .80 <= r["confidence"] <= 1 for r in regions):
        return False
    # Edge-touching regions may contain confidently recognized, clipped words.
    if shape is not None:
        height, width = shape[:2]
        if any(x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2
               for row in regions for x, y in row["box"]):
            return False
    return True


def load_backend(kind, root):
    root = Path(root)
    if kind == "caption":
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        torch.set_num_threads(1)
        folder = str(root / "florence2")
        processor = AutoProcessor.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation="eager").eval()
        return processor, model
    from rapidocr import RapidOCR, OCRVersion, ModelType, LangRec
    for filename in OCR_ASSETS:
        if not (root / "ppocr" / filename).is_file():
            raise FileNotFoundError("Run python voice_vision.py --setup before using PP-OCRv5")
    return RapidOCR(params={
        "Global.use_cls": False, "Global.text_score": 0.0, "Global.log_level": "error",
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.EN,
        "Det.model_path": str(root / "ppocr" / "ch_PP-OCRv5_det_mobile.onnx"),
        "Rec.model_path": str(root / "ppocr" / "en_PP-OCRv5_rec_mobile.onnx"),
        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def infer(kind, backend, frame):
    if kind == "caption":
        import torch
        from PIL import Image
        processor, model = backend
        image = Image.fromarray(frame[:, :, ::-1])
        inputs = processor(text="<CAPTION>", images=image, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=48, do_sample=False, num_beams=1, early_stopping=False)
        generated = processor.batch_decode(ids, skip_special_tokens=False)[0]
        caption = processor.post_process_generation(generated, task="<CAPTION>", image_size=image.size)["<CAPTION>"]
        return dict(text=short_sentence(caption), engine="Florence-2-base")
    # RapidOCR detects quadrilaterals, perspective-rectifies each text crop and
    # recognizes it; it never generates or paraphrases the recognized strings.
    result = backend(frame)
    regions = [] if result.boxes is None else ordered_regions(result.boxes, result.txts, result.scores)
    return dict(text="\n".join(row["text"] for row in regions), regions=regions,
                reliable=ocr_reliable(regions, frame.shape), engine="PP-OCRv5 mobile (ONNX)")


def _serve(kind, root, pipe):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if hasattr(os, "nice"):
        os.nice(10)
    try:
        backend = load_backend(kind, root)
        print(f"[voice vision] {kind} ready: {'Florence-2-base' if kind == 'caption' else 'PP-OCRv5 mobile'} (local, request only)", flush=True)
        pipe.send(dict(ready=True))
        while True:
            frame = pipe.recv()
            if frame is None:
                return
            try:
                pipe.send(infer(kind, backend, frame))
            except Exception as error:
                pipe.send(dict(error=f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        pipe.send(dict(error=f"{type(error).__name__}: {error}"))
    finally:
        pipe.close()


class RequestModel:
    """One bounded request at a time; a hung inference can be killed safely."""
    def __init__(self, kind, root=None):
        self.kind, self.ready = kind, False
        self.lock = threading.Lock()
        context = mp.get_context("spawn")
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(kind, str(root or model_directory()), child), daemon=True)
        self.process.start()
        child.close()

    def request(self, frame, timeout):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(f"{self.kind} is busy")
        try:
            if not self.ready:
                if not self.pipe.poll(0):
                    raise RuntimeError(f"{self.kind} is loading or unavailable")
                status = self.pipe.recv()
                if not status.get("ready"):
                    raise RuntimeError(status.get("error", "model unavailable"))
                self.ready = True
            self.pipe.send(frame)
            if not self.pipe.poll(timeout):
                self.close()
                raise TimeoutError(f"{self.kind} exceeded {timeout}s; process stopped")
            result = self.pipe.recv()
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            self.lock.release()

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=.5)
        self.pipe.close()
        self.ready = False


def setup(root):
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    if any((p / ".git").exists() for p in (root, *root.parents)):
        raise ValueError("model weights must be outside Git")
    snapshot_download("microsoft/Florence-2-base", revision=FLORENCE_REVISION,
                      local_dir=root / "florence2", ignore_patterns=["pytorch_model.bin", "*.md", ".gitattributes"])
    (root / "ppocr").mkdir(exist_ok=True)
    for filename, (task, digest) in OCR_ASSETS.items():
        path = root / "ppocr" / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            url = f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/{task}/{filename}"
            temporary = path.with_suffix(".download")
            urllib.request.urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError(f"model checksum mismatch: {filename}")
            temporary.replace(path)
    print(f"Offline models ready: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--models", type=Path, default=model_directory())
    parser.add_argument("--kind", choices=["caption", "ocr"])
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unmirror", action="store_true")
    args = parser.parse_args()
    if args.setup:
        setup(args.models)
        return
    if not args.kind or not args.image:
        parser.error("use --setup or --kind caption|ocr --image PATH")
    import cv2
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f"cannot decode {args.image}")
    if args.unmirror:
        frame = cv2.flip(frame, 1)
    backend = load_backend(args.kind, args.models)
    results = []
    for _ in range(2):
        started = time.perf_counter()
        result = infer(args.kind, backend, frame)
        results.append(dict(result, elapsed_seconds=time.perf_counter() - started))
    report = dict(input=str(args.image.resolve()), unmirrored=args.unmirror, attempts=results,
                  scope="Windows inference only; command-to-speech and Pi speed unverified")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
