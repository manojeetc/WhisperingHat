"""Offline region-selection experiment. Inputs must already have correct orientation."""
import argparse
import csv
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import cv2
import numpy as np


def straighten(image, quad):
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or not cv2.isContourConvex(points):
        raise ValueError("quad must contain four finite convex corners in perimeter order")
    height, width = image.shape[:2]
    if (points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any():
        raise ValueError("quad lies outside image")
    # First corner is top-left, then clockwise; automatic rectangles are ordered below.
    out_width = int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3]))))
    out_height = int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1]))))
    if min(out_width, out_height) < 2:
        raise ValueError("quad is degenerate")
    target = np.float32([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]])
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (out_width, out_height))
    return cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def select_region(image):
    """Rank dark-on-light text-line contours using geometry and center distance."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 50) | 1, 7)))
    mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 35) | 1, 3)))
    candidates = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        center, sides, angle = cv2.minAreaRect(contour)
        long_side, short_side = max(sides), min(sides)
        if short_side < 8 or not 2 <= long_side / short_side <= 18:
            continue
        if long_side < width * 0.08 or short_side > height * 0.2:
            continue
        angle += 90 if sides[0] < sides[1] else 0
        angle = (angle + 90) % 180 - 90
        if abs(angle) > 35:
            continue
        # Padding is relative to detected text height, never to a particular subject.
        center = np.asarray(center)
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        normal = np.array([-direction[1], direction[0]])
        along, across = direction * long_side * .525, normal * short_side * .625
        quad = np.float32([center - along - across, center + along - across,
                           center + along + across, center - along + across])
        quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
        if not cv2.isContourConvex(quad):
            continue
        distance = np.linalg.norm(center / [width, height] - 0.5)
        fill = cv2.contourArea(contour) / max(long_side * short_side, 1)
        score = fill * min(long_side / short_side, 8) * np.sqrt(long_side * short_side / (width * height)) * np.exp(-4 * distance)
        candidates.append((float(score), quad))
    # ponytail: one geometric line hypothesis; learned text/layout detection is a later experiment.
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def text_difference(reference, actual):
    words = lambda text: re.findall(r"\w+", text.casefold())
    expected, recognized = words(reference), words(actual)
    edits = [dict(operation=tag, expected=expected[i:j], actual=recognized[k:l])
             for tag, i, j, k, l in difflib.SequenceMatcher(None, expected, recognized, autojunk=False).get_opcodes()
             if tag != "equal"]
    positions = []
    for line in reference.splitlines():
        tokens = words(line)
        if not tokens:
            continue
        position = next((i for i in range(len(recognized) - len(tokens) + 1)
                         if recognized[i:i + len(tokens)] == tokens), None)
        positions.append(dict(reference_line=line, word_position=position))
    found = [item["word_position"] for item in positions if item["word_position"] is not None]
    return dict(word_edits=edits, reference_lines=positions,
                line_order_assessable=len(found) >= 2,
                reordered_exact_lines=found != sorted(found))


def recognize(path, psm, engine, tessdata, language):
    if engine == "tesserocr":
        import tesserocr
        with tesserocr.PyTessBaseAPI(path=str(tessdata), lang=language, psm=psm) as api:
            api.SetImageFile(str(path))
            text, tsv = api.GetUTF8Text(), api.GetTSVText(0)
        header = "level page_num block_num par_num line_num word_num left top width height conf text".split()
        rows = csv.DictReader(io.StringIO(tsv), fieldnames=header, delimiter="\t")
        version = tesserocr.tesseract_version().splitlines()[0]
    else:
        command = ["tesseract", str(path), "stdout", "-l", language, "--psm", str(psm)]
        if tessdata:
            command += ["--tessdata-dir", str(tessdata)]
        # Reconstruct text in TSV order without a second recognition pass.
        result = subprocess.run(command + ["tsv"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=30, check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
        lines = {}
        for row in rows:
            if row.get("level") == "5" and row.get("text", "").strip():
                key = tuple(row[key] for key in ("page_num", "block_num", "par_num", "line_num"))
                lines.setdefault(key, []).append(row["text"])
        text = "\n".join(" ".join(line) for line in lines.values())
        version = subprocess.run(["tesseract", "--version"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=5, check=True).stdout.splitlines()[0]
    words = [dict(text=row["text"], confidence=float(row["conf"])) for row in rows
             if row.get("level") == "5" and row.get("text", "").strip()]
    return text, words, version


def compare(image_path, output_root, engine="cli", tessdata=None, language="eng", manual_quad=None, reference=None):
    image_path = Path(image_path).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot decode image: {image_path}")
    root = Path(output_root).resolve()
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise ValueError("private crops must be saved outside Git working trees")
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="ocr-regions-", dir=root))
    height, width = image.shape[:2]
    variants = [("full", image_path, None, 0.0, [3, 6, 7, 11])]
    overlay = image.copy()
    for name, color in (("center", (255, 0, 0)), ("automatic", (0, 255, 0)), ("manual-control", (0, 0, 255))):
        started = time.perf_counter()
        if name == "center":
            # Fixed 70% center window is a baseline, not page localization.
            quad = np.float32([[width * .15, height * .15], [width * .85, height * .15],
                               [width * .85, height * .85], [width * .15, height * .85]])
        elif name == "automatic":
            quad = select_region(image)
        else:
            quad = manual_quad
        if quad is None:
            continue
        crop = straighten(image, quad)
        selection_ms = (time.perf_counter() - started) * 1000
        path = directory / f"{name}.png"
        if not cv2.imwrite(str(path), crop):
            raise OSError(f"could not write {path}")
        cv2.polylines(overlay, [np.int32(quad)], True, color, 2)
        variants.append((name, path, np.asarray(quad).tolist(), selection_ms, [7] if name == "manual-control" else [6, 7]))
    if not cv2.imwrite(str(directory / "regions.png"), overlay):
        raise OSError("could not save region overlay")
    results = []
    for name, path, quad, selection_ms, modes in variants:
        for psm in modes:
            started = time.perf_counter()
            try:
                text, words, version = recognize(path, psm, engine, tessdata, language)
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = dict(region=name, quad=quad, image=str(path), psm=psm,
                              selection_ms=selection_ms, ocr_ms=elapsed_ms,
                              total_ms=selection_ms + elapsed_ms, text=text, words=words, engine=version)
                if reference:
                    result["comparison"] = text_difference(reference, text)
            except (OSError, subprocess.SubprocessError, RuntimeError, ImportError) as error:
                result = dict(region=name, psm=psm, error=str(error), image=str(path), quad=quad,
                              selection_ms=selection_ms, ocr_ms=(time.perf_counter() - started) * 1000)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report = dict(input=str(image_path), dimensions=[width, height], language=language,
                  input_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(), opencv=cv2.__version__,
                  engine_backend=engine, tessdata=str(tessdata) if tessdata else None,
                  automatic_found=any(v[0] == "automatic" for v in variants), results=results)
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report and private crops: {directory}", flush=True)
    if all("error" in result for result in results):
        raise RuntimeError("all OCR attempts failed; see report.json")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="local directory outside Git")
    parser.add_argument("--engine", choices=["cli", "tesserocr"], default="cli")
    parser.add_argument("--tessdata", type=Path)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--manual-quad", nargs=8, type=float, metavar="COORD", help="optional control: TL TR BR BL pixel coordinates")
    parser.add_argument("--reference", type=Path, help="UTF-8 printed text, one physical line per line")
    args = parser.parse_args()
    if args.engine == "tesserocr" and args.tessdata is None:
        parser.error("--engine tesserocr requires --tessdata")
    compare(args.image, args.output, args.engine, args.tessdata, args.language,
            np.asarray(args.manual_quad, np.float32).reshape(4, 2) if args.manual_quad else None,
            args.reference.read_text(encoding="utf-8") if args.reference else None)


if __name__ == "__main__":
    main()
