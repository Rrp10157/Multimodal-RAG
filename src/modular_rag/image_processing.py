import io
import json
import logging
import re

import ollama as ollama_client
import PIL.Image
import PIL.ImageOps

from modular_rag.config import VISION_MODEL

logger = logging.getLogger(__name__)

COLOUR_TABLE_PROMPT = """\
This image shows a data table from an Indian economic report.
The table uses COLOUR CODING to show performance relative to historical averages:
  GREEN or TEAL cells = ABOVE historical average (POSITIVE)
  RED or PINK cells = BELOW historical average (NEGATIVE)
  YELLOW or AMBER = NEUTRAL
  WHITE = no special significance

Extract all visible data with colour context and output plain text.
"""

DUAL_CHART_PROMPT = """\
This image contains exactly two charts side by side.
Extract left and right charts separately with title, axes, series and key values.
"""

SINGLE_CHART_PROMPT = """\
Identify chart/table type and extract all visible numbers and labels.
Return strict JSON with title, subtitle, data, key facts and source note.
"""

EQUATION_PROMPT = """\
Extract every equation exactly as written, each on a separate line.
Include variable definitions and short context if present.
"""


def _image_bytes(image: PIL.Image.Image) -> bytes:
    img = PIL.ImageOps.exif_transpose(image).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def classify_image(image: PIL.Image.Image) -> str:
    width, height = image.size
    aspect = width / height if height > 0 else 1.0
    if aspect > 1.6:
        try:
            resp = ollama_client.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": "Reply with one word: dual, table, chart, equation, or other.",
                    "images": [_image_bytes(image)],
                }],
                options={"num_predict": 5, "temperature": 0},
            )
            answer = resp["message"]["content"].strip().lower()
            if "dual" in answer:
                return "dual_chart"
            if "table" in answer:
                return "colour_table"
            if "chart" in answer:
                return "single_chart"
            if "equat" in answer:
                return "equation"
        except Exception:
            pass
        return "dual_chart"
    if 0.7 < aspect < 1.6:
        return "single_chart"
    return "other"


def _extract_with_prompt(image: PIL.Image.Image, prompt: str, max_tokens: int) -> str:
    try:
        resp = ollama_client.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [_image_bytes(image)]}],
            options={"num_predict": max_tokens, "temperature": 0},
        )
        return resp["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Image extraction failed: %s", exc)
        return "[Image extraction unavailable]"


def extract_colour_table(image: PIL.Image.Image) -> str:
    return _extract_with_prompt(image, COLOUR_TABLE_PROMPT, 1500)


def extract_dual_chart(image: PIL.Image.Image) -> str:
    return _extract_with_prompt(image, DUAL_CHART_PROMPT, 1500)


def extract_single_chart(image: PIL.Image.Image) -> str:
    raw = _extract_with_prompt(image, SINGLE_CHART_PROMPT, 1024)
    clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(clean)
        title = parsed.get("title") or ""
        lines = [f"TITLE: {title}"] if title else []
        for fact in parsed.get("key_facts", []):
            lines.append(f"FACT: {fact}")
        return "\n".join(lines) if lines else raw
    except Exception:
        return raw


def extract_equation(image: PIL.Image.Image) -> str:
    return _extract_with_prompt(image, EQUATION_PROMPT, 512)


def describe_image_smart(image: PIL.Image.Image, ref: str = "") -> str:
    img_type = classify_image(image)
    logger.info("Image type=%s ref=%s", img_type, ref)
    if img_type == "dual_chart":
        return "DUAL CHART EXTRACTION:\n" + extract_dual_chart(image)
    if img_type == "colour_table":
        return "COLOUR-CODED TABLE EXTRACTION:\n" + extract_colour_table(image)
    if img_type == "equation":
        return "MATHEMATICAL EQUATION:\n" + extract_equation(image)
    return "CHART/IMAGE EXTRACTION:\n" + extract_single_chart(image)
