"""
parse_chandra_output.py
-----------------------
Parse Chandra 2 HTML output → dots.ocr-compatible JSON format.

Chandra 2 HTML output example:
    <div data-bbox="111 67 204 86" data-label="Section-Header">
        <h2>Vận dụng</h2>
    </div>

dots.ocr JSON format:
    [{"bbox": [111, 67, 204, 86], "category": "Section-header", "text": "..."}, ...]

Với Table: text giữ nguyên HTML <table>...</table> để không mất cấu trúc.
"""

import json
import re
from typing import Optional


LABEL_MAP = {
    "section-header":   "Section-header",
    "title":            "Title",
    "text":             "Text",
    "caption":          "Caption",
    "footnote":         "Footnote",
    "formula":          "Formula",
    "equation-block":   "Formula",
    "list-item":        "List-item",
    "list-group":       "List-item",
    "page-header":      "Page-header",
    "page-footer":      "Page-footer",
    "picture":          "Picture",
    "figure":           "Picture",
    "image":            "Picture",
    "diagram":          "Picture",
    "table":            "Table",
    "table-of-contents":"Text",
    "code-block":       "Text",
    "chemical-block":   "Formula",
    "bibliography":     "Footnote",
    "complex-block":    "Text",
}

DOTS_OCR_CATEGORIES = {
    "Caption", "Footnote", "Formula", "List-item",
    "Page-footer", "Page-header", "Picture",
    "Section-header", "Table", "Text", "Title",
}


def _normalize_label(raw_label: str) -> str:
    key = raw_label.strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    title_case = raw_label.strip().title().replace(" ", "-")
    if title_case in DOTS_OCR_CATEGORIES:
        return title_case
    return "Text"


def _parse_bbox(bbox_str: str) -> list:
    try:
        return [int(float(p)) for p in bbox_str.strip().split()[:4]]
    except Exception:
        return [0, 0, 0, 0]


def _extract_inner_html(div_content: str) -> str:
    """Lấy nội dung bên trong div (sau khi bỏ tag div ngoài cùng)."""
    # Bỏ opening tag của div ngoài cùng
    inner = re.sub(r"^<div[^>]*>", "", div_content, count=1)
    # Bỏ closing </div> cuối cùng
    inner = re.sub(r"</div>\s*$", "", inner)
    return inner.strip()


def _html_to_text(html: str) -> str:
    """Strip tất cả HTML tags, giữ lại text thuần."""
    # Xử lý math: wrap trong $...$
    html = re.sub(r"<math>(.*?)</math>", r"$\1$", html, flags=re.DOTALL)
    # Bỏ tất cả tags còn lại
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    return " ".join(text.split()).strip()


def parse_chandra_html(html: str) -> list:
    """
    Parse Chandra 2 HTML string → list of dots.ocr-compatible dicts.

    - Table block: field "text" chứa HTML <table>...</table> nguyên vẹn
                   để không mất cấu trúc rows/cols/headers.
    - Các block khác: field "text" là plain text (math wrap trong $...$).
    """
    if not re.search(r'<div\s[^>]*data-bbox', html, re.IGNORECASE):
        text = html.strip()
        return [{"bbox": [0, 0, 0, 0], "category": "Text", "text": text}] if text else []

    results = []

    # Match từng div có data-bbox — dùng regex thay vì HTMLParser
    # để giữ nguyên inner HTML của table mà không bị parse lại
    pattern = re.compile(
        r'<div\s[^>]*data-bbox=["\']([^"\']+)["\'][^>]*data-label=["\']([^"\']+)["\'][^>]*>'
        r'(.*?)'
        r'</div>',
        re.DOTALL | re.IGNORECASE,
    )

    # Cũng match thứ tự ngược (data-label trước data-bbox)
    pattern_rev = re.compile(
        r'<div\s[^>]*data-label=["\']([^"\']+)["\'][^>]*data-bbox=["\']([^"\']+)["\'][^>]*>'
        r'(.*?)'
        r'</div>',
        re.DOTALL | re.IGNORECASE,
    )

    # Collect tất cả matches từ cả 2 pattern, sort theo position
    matches = []
    for m in pattern.finditer(html):
        bbox_str, label_raw, inner = m.group(1), m.group(2), m.group(3)
        matches.append((m.start(), bbox_str, label_raw, inner))
    for m in pattern_rev.finditer(html):
        label_raw, bbox_str, inner = m.group(1), m.group(2), m.group(3)
        matches.append((m.start(), bbox_str, label_raw, inner))

    # Sort theo position trong document
    matches.sort(key=lambda x: x[0])

    # Dedup (cùng position)
    seen_pos = set()
    for pos, bbox_str, label_raw, inner in matches:
        if pos in seen_pos:
            continue
        seen_pos.add(pos)

        bbox     = _parse_bbox(bbox_str)
        category = _normalize_label(label_raw)
        inner    = inner.strip()

        if category == "Table":
            # Giữ nguyên HTML table để không mất cấu trúc
            table_match = re.search(r'<table.*?</table>', inner, re.DOTALL | re.IGNORECASE)
            text = table_match.group(0).strip() if table_match else _html_to_text(inner)
        else:
            text = _html_to_text(inner)

        if text:
            results.append({
                "bbox":     bbox,
                "category": category,
                "text":     text,
            })

    return results


def parse_chandra_html_to_json_string(html: str) -> str:
    """Wrapper trả về JSON string — dùng trực tiếp trong field 'response'."""
    return json.dumps(parse_chandra_html(html), ensure_ascii=False)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        '<div data-bbox="111 67 204 86" data-label="Section-Header"><h2>Vận dụng</h2></div>'
        '<div data-bbox="117 94 893 155" data-label="Text"><p>Chú Đức lái ô tô. '
        'Vận tốc <math>x</math> (km/h).</p></div>'
        '<div data-bbox="92 229 927 856" data-label="Table">'
        '<table border="1"><thead><tr><th>Cột 1</th><th>Cột 2</th></tr></thead>'
        '<tbody><tr><td>A</td><td>B</td></tr></tbody></table>'
        '</div>'
        '<div data-bbox="853 939 881 957" data-label="Page-Footer"><p>19</p></div>'
    )
    results = parse_chandra_html(sample)
    print(json.dumps(results, ensure_ascii=False, indent=2))
