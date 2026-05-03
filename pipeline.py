"""
pipeline.py
============================================================
Weeding Defect Detection Pipeline — for HARP/INC test sheets.

Every function in this file corresponds verbatim to one cell of
the original notebook (Weeding_Defect_Detection_in_Cut_graphics.ipynb).
The algorithm logic is identical — only the I/O layer was changed:
- file upload widget instead of files.upload()
- single image instead of IMAGE_PATHS list
- function returns instead of plt.show()

Public entry-point: run_full_pipeline(image_path) -> dict
"""

from collections import Counter

import cv2
import numpy as np


# ============================================================
# CELL 3 — Smart crop to sheet
# ============================================================
def load_image(path):
    """Load image from disk, convert BGR -> RGB."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not open: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_to_sheet(img_rgb):
    """
    Detect the sheet boundary (bright region) and crop to it.

    Strategy:
      - Convert to grayscale
      - Threshold at fixed level to find the bright sheet region
      - Find the largest bright rectangle = the sheet
      - Crop with a small margin
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((20, 20), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_rgb

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    if area < img_area * 0.1:
        return img_rgb

    x, y, w, h = cv2.boundingRect(largest)
    margin = 15
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_rgb.shape[1] - x, w + 2 * margin)
    h = min(img_rgb.shape[0] - y, h + 2 * margin)
    return img_rgb[y:y + h, x:x + w]


# ============================================================
# CELL 4 — Grayscale
# ============================================================
def to_grayscale(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# ============================================================
# CELL 5 — Otsu threshold
# ============================================================
def smart_threshold(gray):
    """Otsu thresholding with a small Gaussian blur to suppress noise."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, binary_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_otsu, float(otsu_val)


# ============================================================
# CELL 6 — Auto invert so letters are white
# ============================================================
def ensure_white_on_black(binary):
    white_px = np.sum(binary == 255)
    black_px = np.sum(binary == 0)
    if white_px > black_px:
        return cv2.bitwise_not(binary)
    return binary


# ============================================================
# CELL 7 — Denoise + border crop
# ============================================================
def denoise(binary):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def crop_border(binary, border_px=10):
    result = binary.copy()
    result[:border_px, :] = 0
    result[-border_px:, :] = 0
    result[:, :border_px] = 0
    result[:, -border_px:] = 0
    return result


# ============================================================
# CELL 8 — Find blobs (connected components)
# ============================================================
def find_blobs(clean_mask):
    """Filtered connected-component analysis with image-relative thresholds."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        clean_mask, connectivity=8)

    img_h, img_w = clean_mask.shape
    total_px = img_h * img_w
    MIN_AREA = int(total_px * 0.00010)
    MAX_AREA = int(total_px * 0.08)
    BORDER_MARGIN = int(min(img_h, img_w) * 0.003)

    blobs = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        if x <= BORDER_MARGIN or y <= BORDER_MARGIN:
            continue
        if (x + w) >= (img_w - BORDER_MARGIN):
            continue
        if (y + h) >= (img_h - BORDER_MARGIN):
            continue
        if area < MIN_AREA or area > MAX_AREA:
            continue
        aspect = w / (h + 1e-6)
        if aspect > 8 or aspect < 0.1:
            continue

        roi = clean_mask[y:y + h, x:x + w]
        blobs.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "cx": float(cx), "cy": float(cy),
            "roi": roi, "aspect_ratio": round(float(aspect), 3),
        })

    return blobs, {"MIN_AREA": MIN_AREA, "MAX_AREA": MAX_AREA,
                   "BORDER_MARGIN": BORDER_MARGIN}


# ============================================================
# CELL 10 — Row detection (HARP/INC track split + zip)
# ============================================================
def split_blobs_by_xzone(blobs, x_split_ratio=0.6):
    if not blobs:
        return [], []
    all_cx = [b["cx"] for b in blobs]
    img_width_est = max(all_cx) + 50
    split_x = img_width_est * x_split_ratio
    left = [b for b in blobs if b["cx"] < split_x]
    right = [b for b in blobs if b["cx"] >= split_x]
    return left, right


def gap_cluster_into_rows(blobs):
    if not blobs:
        return []
    sorted_blobs = sorted(blobs, key=lambda b: b["cy"])
    if len(sorted_blobs) == 1:
        return [sorted_blobs]

    cy_vals = [b["cy"] for b in sorted_blobs]
    gaps = [cy_vals[i] - cy_vals[i - 1] for i in range(1, len(cy_vals))]
    if not gaps:
        return [sorted_blobs]

    median_gap = np.median(gaps)
    median_h = np.median([b["h"] for b in blobs])
    threshold = max(median_gap * 1.5, median_h * 0.6)

    rows = []
    current = [sorted_blobs[0]]
    for i in range(1, len(sorted_blobs)):
        gap = cy_vals[i] - cy_vals[i - 1]
        if gap > threshold:
            rows.append(sorted(current, key=lambda b: b["cx"]))
            current = [sorted_blobs[i]]
        else:
            current.append(sorted_blobs[i])
    rows.append(sorted(current, key=lambda b: b["cx"]))
    return rows


def zip_tracks_into_rows(left_rows, right_rows):
    if not left_rows and not right_rows:
        return []
    if not left_rows:
        return right_rows
    if not right_rows:
        return left_rows

    left_cy = [np.mean([b["cy"] for b in r]) for r in left_rows]
    right_cy = [np.mean([b["cy"] for b in r]) for r in right_rows]

    used_right = set()
    combined = []

    for li, lcy in enumerate(left_cy):
        best_ri, best_dist = -1, float("inf")
        for ri, rcy in enumerate(right_cy):
            if ri in used_right:
                continue
            d = abs(lcy - rcy)
            if d < best_dist:
                best_dist = d
                best_ri = ri
        if best_ri >= 0 and best_dist < 300:
            used_right.add(best_ri)
            merged = left_rows[li] + right_rows[best_ri]
        else:
            merged = left_rows[li]
        combined.append(sorted(merged, key=lambda b: b["cx"]))

    for ri, rrow in enumerate(right_rows):
        if ri not in used_right:
            combined.append(sorted(rrow, key=lambda b: b["cx"]))

    return combined


def assign_columns_conflict_free(rows, anchors):
    for row_idx, row in enumerate(rows):
        n_blobs = len(row)
        n_anchors = len(anchors)

        dist = np.zeros((n_blobs, n_anchors))
        for i, blob in enumerate(row):
            for j, a in enumerate(anchors):
                dist[i, j] = abs(blob["cx"] - a)

        assigned_anchor = [-1] * n_blobs
        used_anchors = set()
        pairs = [(dist[i, j], i, j)
                 for i in range(n_blobs)
                 for j in range(n_anchors)]
        pairs.sort()

        blobs_assigned = set()
        for d, i, j in pairs:
            if i in blobs_assigned or j in used_anchors:
                continue
            assigned_anchor[i] = j
            blobs_assigned.add(i)
            used_anchors.add(j)
            if len(blobs_assigned) == n_blobs or len(used_anchors) == n_anchors:
                break

        for i, blob in enumerate(row):
            blob["row"] = row_idx
            blob["col"] = assigned_anchor[i]
            blob["group"] = assigned_anchor[i]


def compute_column_anchors(rows):
    if not rows:
        return [], 0
    lengths = Counter(len(r) for r in rows)
    expected_cols = lengths.most_common(1)[0][0]
    complete_rows = [r for r in rows if len(r) == expected_cols]
    anchors = []
    for col_idx in range(expected_cols):
        xs = [row[col_idx]["cx"] for row in complete_rows]
        anchors.append(float(np.mean(xs)))
    return anchors, expected_cols


def detect_row_anomalies(rows, expected_col_count):
    anomalies = {}
    for row_idx, row in enumerate(rows):
        n = len(row)
        if n < expected_col_count:
            anomalies[row_idx] = f"missing {expected_col_count - n} blob(s)"
        elif n > expected_col_count:
            anomalies[row_idx] = f"extra {len(row) - expected_col_count} blob(s)"
    return anomalies


def detect_rows_and_columns(blobs):
    """Top-level wrapper for cell 10: returns clean_blobs + groups."""
    if not blobs:
        return {}, [], {}

    left_blobs, right_blobs = split_blobs_by_xzone(blobs, x_split_ratio=0.6)
    left_rows = gap_cluster_into_rows(left_blobs)
    right_rows = gap_cluster_into_rows(right_blobs)
    rows = zip_tracks_into_rows(left_rows, right_rows)
    anchors, expected_cols = compute_column_anchors(rows)
    assign_columns_conflict_free(rows, anchors)
    anomalies = detect_row_anomalies(rows, expected_cols)

    groups = {}
    for row in rows:
        for blob in row:
            gid = blob["group"]
            groups.setdefault(gid, []).append(blob)

    clean_blobs = [b for row in rows for b in row]
    return groups, clean_blobs, anomalies


# ============================================================
# CELL 11 — Build reference templates
# ============================================================
def filter_extreme_aspect(blobs, min_aspect=0.2, max_aspect=3.5):
    kept, rejected = [], []
    for b in blobs:
        if min_aspect <= b["aspect_ratio"] <= max_aspect:
            kept.append(b)
        else:
            rejected.append(b)
    return kept, rejected


def resize_roi(roi, target_size=(200, 200)):
    return cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)


def compute_solidity_if_missing(blob):
    if "solidity" in blob:
        return
    roi = blob["roi"]
    if roi is None:
        blob["solidity"] = 0.0
        return
    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        blob["solidity"] = 0.0
        return
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    blob["solidity"] = (
        round(blob["area"] / hull_area, 3) if hull_area > 0 else 0.0)


def fit_row_line(row_blobs):
    if len(row_blobs) == 1:
        return 0.0, row_blobs[0]["cy"]
    xs = np.array([b["cx"] for b in row_blobs])
    ys = np.array([b["cy"] for b in row_blobs])
    if np.std(xs) < 1e-6:
        return 0.0, float(np.mean(ys))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def cluster_large_into_rows(large_blobs):
    if not large_blobs:
        return []
    sorted_blobs = sorted(large_blobs, key=lambda b: b["cy"])
    if len(sorted_blobs) == 1:
        return [sorted_blobs]

    gaps = [sorted_blobs[i]["cy"] - sorted_blobs[i - 1]["cy"]
            for i in range(1, len(sorted_blobs))]
    median_h = np.median([b["h"] for b in large_blobs])
    p75_gap = float(np.percentile(gaps, 75))
    row_break_threshold = max(p75_gap * 1.2, median_h * 0.5)

    rows = []
    current_row = [sorted_blobs[0]]
    for i in range(1, len(sorted_blobs)):
        gap = sorted_blobs[i]["cy"] - sorted_blobs[i - 1]["cy"]
        if gap > row_break_threshold:
            rows.append(current_row)
            current_row = [sorted_blobs[i]]
        else:
            current_row.append(sorted_blobs[i])
    rows.append(current_row)
    return rows


def assign_small_blobs_to_large_rows(small_blobs, large_rows):
    if not large_rows:
        return {i: [] for i in range(len(large_rows))}, list(small_blobs)

    if len(small_blobs) >= 3:
        small_areas = [b["area"] for b in small_blobs]
        median_small_area = np.median(small_areas)
        max_allowed_area = median_small_area * 3.0
    else:
        max_allowed_area = float("inf")

    row_lines = [fit_row_line(row) for row in large_rows]
    row_heights = [np.mean([b["h"] for b in row]) for row in large_rows]

    small_by_row = {i: [] for i in range(len(large_rows))}
    orphans = []

    for blob in small_blobs:
        if blob["area"] > max_allowed_area:
            orphans.append(blob)
            continue
        bx, by = blob["cx"], blob["cy"]
        distances = []
        for (slope, intercept) in row_lines:
            expected_y = slope * bx + intercept
            distances.append(abs(by - expected_y))

        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        allowed = row_heights[nearest_idx] * 1.5
        if nearest_dist <= allowed:
            small_by_row[nearest_idx].append(blob)
        else:
            orphans.append(blob)

    return small_by_row, orphans


def build_super_blob(component_blobs, clean_mask, max_height=None):
    if not component_blobs:
        return None

    xs = [b["x"] for b in component_blobs]
    ys = [b["y"] for b in component_blobs]
    xs_end = [b["x"] + b["w"] for b in component_blobs]
    ys_end = [b["y"] + b["h"] for b in component_blobs]

    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs_end), max(ys_end)
    w = x_max - x_min
    h = y_max - y_min

    if max_height is not None and h > max_height:
        return None

    clean_roi = np.zeros((h, w), dtype=np.uint8)
    for comp in component_blobs:
        cx_off = comp["x"] - x_min
        cy_off = comp["y"] - y_min
        clean_roi[cy_off:cy_off + comp["h"], cx_off:cx_off + comp["w"]] = (
            np.maximum(
                clean_roi[cy_off:cy_off + comp["h"],
                          cx_off:cx_off + comp["w"]],
                comp["roi"]))

    total_area = sum(b["area"] for b in component_blobs)
    return {
        "x": int(x_min), "y": int(y_min), "w": int(w), "h": int(h),
        "area": int(total_area),
        "cx": x_min + w / 2.0,
        "cy": y_min + h / 2.0,
        "roi": clean_roi,
        "aspect_ratio": round(w / (h + 1e-6), 3),
        "component_count": len(component_blobs),
        "components": component_blobs,
        "is_super": True,
    }


def compute_trimmed_reference(blobs_in_column, template_size=(200, 200),
                              trim_fraction=0.25, is_super=False):
    if not blobs_in_column:
        return None

    for b in blobs_in_column:
        compute_solidity_if_missing(b)

    areas = np.array([b["area"] for b in blobs_in_column])
    median_area = np.median(areas)
    median_solidity = np.median([b["solidity"] for b in blobs_in_column])
    median_aspect = np.median([b["aspect_ratio"] for b in blobs_in_column])

    deviations = []
    for b in blobs_in_column:
        a_dev = abs(b["area"] - median_area) / (median_area + 1e-6)
        s_dev = abs(b["solidity"] - median_solidity)
        ar_dev = (abs(b["aspect_ratio"] - median_aspect)
                  / (median_aspect + 1e-6))
        deviations.append(a_dev + s_dev * 2 + ar_dev)

    n_keep = max(2, int(len(blobs_in_column) * (1 - trim_fraction)))
    sorted_indices = np.argsort(deviations)
    trusted = [blobs_in_column[i] for i in sorted_indices[:n_keep]]

    if is_super and len(trusted) >= 3:
        comp_counts = [b["component_count"] for b in trusted]
        most_common_count = Counter(comp_counts).most_common(1)[0][0]
        trusted_by_count = [
            b for b in trusted if b["component_count"] == most_common_count]
        if len(trusted_by_count) >= 2:
            trusted = trusted_by_count

    resized_rois = [resize_roi(b["roi"], template_size) for b in trusted]
    stacked = np.stack(resized_rois, axis=0)
    template_median = np.median(stacked, axis=0).astype(np.uint8)
    _, template = cv2.threshold(
        template_median, 127, 255, cv2.THRESH_BINARY)

    if is_super:
        kernel_close = np.ones((3, 3), np.uint8)
        template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((2, 2), np.uint8)
        template = cv2.morphologyEx(template, cv2.MORPH_OPEN, kernel_open)

    ref = {
        "count": len(blobs_in_column),
        "trusted_count": len(trusted),
        "excluded_count": len(blobs_in_column) - len(trusted),
        "ref_area": int(np.median([b["area"] for b in trusted])),
        "ref_aspect": round(
            float(np.median([b["aspect_ratio"] for b in trusted])), 3),
        "ref_solidity": round(
            float(np.median([b["solidity"] for b in trusted])), 3),
        "ref_width": int(np.median([b["w"] for b in trusted])),
        "ref_height": int(np.median([b["h"] for b in trusted])),
        "template": template,
        "area_std": float(np.std([b["area"] for b in trusted])),
    }
    if all(b.get("is_super") for b in trusted):
        ref["ref_component_count"] = int(
            np.median([b["component_count"] for b in trusted]))
    return ref


def group_large_blobs_into_columns(large_blobs, image_width):
    if not large_blobs:
        return {}, []

    sorted_by_x = sorted(large_blobs, key=lambda b: b["cx"])
    xs = [b["cx"] for b in sorted_by_x]
    gaps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    if not gaps:
        return {0: [sorted_by_x[0]]}, [xs[0]]

    median_gap = np.median(gaps)
    column_break = max(median_gap * 3.0, image_width * 0.03)

    columns = [[sorted_by_x[0]]]
    for i in range(1, len(sorted_by_x)):
        if xs[i] - xs[i - 1] > column_break:
            columns.append([sorted_by_x[i]])
        else:
            columns[-1].append(sorted_by_x[i])

    col_map = {i: c for i, c in enumerate(columns)}
    anchors = [float(np.mean([b["cx"] for b in c])) for c in columns]
    return col_map, anchors


def build_references(clean_blobs, clean_img):
    """Wrapper for Cell 11 — builds large_refs + super_refs."""
    valid_blobs = [b for b in clean_blobs if b.get("group", -1) != -1]
    if not valid_blobs:
        return {}, {}, [], [], [], []

    img_h, img_w = clean_img.shape[:2]
    valid_blobs, _ = filter_extreme_aspect(valid_blobs)

    areas = np.array([b["area"] for b in valid_blobs])
    median_area = np.median(areas)
    size_threshold = median_area * 0.3
    large_blobs = [b for b in valid_blobs if b["area"] >= size_threshold]
    small_blobs = [b for b in valid_blobs if b["area"] < size_threshold]

    large_rows = cluster_large_into_rows(large_blobs)
    small_by_row, orphans = assign_small_blobs_to_large_rows(
        small_blobs, large_rows)

    if small_blobs:
        typical_small_h = np.median([b["h"] for b in small_blobs])
        max_super_height = int(typical_small_h * 2.5)
    else:
        max_super_height = None

    super_blobs = []
    for row_idx in range(len(large_rows)):
        components = small_by_row[row_idx]
        if components:
            super_b = build_super_blob(
                components, clean_img, max_height=max_super_height)
            if super_b is not None:
                super_b["row"] = row_idx
                super_blobs.append(super_b)

    large_columns, large_anchors = group_large_blobs_into_columns(
        large_blobs, img_w)

    large_refs = {}
    for col_idx, col_blobs in large_columns.items():
        ref = compute_trimmed_reference(
            col_blobs, trim_fraction=0.25, is_super=False)
        large_refs[col_idx] = ref
        for b in col_blobs:
            b["large_col"] = col_idx

    super_refs = {}
    if super_blobs:
        ref = compute_trimmed_reference(
            super_blobs, trim_fraction=0.25, is_super=True)
        super_refs[0] = ref

    return (large_refs, super_refs, large_blobs, large_rows,
            super_blobs, large_anchors)


# ============================================================
# CELL 12 — Defect scoring + missing-letter detection
# ============================================================
EDGE_MARGIN = 20


def touches_edge(blob, img_w, img_h, margin=EDGE_MARGIN):
    x, y, w, h = blob["x"], blob["y"], blob["w"], blob["h"]
    return (x <= margin or y <= margin
            or (x + w) >= (img_w - margin)
            or (y + h) >= (img_h - margin))


def is_oversized_artifact(blob, ref, max_area_ratio=3.0):
    return blob["area"] > ref["ref_area"] * max_area_ratio


def identify_edge_columns(large_refs, large_columns_map, img_w, img_h,
                          threshold=0.5):
    skip = set()
    for col_idx, ref in large_refs.items():
        members = large_columns_map.get(col_idx, [])
        if not members:
            continue
        touching = sum(1 for b in members if touches_edge(b, img_w, img_h))
        if touching / len(members) >= threshold:
            skip.add(col_idx)
    return skip


def compute_iou_with_template(blob_roi, template, template_size=(200, 200)):
    if blob_roi is None or template is None:
        return 0.0
    resized = cv2.resize(blob_roi, template_size, interpolation=cv2.INTER_AREA)
    _, resized_bin = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    intersection = np.logical_and(resized_bin > 0, template > 0).sum()
    union = np.logical_or(resized_bin > 0, template > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def score_blob_against_reference(blob, ref, is_super=False, weights=None):
    if weights is None:
        weights = {"area": 0.30, "solidity": 0.20, "aspect": 0.15, "iou": 0.35}

    compute_solidity_if_missing(blob)

    area_ratio = blob["area"] / (ref["ref_area"] + 1e-6)
    area_penalty = min(abs(area_ratio - 1.0), 1.5)
    solidity_dev = abs(blob["solidity"] - ref["ref_solidity"])
    aspect_dev = (abs(blob["aspect_ratio"] - ref["ref_aspect"])
                  / (ref["ref_aspect"] + 1e-6))
    aspect_dev = min(aspect_dev, 1.5)
    iou = compute_iou_with_template(blob["roi"], ref["template"])
    iou_penalty = 1.0 - iou

    score = (weights["area"] * area_penalty
             + weights["solidity"] * solidity_dev
             + weights["aspect"] * aspect_dev
             + weights["iou"] * iou_penalty)

    reasons = []
    if area_ratio < 0.7:
        reasons.append(f"too small (area={area_ratio:.2f}× ref)")
    elif area_ratio > 1.3:
        reasons.append(f"too large (area={area_ratio:.2f}× ref)")
    if solidity_dev > 0.12:
        reasons.append(
            f"solidity off ({blob['solidity']:.2f} vs {ref['ref_solidity']:.2f})")
    if aspect_dev > 0.25:
        reasons.append(
            f"aspect off ({blob['aspect_ratio']:.2f} vs {ref['ref_aspect']:.2f})")
    if iou < 0.6:
        reasons.append(f"low shape match (IoU={iou:.2f})")

    component_mismatch = False
    if is_super and "ref_component_count" in ref:
        if blob["component_count"] != ref["ref_component_count"]:
            component_mismatch = True
            reasons.append(
                f"component count {blob['component_count']} "
                f"vs ref {ref['ref_component_count']}")

    if component_mismatch:
        verdict = "major"
    elif score >= 0.35:
        verdict = "major"
    elif score >= 0.15:
        verdict = "minor"
    else:
        verdict = "clean"

    return {
        "score": round(float(score), 3),
        "area_ratio": round(float(area_ratio), 3),
        "solidity_dev": round(float(solidity_dev), 3),
        "aspect_dev": round(float(aspect_dev), 3),
        "iou": round(float(iou), 3),
        "component_mismatch": component_mismatch,
        "verdict": verdict,
        "reasons": reasons,
    }


def detect_missing_letters(large_rows, large_anchors, large_refs,
                           skip_columns, img_w, img_h):
    missing_defects = []
    for row_idx, row in enumerate(large_rows):
        slope, intercept = fit_row_line(row)
        cols_present = set()
        for blob in row:
            c = blob.get("large_col", None)
            if c is not None:
                cols_present.add(c)

        for col_idx, ref in large_refs.items():
            if col_idx in skip_columns or col_idx in cols_present:
                continue

            anchor_x = large_anchors[col_idx]
            expected_y = slope * anchor_x + intercept
            w = ref["ref_width"]
            h = ref["ref_height"]
            x = int(anchor_x - w / 2)
            y = int(expected_y - h / 2)

            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            synthetic_blob = {
                "x": x, "y": y, "w": w, "h": h,
                "area": 0,
                "cx": x + w / 2, "cy": y + h / 2,
                "aspect_ratio": round(w / (h + 1e-6), 3),
                "roi": None,
                "is_synthetic_missing": True,
            }
            missing_defects.append({
                "score": 1.0,
                "verdict": "major",
                "reasons": [f"missing letter (column {col_idx} "
                            f"absent in row {row_idx})"],
                "blob": synthetic_blob,
                "type": "missing",
                "col": col_idx,
                "row": row_idx,
                "area_ratio": 0,
                "solidity_dev": 0,
                "aspect_dev": 0,
                "iou": 0,
                "component_mismatch": False,
            })
    return missing_defects


def score_all_blobs(blobs, clean_img, large_refs, super_refs, super_blobs,
                    large_blobs, large_rows, large_anchors):
    """Wrapper for Cell 12 — produces the list of defect results."""
    img_h, img_w = clean_img.shape[:2]
    large_columns_map, _ = group_large_blobs_into_columns(large_blobs, img_w)
    skip_columns = identify_edge_columns(
        large_refs, large_columns_map, img_w, img_h, threshold=0.5)

    defects = []

    # Score LARGE blobs
    for blob in blobs:
        if blob.get("group", -1) == -1:
            continue
        col = blob.get("large_col", None)
        if col is None or col not in large_refs:
            continue
        if col in skip_columns:
            continue
        ref = large_refs[col]
        if is_oversized_artifact(blob, ref, max_area_ratio=3.0):
            continue
        if touches_edge(blob, img_w, img_h):
            continue

        result = score_blob_against_reference(blob, ref, is_super=False)
        result["blob"] = blob
        result["type"] = "large"
        result["col"] = col
        defects.append(result)

    # Score INC super-blobs
    if super_refs and 0 in super_refs:
        super_ref = super_refs[0]
        for sb in super_blobs:
            if is_oversized_artifact(sb, super_ref, max_area_ratio=3.0):
                continue
            if touches_edge(sb, img_w, img_h):
                continue
            result = score_blob_against_reference(sb, super_ref, is_super=True)
            result["blob"] = sb
            result["type"] = "super"
            result["col"] = 0
            defects.append(result)

    # Detect missing letters
    missing = detect_missing_letters(
        large_rows, large_anchors, large_refs,
        skip_columns, img_w, img_h)
    defects.extend(missing)

    return defects


# ============================================================
# CELL 13 — Visualization
# ============================================================
def draw_defects_on_image(image_rgb, defect_results):
    viz = image_rgb.copy()
    COLOR_MINOR = (255, 200, 0)
    COLOR_MAJOR = (255, 0, 0)
    THICKNESS_MINOR = 3
    THICKNESS_MAJOR = 4

    for r in defect_results:
        if r["verdict"] == "clean":
            continue
        b = r["blob"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        if r["verdict"] == "major":
            color = COLOR_MAJOR
            thickness = THICKNESS_MAJOR
        else:
            color = COLOR_MINOR
            thickness = THICKNESS_MINOR
        cv2.rectangle(viz, (x, y), (x + w, y + h), color, thickness)
    return viz


# ============================================================
# Master orchestrator
# ============================================================
def run_full_pipeline(image_path, progress_callback=None):
    """
    Run the full pipeline on a single image. Returns a dict with all
    intermediate and final outputs ready for the Streamlit UI.
    """
    def step(msg):
        if progress_callback:
            progress_callback(msg)

    step("Loading image …")
    raw_rgb = load_image(image_path)

    step("Cropping to sheet …")
    cropped_rgb = crop_to_sheet(raw_rgb)

    step("Converting to grayscale …")
    gray = to_grayscale(cropped_rgb)

    step("Applying threshold …")
    binary, otsu_val = smart_threshold(gray)

    step("Auto-inverting …")
    corrected = ensure_white_on_black(binary)

    step("Denoising …")
    clean = denoise(corrected)
    clean = crop_border(clean, border_px=10)

    step("Detecting blobs …")
    blobs, blob_thresholds = find_blobs(clean)
    if not blobs:
        raise ValueError("No blobs detected. "
                         "Image may not contain a HARP/INC test sheet.")

    step("Detecting rows and columns …")
    groups, clean_blobs, anomalies = detect_rows_and_columns(blobs)

    step("Building reference templates …")
    (large_refs, super_refs, large_blobs, large_rows,
     super_blobs, large_anchors) = build_references(clean_blobs, clean)

    if not large_refs:
        raise ValueError("Could not build reference templates. "
                         "Image may not contain a HARP/INC test sheet "
                         "with enough repetitions.")

    step("Scoring blobs and detecting defects …")
    defect_results = score_all_blobs(
        clean_blobs, clean, large_refs, super_refs, super_blobs,
        large_blobs, large_rows, large_anchors)

    step("Drawing defect visualization …")
    annotated = draw_defects_on_image(cropped_rgb, defect_results)

    # Tally
    minor_count = sum(1 for r in defect_results if r["verdict"] == "minor")
    major_count = sum(1 for r in defect_results if r["verdict"] == "major")
    missing_count = sum(1 for r in defect_results if r["type"] == "missing")
    clean_count = sum(1 for r in defect_results if r["verdict"] == "clean")
    total_blobs_inspected = len(defect_results) - missing_count

    # Defects only for the table
    defects_only = [r for r in defect_results if r["verdict"] != "clean"]

    return {
        "cropped_image": cropped_rgb,
        "clean_mask": clean,
        "annotated_image": annotated,
        "defect_results": defect_results,
        "defects_only": defects_only,
        "large_refs": large_refs,
        "super_refs": super_refs,
        "tally": {
            "minor": minor_count,
            "major": major_count,
            "missing": missing_count,
            "clean": clean_count,
            "total_blobs_inspected": total_blobs_inspected,
            "total_defects": minor_count + major_count,
        },
        "row_anomalies": anomalies,
    }
