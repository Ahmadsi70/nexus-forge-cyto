"""
Nexus-Forge Cyto — Classifier Validation Script
================================================
این اسکریپت دقت classifier پروژه را روی داده‌های واقعی اندازه می‌گیرد.

داده‌ی مورد استفاده:
  CoNSeP (Colon Nuclear Segmentation and Phenotypes)
  منبع: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/
  برچسب‌های موجود:
    1 = Other/Misc    → Normal
    2 = Inflammatory  → Normal (Immune)
    3 = Healthy Epi   → Normal
    4 = Malignant Epi → Malignant ✓
    5 = Fibroblast    → Normal
    6 = Muscle        → Normal
    7 = Endothelial   → Normal

اگر CoNSeP موجود نبود، اسکریپت از داده‌ی مصنوعی مبتنی بر
ادبیات پاتولوژی استفاده می‌کند تا ساختار خروجی را نشان دهد.

اجرا:
  python validate_classifier.py [--consep-dir PATH_TO_CONSEP]
  python validate_classifier.py --synthetic-only
"""

from __future__ import annotations

import argparse
import math
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# مسیرهای پروژه
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICES  = _REPO_ROOT / "services"
if str(_SERVICES) not in sys.path:
    sys.path.insert(0, str(_SERVICES))

# ─────────────────────────────────────────────────────────────────────────────
# کپی وزن‌های classifier از classifier.rs (Single Source of Truth)
# اگر وزن‌ها در Rust تغییر کرد، اینجا هم باید آپدیت شود!
# ─────────────────────────────────────────────────────────────────────────────
W_AREA_PLEOMORPHISM  = 0.17
W_IRREGULAR_MEMBRANE = 0.21
W_RBF_MICROENV       = 0.21
W_PACKING_CLUSTER    = 0.17
W_ECCENTRICITY       = 0.24

MALIGNANCY_THRESHOLD  = 0.35   # کالیبره‌شده در 2026-06 — قبلاً 0.65 بود (اشتباه)
AREA_RATIO_ONSET      = 1.08
AREA_RATIO_SATURATES  = 1.75
CIRCULARITY_SOFT_CAP  = 0.75
RBF_RESPONSE_GAIN     = 2.8
PACKING_GAIN          = 1.15

assert abs(W_AREA_PLEOMORPHISM + W_IRREGULAR_MEMBRANE + W_RBF_MICROENV
           + W_PACKING_CLUSTER + W_ECCENTRICITY - 1.0) < 1e-9, \
    "وزن‌ها باید جمع‌شان ۱ شود!"


# ─────────────────────────────────────────────────────────────────────────────
# فرمول‌های هندسی (بازنویسی Python از spatial.rs)
# ─────────────────────────────────────────────────────────────────────────────

def polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula — مطابق spatial.rs::polygon_area_shoelace_abs"""
    n = len(vertices)
    if n < 3:
        return 0.0
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(vertices: np.ndarray) -> float:
    """مجموع طول یال‌های بسته — مطابق spatial.rs::polygon_perimeter"""
    n = len(vertices)
    if n < 2:
        return 0.0
    diffs = np.diff(np.vstack([vertices, vertices[0]]), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def polygon_circularity(area: float, perimeter: float) -> float:
    """C = 4πA / P² — مطابق spatial.rs::polygon_circularity"""
    if perimeter < 1e-18 or area < 0:
        return 0.0
    c = (4.0 * math.pi * area) / (perimeter ** 2)
    return min(c, 1.0)


def polygon_eccentricity(ring32: np.ndarray) -> float:
    """کشیدگی از کوواریانس ۳۲ نقطه — مطابق spatial.rs::polygon_eccentricity"""
    if len(ring32) < 2:
        return 0.0
    pts = ring32.astype(np.float64)
    cov = np.cov(pts.T)
    if cov.ndim < 2:
        return 0.0
    trace = cov[0, 0] + cov[1, 1]
    det   = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    disc  = trace ** 2 - 4 * det
    if disc < 0 or not math.isfinite(disc):
        return 0.0
    lam1 = 0.5 * (trace + math.sqrt(disc))
    lam2 = 0.5 * (trace - math.sqrt(disc))
    if lam1 < 1e-18:
        return 0.0
    ratio = max(0.0, min(1.0, lam2 / lam1))
    inner = 1.0 - ratio
    return math.sqrt(inner) if inner > 0 else 0.0


def ring32_from_polygon(xy_list: list[tuple[float, float]]) -> np.ndarray:
    """نمونه‌برداری یکنواخت از polygon به ۳۲ نقطه — مطابق coord_adapter.py"""
    pts = np.array(xy_list, dtype=np.float64)
    n = len(pts)
    if n < 3:
        # دایره پیش‌فرض با شعاع ۴
        cx, cy = (pts[:, 0].mean(), pts[:, 1].mean()) if n > 0 else (0.0, 0.0)
        t = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        return np.stack([cx + 4 * np.cos(t), cy + 4 * np.sin(t)], axis=1)
    # محیط تجمعی برای نمونه‌برداری یکنواخت
    closed = np.vstack([pts, pts[0]])
    diffs  = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    if total < 1e-12:
        return np.tile(pts[0], (32, 1))
    sample_t = np.linspace(0, total, 32, endpoint=False)
    xs = np.interp(sample_t, cum_len, closed[:, 0])
    ys = np.interp(sample_t, cum_len, closed[:, 1])
    return np.stack([xs, ys], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# تابع امتیازدهی (بازنویسی Python از classifier.rs)
# ─────────────────────────────────────────────────────────────────────────────

class CellFeatures(NamedTuple):
    area:          float   # نسبت به میانه اسلاید
    circularity:   float   # [0, 1]
    rbf_risk_score: float  # نرمال‌شده
    knn_density:   float   # نسبت به میانه اسلاید
    eccentricity:  float   # [0, 1]


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))


def malignancy_score(f: CellFeatures) -> float:
    """امتیاز بدخیمی [0,1] — مطابق classifier.rs::malignancy_score"""
    # area component
    if f.area <= AREA_RATIO_ONSET:
        s_area = 0.0
    else:
        span   = max(AREA_RATIO_SATURATES - AREA_RATIO_ONSET, 1e-9)
        s_area = _clamp01((f.area - AREA_RATIO_ONSET) / span)

    # circularity component
    c = max(0.0, min(1.0, f.circularity))
    s_circ = _clamp01((CIRCULARITY_SOFT_CAP - c) / CIRCULARITY_SOFT_CAP) if c < CIRCULARITY_SOFT_CAP else 0.0

    # RBF component
    s_rbf = _clamp01(f.rbf_risk_score * RBF_RESPONSE_GAIN)

    # packing component
    s_pack = _clamp01((1.0 - f.knn_density) * PACKING_GAIN) if f.knn_density < 1.0 else 0.0

    # eccentricity component
    s_ecc = _clamp01(f.eccentricity)

    score = (W_AREA_PLEOMORPHISM  * s_area
           + W_IRREGULAR_MEMBRANE * s_circ
           + W_RBF_MICROENV       * s_rbf
           + W_PACKING_CLUSTER    * s_pack
           + W_ECCENTRICITY       * s_ecc)
    return _clamp01(score)


def predict_malignant(f: CellFeatures) -> bool:
    return malignancy_score(f) > MALIGNANCY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# محاسبه ویژگی‌ها برای یک سلول
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_for_cell(
    polygon_xy:       list[tuple[float, float]],
    median_area:      float,
    median_knn:       float,
    all_centroids:    np.ndarray,
    rbf_sigma:        float,
) -> CellFeatures:
    ring32    = ring32_from_polygon(polygon_xy)
    verts_f64 = np.array(polygon_xy, dtype=np.float64)
    area      = polygon_area(verts_f64)
    perim     = polygon_perimeter(verts_f64)
    circ      = polygon_circularity(area, perim)
    ecc       = polygon_eccentricity(ring32)
    cx, cy    = ring32[:, 0].mean(), ring32[:, 1].mean()

    # kNN density (k=8)
    if len(all_centroids) > 1:
        dists = np.linalg.norm(all_centroids - np.array([cx, cy]), axis=1)
        dists = np.sort(dists[dists > 1e-9])
        k     = min(8, len(dists))
        mean_knn = float(dists[:k].mean()) if k > 0 else 0.0
    else:
        mean_knn = 0.0

    # RBF field
    dists_all = np.linalg.norm(all_centroids - np.array([cx, cy]), axis=1)
    dists_all = dists_all[dists_all > 1e-9]
    rbf_raw   = float(np.sum(np.exp(-(dists_all ** 2) / (2 * rbf_sigma ** 2)))) if len(dists_all) else 0.0
    n_cells   = max(1, len(all_centroids) - 1)
    rbf_norm  = rbf_raw / n_cells

    area_ratio = area / max(median_area, 1e-9)
    knn_ratio  = mean_knn / max(median_knn, 1e-9)

    return CellFeatures(
        area=area_ratio,
        circularity=circ,
        rbf_risk_score=rbf_norm,
        knn_density=knn_ratio,
        eccentricity=ecc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# بارگذاری داده‌های CoNSeP
# ─────────────────────────────────────────────────────────────────────────────

def load_consep(consep_dir: Path) -> list[dict]:
    """
    بارگذاری CoNSeP از پوشه‌ی:
      consep_dir/
        Test/
          Labels/  ← فایل‌های .npy با برچسب‌های سلولی
          Images/
    یا فرمت JSON ساده‌شده.
    """
    records = []
    # جستجوی فایل‌های JSON اگر از قبل convert شده
    json_files = list(consep_dir.rglob("*.json"))
    if json_files:
        for jf in json_files[:50]:  # حداکثر ۵۰ فایل
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if "cells" in data:
                    records.extend(data["cells"])
            except Exception:
                continue
        if records:
            print(f"  ✅ {len(records)} سلول از {len(json_files)} فایل JSON بارگذاری شد")
            return records

    # جستجوی فایل‌های .mat یا .npy
    try:
        import scipy.io as sio
        mat_files = list(consep_dir.rglob("*.mat"))
        for mf in mat_files[:20]:
            try:
                mat = sio.loadmat(str(mf))
                # CoNSeP format: inst_map, type_map, inst_type, inst_centroid
                if "inst_type" in mat and "inst_centroid" in mat:
                    types     = mat["inst_type"].flatten()
                    centroids = mat["inst_centroid"]
                    for i, (t, c) in enumerate(zip(types, centroids)):
                        records.append({
                            "type": int(t),
                            "centroid": [float(c[0]), float(c[1])],
                            "source": mf.name,
                        })
            except Exception:
                continue
        if records:
            print(f"  ✅ {len(records)} سلول از {len(mat_files)} فایل .mat بارگذاری شد")
            return records
    except ImportError:
        pass

    return records


# ─────────────────────────────────────────────────────────────────────────────
# تولید داده‌ی مصنوعی مبتنی بر ادبیات پاتولوژی
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_normal: int = 300, n_malignant: int = 300,
                                seed: int = 42) -> list[dict]:
    """
    تولید سلول‌های مصنوعی بر اساس آمار منتشرشده در:
      - Gurcan et al. (2009) Histopathological image analysis
      - Veta et al. (2014) Breast cancer histology
      - CoNSeP paper (Graham et al. 2019)

    سلول‌های نرمال:
      - مساحت: ≈ ۵۰–۱۵۰ px² (دایره‌وار)
      - دایرویی: ≈ ۰.۸۵–۰.۹۸
      - کشیدگی: ≈ ۰.۰۵–۰.۲۵

    سلول‌های بدخیم:
      - مساحت: ≈ ۱۵۰–۴۰۰ px² (بزرگ‌تر، پلئومورف)
      - دایرویی: ≈ ۰.۳۰–۰.۶۵ (نامنظم)
      - کشیدگی: ≈ ۰.۵۰–۰.۹۵ (کشیده)
    """
    rng = np.random.default_rng(seed)
    cells = []

    # ── سلول‌های نرمال ──────────────────────────────────────────────────────
    for i in range(n_normal):
        cx = rng.uniform(0, 1000)
        cy = rng.uniform(0, 1000)
        r  = rng.uniform(4.0, 8.0)
        n_pts = 32
        noise_r = rng.uniform(0, r * 0.08, n_pts)
        t = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        xs = cx + (r + noise_r) * np.cos(t)
        ys = cy + (r + noise_r) * np.sin(t)
        polygon = list(zip(xs.tolist(), ys.tolist()))
        cells.append({"polygon": polygon, "centroid": [cx, cy], "label": 0, "type_name": "Normal"})

    # ── سلول‌های بدخیم ──────────────────────────────────────────────────────
    # خوشه‌های تومور (سلول‌های بدخیم به هم نزدیک‌اند)
    n_clusters = 5
    cluster_centers = rng.uniform(100, 900, (n_clusters, 2))
    for i in range(n_malignant):
        cc = cluster_centers[i % n_clusters]
        cx = cc[0] + rng.normal(0, 25)
        cy = cc[1] + rng.normal(0, 25)
        r_major = rng.uniform(7.0, 15.0)   # بزرگ‌تر
        r_minor = r_major * rng.uniform(0.3, 0.75)  # کشیده
        angle   = rng.uniform(0, math.pi)
        n_pts   = 32
        noise_r = rng.uniform(0, r_major * 0.22, n_pts)
        t  = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        xr = r_major * np.cos(t) + noise_r * np.cos(t)
        yr = r_minor * np.sin(t) + noise_r * np.sin(t)
        # چرخش
        xs = cx + xr * math.cos(angle) - yr * math.sin(angle)
        ys = cy + xr * math.sin(angle) + yr * math.cos(angle)
        polygon = list(zip(xs.tolist(), ys.tolist()))
        cells.append({"polygon": polygon, "centroid": [cx, cy], "label": 1, "type_name": "Malignant"})

    rng.shuffle(cells)
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# اجرای validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(cells: list[dict], dataset_name: str) -> dict:
    """اجرای pipeline روی سلول‌ها و محاسبه‌ی معیارهای ارزیابی"""
    print(f"\n{'='*60}")
    print(f"  📊 Validation روی: {dataset_name}")
    print(f"  تعداد سلول: {len(cells)}")
    print(f"{'='*60}")

    # ── استخراج مختصات مرکز برای محاسبه‌ی kNN و RBF ──────────────────────
    centroids = np.array([[c["centroid"][0], c["centroid"][1]] for c in cells])

    # ── محاسبه‌ی میانه‌ی اسلاید ──────────────────────────────────────────────
    areas = []
    for cell in cells:
        poly = np.array(cell["polygon"], dtype=np.float64)
        areas.append(polygon_area(poly))
    median_area = float(np.median([a for a in areas if a > 0]) or 1.0)

    # kNN برای همه سلول‌ها
    knn_means = []
    for i, cell in enumerate(cells):
        cx, cy = cell["centroid"]
        dists = np.linalg.norm(centroids - np.array([cx, cy]), axis=1)
        dists = np.sort(dists[dists > 1e-9])
        k = min(8, len(dists))
        knn_means.append(float(dists[:k].mean()) if k > 0 else 0.0)
    median_knn = float(np.median([d for d in knn_means if d > 0]) or 1.0)
    rbf_sigma  = median_knn

    print(f"  میانه‌ی مساحت اسلاید:  {median_area:.1f} px²")
    print(f"  میانه‌ی kNN (σ):       {median_knn:.1f} px")

    # ── پیش‌بینی برای هر سلول ────────────────────────────────────────────────
    y_true, y_pred, scores = [], [], []
    for cell in cells:
        feats = compute_features_for_cell(
            polygon_xy    = cell["polygon"],
            median_area   = median_area,
            median_knn    = median_knn,
            all_centroids = centroids,
            rbf_sigma     = rbf_sigma,
        )
        score  = malignancy_score(feats)
        pred   = int(score > MALIGNANCY_THRESHOLD)
        truth  = int(cell["label"])
        y_true.append(truth)
        y_pred.append(pred)
        scores.append(score)

    # ── محاسبه‌ی معیارها ──────────────────────────────────────────────────────
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy   = (tp + tn) / len(y_true)

    # AUC تقریبی (trapezoidal)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        # محاسبه‌ی دستی AUC با روش ذوزنقه
        thresholds = np.linspace(0, 1, 101)
        tprs, fprs = [], []
        for thr in thresholds:
            yp = (scores > thr).astype(int)
            _tp = np.sum((yp == 1) & (y_true == 1))
            _fp = np.sum((yp == 1) & (y_true == 0))
            _fn = np.sum((yp == 0) & (y_true == 1))
            _tn = np.sum((yp == 0) & (y_true == 0))
            tprs.append(_tp / (_tp + _fn + 1e-9))
            fprs.append(_fp / (_fp + _tn + 1e-9))
        tprs = np.array(tprs[::-1])
        fprs = np.array(fprs[::-1])
        auc  = float(np.trapz(tprs, fprs))

    # ── نمایش نتایج ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  📋 Confusion Matrix:")
    print(f"  {'─'*50}")
    print(f"                   پیش‌بینی Normal   پیش‌بینی Malignant")
    print(f"  واقعی Normal        TN={tn:>5}          FP={fp:>5}")
    print(f"  واقعی Malignant     FN={fn:>5}          TP={tp:>5}")
    print(f"  {'─'*50}")
    print(f"\n  📈 معیارهای ارزیابی:")
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │  Accuracy    : {accuracy:.4f}  ({accuracy*100:.1f}%)  │")
    print(f"  │  Precision   : {precision:.4f}  ({precision*100:.1f}%)  │")
    print(f"  │  Recall      : {recall:.4f}  ({recall*100:.1f}%)  │")
    print(f"  │  Specificity : {specificity:.4f}  ({specificity*100:.1f}%)  │")
    print(f"  │  F1-Score    : {f1:.4f}  ({f1*100:.1f}%)  │")
    print(f"  │  AUC-ROC     : {auc:.4f}               │")
    print(f"  └─────────────────────────────────┘")

    # ── تفسیر نتایج ──────────────────────────────────────────────────────────
    print(f"\n  🔍 تفسیر:")
    if f1 >= 0.85:
        grade = "🟢 عالی"
        msg   = "classifier به خوبی کار می‌کند. آماده برای انتشار."
    elif f1 >= 0.70:
        grade = "🟡 قابل قبول"
        msg   = "نیاز به بهبود وزن‌ها با داده‌ی بیشتر دارد."
    elif f1 >= 0.55:
        grade = "🟠 ضعیف"
        msg   = "وزن‌ها باید با machine learning بهینه شوند."
    else:
        grade = "🔴 ناموفق"
        msg   = "classifier نیاز به بازطراحی اساسی دارد."

    print(f"  نتیجه: {grade}")
    print(f"  پیام:  {msg}")

    # ── توزیع امتیازها ──────────────────────────────────────────────────────
    mal_scores = scores[y_true == 1]
    nor_scores = scores[y_true == 0]
    print(f"\n  📊 توزیع امتیاز بدخیمی:")
    print(f"  سلول‌های واقعاً Malignant → میانه: {np.median(mal_scores):.3f}, "
          f"میانگین: {np.mean(mal_scores):.3f}")
    print(f"  سلول‌های واقعاً Normal    → میانه: {np.median(nor_scores):.3f}, "
          f"میانگین: {np.mean(nor_scores):.3f}")
    sep = np.median(mal_scores) - np.median(nor_scores)
    print(f"  جدایی‌پذیری (median gap): {sep:.3f} "
          f"{'✅ خوب' if sep > 0.2 else '⚠️ کم'}")

    return {
        "dataset":     dataset_name,
        "n_cells":     len(cells),
        "accuracy":    round(accuracy, 4),
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "specificity": round(specificity, 4),
        "f1":          round(f1, 4),
        "auc":         round(auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Nexus-Forge Cyto — Classifier Validation"
    )
    parser.add_argument(
        "--consep-dir", type=Path, default=None,
        help="مسیر پوشه‌ی CoNSeP (اختیاری). اگر داده نداشت، از داده مصنوعی استفاده می‌کند."
    )
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="فقط از داده‌ی مصنوعی استفاده کن"
    )
    parser.add_argument(
        "--n-cells", type=int, default=300,
        help="تعداد سلول در هر کلاس برای داده مصنوعی (پیش‌فرض: 300)"
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="ذخیره نتایج در JSON"
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  🔬 Nexus-Forge Cyto — Classifier Validation")
    print("═" * 60)
    print(f"\n  وزن‌های classifier:")
    print(f"    Area Pleomorphism  : {W_AREA_PLEOMORPHISM}")
    print(f"    Irregular Membrane : {W_IRREGULAR_MEMBRANE}")
    print(f"    RBF Microenv       : {W_RBF_MICROENV}")
    print(f"    Packing/Cluster    : {W_PACKING_CLUSTER}")
    print(f"    Eccentricity       : {W_ECCENTRICITY}")
    print(f"    آستانه تصمیم      : {MALIGNANCY_THRESHOLD}")

    all_results = []

    # ── داده‌ی CoNSeP ─────────────────────────────────────────────────────────
    if not args.synthetic_only and args.consep_dir and args.consep_dir.exists():
        print(f"\n  🔍 بارگذاری داده‌های CoNSeP از: {args.consep_dir}")
        raw = load_consep(args.consep_dir)
        if raw:
            # نقشه‌برداری برچسب CoNSeP به Malignant/Normal
            # نوع ۴ (Malignant Epithelial) → 1, بقیه → 0
            cells = []
            for r in raw:
                t = r.get("type", 0)
                label = 1 if t == 4 else 0
                if "polygon" in r:
                    cells.append({"polygon": r["polygon"],
                                  "centroid": r.get("centroid", [0, 0]),
                                  "label": label,
                                  "type_name": "Malignant" if label else "Normal"})
                elif "centroid" in r:
                    cx, cy = r["centroid"]
                    r_px = 5.0
                    t_arr = np.linspace(0, 2 * math.pi, 32, endpoint=False)
                    poly = list(zip(
                        (cx + r_px * np.cos(t_arr)).tolist(),
                        (cy + r_px * np.sin(t_arr)).tolist()
                    ))
                    cells.append({"polygon": poly,
                                  "centroid": [cx, cy],
                                  "label": label,
                                  "type_name": "Malignant" if label else "Normal"})

            n_mal = sum(1 for c in cells if c["label"] == 1)
            n_nor = sum(1 for c in cells if c["label"] == 0)
            print(f"  تعداد Malignant: {n_mal}, Normal: {n_nor}")
            if cells and n_mal > 0 and n_nor > 0:
                result = run_validation(cells, "CoNSeP (Real Data)")
                all_results.append(result)
        else:
            print("  ⚠️  داده‌ی CoNSeP یافت نشد — به داده‌ی مصنوعی رجوع می‌شود.")

    # ── داده‌ی مصنوعی ─────────────────────────────────────────────────────────
    print(f"\n  🧪 تولید داده‌ی مصنوعی...")
    print(f"     {args.n_cells} سلول نرمال + {args.n_cells} سلول بدخیم")
    print(f"     (بر اساس آمار منتشرشده در ادبیات پاتولوژی)")
    synthetic_cells = generate_synthetic_dataset(
        n_normal=args.n_cells, n_malignant=args.n_cells
    )
    result = run_validation(synthetic_cells, "Synthetic (Literature-Based)")
    all_results.append(result)

    # ── ذخیره نتایج ──────────────────────────────────────────────────────────
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"validation_results": all_results}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n  💾 نتایج ذخیره شد: {args.output_json}")

    print(f"\n{'═'*60}")
    print("  ✅ Validation کامل شد")
    print(f"{'═'*60}\n")

    return all_results


if __name__ == "__main__":
    main()
