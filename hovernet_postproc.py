#!/usr/bin/env python3
"""Standalone HoVer-Net post-processing (ported from TIAToolbox, no tiatoolbox dependency).

Turns raw head outputs (np, hv, tp) into instance map + per-instance info dict.
Dependencies: numpy, scipy, scikit-image, opencv-python.
"""
import math
import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed


def _get_bounding_box(img):
    ys, xs = np.where(img)
    if len(ys) == 0:
        return np.array([0, 0, 0, 0])
    return np.array([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])


def _proc_np_hv(np_map, hv_map, scale_factor=1):
    """Extract nuclei instances using the NP and HV maps (HoVer watershed)."""
    blb_raw = np_map[..., 0]
    h_dir_raw = hv_map[..., 0]
    v_dir_raw = hv_map[..., 1]

    blb = np.array(blb_raw >= 0.3, dtype=np.int32)
    blb = ndimage.label(blb)[0]
    blb = remove_small_objects(blb, max_size=9)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    ksize = int((20 * scale_factor) + 1)
    obj_size = math.ceil(10 * (scale_factor ** 2))

    sobel_h = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_v = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_h = 1 - cv2.normalize(sobel_h, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    sobel_v = 1 - cv2.normalize(sobel_v, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    overall = np.maximum(sobel_h, sobel_v)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)
    marker = blb - overall
    marker[marker < 0] = 0
    marker = ndimage.binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndimage.label(marker)[0]
    marker = remove_small_objects(marker, max_size=obj_size - 1)

    return watershed(dist, markers=marker, mask=blb)


def get_instance_info(pred_inst, pred_type=None):
    """Collect per-instance bounding box, centroid, contour, type and prob."""
    inst_id_list = np.unique(pred_inst)[1:]  # exclude background
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id
        inst_box = _get_bounding_box(inst_map)
        inst_box_tl = inst_box[:2]
        inst_map = inst_map[inst_box[1]:inst_box[3], inst_box[0]:inst_box[2]]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        contours, _ = cv2.findContours(inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        inst_contour = np.squeeze(contours[0].astype(np.int32))
        if inst_contour.ndim != 2 or inst_contour.shape[0] < 3:
            continue
        inst_centroid = np.array([inst_moment["m10"] / inst_moment["m00"],
                                  inst_moment["m01"] / inst_moment["m00"]])
        inst_contour = inst_contour + inst_box_tl[None]
        inst_centroid = inst_centroid + inst_box_tl
        inst_info_dict[int(inst_id)] = {
            "box": inst_box,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "prob": None,
            "type": None,
        }

    if pred_type is not None:
        for inst_id in list(inst_info_dict.keys()):
            c_min, r_min, c_max, r_max = inst_info_dict[inst_id]["box"]
            inst_map_crop = pred_inst[r_min:r_max, c_min:c_max] == inst_id
            inst_type_crop = pred_type[r_min:r_max, c_min:c_max]
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = sorted(zip(type_list, type_pixels), key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0 and len(type_list) > 1:
                inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["prob"] = float(type_prob)

    return inst_info_dict


def postproc(np_map, hv_map, tp_map):
    """Full post-processing: np/hv/tp (numpy) -> (instance_map, instance_info_dict)."""
    tp_map = np.around(tp_map).astype("uint8")
    pred_inst = _proc_np_hv(np_map, hv_map)
    nuc_inst_info_dict = get_instance_info(pred_inst, tp_map)
    return pred_inst, nuc_inst_info_dict
