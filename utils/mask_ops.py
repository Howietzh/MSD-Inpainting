import json
import random
from math import comb
from pathlib import Path

import cv2
import numpy as np
import torch


def build_random_box_mask(mask, num_boxes: int = 30):
    batch_size, _, height, width = mask.shape
    random_mask = torch.zeros_like(mask)

    for batch_idx in range(batch_size):
        for _ in range(num_boxes):
            box_h = torch.randint(max(1, height // 32), max(2, height // 10), (1,), device=mask.device).item()
            box_w = torch.randint(max(1, width // 32), max(2, width // 10), (1,), device=mask.device).item()
            top = torch.randint(0, max(1, height - box_h + 1), (1,), device=mask.device).item()
            left = torch.randint(0, max(1, width - box_w + 1), (1,), device=mask.device).item()
            random_mask[batch_idx, :, top:top + box_h, left:left + box_w] = 1.0

    return torch.clamp(random_mask, 0.0, 1.0)


class DefectMaskEngine:
    def __init__(self, train_dir: Path, cache_file: Path, target_size: int = 512):
        self.shape = (target_size, target_size)
        self.train_dir = train_dir
        self.cache_file = cache_file
        self.stats_cache = {}

    def _compute_single_defect_stats(self, defect_token):
        metadata_path = self.train_dir / "metadata.jsonl"
        lengths, thicknesses = [], []

        if not metadata_path.exists():
            return 50, 150, 5, 15

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("defect_token") == defect_token and "defect_mask_path" in item:
                    mask_path = self.train_dir / item["defect_mask_path"]
                    if not mask_path.exists():
                        continue

                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and cv2.countNonZero(mask) > 0:
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area < 4:
                                continue

                            _, _, w, h = cv2.boundingRect(cnt)
                            length = max(w, h)

                            if "particle" in defect_token:
                                thicknesses.append(max(2, int(np.sqrt(area / np.pi))))
                                lengths.append(length)
                            else:
                                thicknesses.append(max(2, int(area / length)) if length > 0 else 2)
                                lengths.append(length)

        if not lengths:
            return 50, 150, 5, 15

        min_l, max_l = int(np.percentile(lengths, 10)), int(np.percentile(lengths, 90))
        min_t, max_t = int(np.percentile(thicknesses, 10)), int(np.percentile(thicknesses, 90))

        print(
            f"\n📈 [{defect_token}] 尺寸统计: 实例数 {len(lengths)} | "
            f"Length [{min_l}, {max(min_l + 10, max_l)}] | Thick/Rad [{min_t}, {max(min_t + 2, max_t)}]"
        )
        return min_l, max(min_l + 10, max_l), min_t, max(min_t + 2, max_t)

    def load_or_compute_stats(self, tasks):
        unique_defects = {t["defect"] for t in tasks}

        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.stats_cache = json.load(f)
            if not [token for token in unique_defects if token not in self.stats_cache]:
                return

        for token in unique_defects:
            if token not in self.stats_cache:
                self.stats_cache[token] = self._compute_single_defect_stats(token)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.stats_cache, f, indent=4, ensure_ascii=False)

    def generate_dynamic_mask(self, comp_mask_np, defect_token):
        min_l, max_l, min_t, max_t = self.stats_cache.get(defect_token, (50, 150, 5, 15))
        defect_mask = None

        if "crack" in defect_token or "tear" in defect_token:
            defect_mask = self._generate_flexible_printed_circuit_tear(
                comp_mask_np, length=random.randint(min_l, max_l), width=random.randint(min_t, max_t)
            )
        elif "scratch" in defect_token:
            defect_mask = self._generate_scratch(
                comp_mask_np, length=random.randint(min_l, max_l), thickness=random.randint(min_t, max_t)
            )
        elif "particle" in defect_token:
            defect_mask = self._generate_particle(
                comp_mask_np, radius=random.randint(min_t, max_t), count=random.randint(1, 3)
            )

        if defect_mask is None or cv2.countNonZero(defect_mask) == 0:
            defect_mask = np.zeros(self.shape, dtype=np.uint8)
            ys, xs = np.where(comp_mask_np > 0)
            if len(ys) > 0:
                idx = random.randint(0, len(ys) - 1)
                cv2.circle(defect_mask, (xs[idx], ys[idx]), max(3, min_t), 255, -1)

        return defect_mask

    def _bezier_curve(self, points, num_points=100):
        n, t = len(points) - 1, np.linspace(0.0, 1.0, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            curve += np.outer(comb(n, i) * (1 - t) ** (n - i) * t ** i, points[i])
        return curve.astype(np.float32)

    def _draw_variable_thickness_poly(self, points, max_thickness, roughness=0.3):
        defect = np.zeros(self.shape, dtype=np.uint8)
        if len(points) < 3:
            return defect
        pts_l, pts_r = [], []
        for i in range(len(points)):
            variable_t = max_thickness * (
                np.sin(np.pi * (i / (len(points) - 1))) * (1 - roughness) + roughness * random.random()
            )
            tangent = points[i + 1] - points[i] if i < len(points) - 1 else points[i] - points[i - 1]
            norm = np.array([-tangent[1], tangent[0]], dtype=np.float32)
            norm_unit = norm / np.linalg.norm(norm) if np.linalg.norm(norm) > 0 else np.array([1, 0])
            pts_l.append(points[i] + norm_unit * (variable_t / 2))
            pts_r.append(points[i] - norm_unit * (variable_t / 2))
        cv2.fillPoly(defect, [np.concatenate([pts_l, pts_r[::-1]], axis=0).astype(np.int32)], color=255)
        return defect

    def _generate_scratch(self, comp_mask, length=200, thickness=5, margin=10, curvature=50):
        dist_map = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        safe_ys, safe_xs = np.where(dist_map > (margin + thickness))
        if len(safe_xs) == 0:
            return np.zeros(self.shape, dtype=np.uint8)

        weights = dist_map[safe_ys, safe_xs]
        idx = (
            np.random.choice(len(safe_xs), p=weights / np.sum(weights))
            if np.sum(weights) > 0
            else random.randint(0, len(safe_xs) - 1)
        )
        p0 = [safe_xs[idx], safe_ys[idx]]
        ang = random.uniform(0, 2 * np.pi)
        p2 = [p0[0] + length * np.cos(ang), p0[1] + length * np.sin(ang)]
        p1 = [
            (p0[0] + p2[0]) / 2 + random.randint(-int(curvature), int(curvature)),
            (p0[1] + p2[1]) / 2 + random.randint(-int(curvature), int(curvature)),
        ]

        return cv2.bitwise_and(
            self._draw_variable_thickness_poly(self._bezier_curve([p0, p1, p2]), thickness),
            comp_mask,
        )

    def _generate_particle(self, comp_mask, radius=10, count=3):
        defect = np.zeros(self.shape, dtype=np.uint8)
        dist_map = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        safe_ys, safe_xs = np.where(dist_map >= radius)
        if len(safe_xs) == 0:
            return defect

        weights = dist_map[safe_ys, safe_xs]
        probs = (weights / np.sum(weights)) if np.sum(weights) > 0 else None

        for _ in range(int(count)):
            idx = np.random.choice(len(safe_xs), p=probs) if probs is not None else random.randint(0, len(safe_xs) - 1)
            cx, cy, base_r = safe_xs[idx], safe_ys[idx], max(2, int(radius) + random.randint(-2, 2))
            pts = [
                [cx + base_r * random.uniform(0.7, 1.3) * np.cos(a), cy + base_r * random.uniform(0.7, 1.3) * np.sin(a)]
                for a in np.linspace(0, 2 * np.pi, random.randint(6, 12), endpoint=False)
            ]
            temp_mask = np.zeros_like(defect)
            cv2.fillPoly(temp_mask, [np.array(pts, np.int32).reshape((-1, 1, 2))], 255)
            defect = np.maximum(defect, temp_mask)

        return cv2.bitwise_and(defect, comp_mask)

    def _generate_flexible_printed_circuit_tear(self, comp_mask, length=120, width=25):
        defect_mask = np.zeros(self.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return defect_mask

        box = np.int32(cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))))
        d01, d12 = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        edge = random.choice([(box[0], box[1]), (box[2], box[3])] if d01 < d12 else [(box[1], box[2]), (box[3], box[0])])

        p_start = np.float32(edge[0]) + random.uniform(0.2, 0.8) * (np.float32(edge[1]) - np.float32(edge[0]))
        _, _, _, max_loc = cv2.minMaxLoc(cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5))
        inward_vec = np.array(max_loc, dtype=np.float32) - p_start
        inward_unit = inward_vec / np.linalg.norm(inward_vec) if np.linalg.norm(inward_vec) > 0 else np.array([0, 1], dtype=np.float32)
        p_end, path_pts = p_start + inward_unit * length, np.linspace(p_start, p_start + inward_unit * length, 60)

        pts_l, pts_r, norm_vec = [], [], np.array([-inward_unit[1], inward_unit[0]], dtype=np.float32)
        for i in range(60):
            offset = norm_vec * (width * (1 - i / 59) ** 2 / 2 * (1 + random.uniform(-0.4, 0.4)))
            pts_l.append(path_pts[i] + offset)
            pts_r.append(path_pts[i] - offset)

        cv2.fillPoly(
            defect_mask,
            [np.concatenate([pts_l[:-3], [p_end], pts_r[:-3][::-1]], axis=0).astype(np.int32)],
            color=255,
        )
        return cv2.bitwise_and(defect_mask, comp_mask)
