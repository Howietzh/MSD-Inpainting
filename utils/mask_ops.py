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
        self._reference_masks = None

    def _load_reference_masks(self):
        if self._reference_masks is not None:
            return

        self._reference_masks = {}
        metadata_path = self.train_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Reference-mask ablation requires training metadata: {metadata_path}"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                defect_token = item.get("defect_token")
                component_token = item.get("object_token")
                relative_path = item.get("defect_mask_path")
                if not defect_token or not component_token or not relative_path:
                    continue
                mask_path = self.train_dir / relative_path
                if mask_path.exists():
                    self._reference_masks.setdefault(
                        (defect_token, component_token), []
                    ).append(mask_path)

    def generate_reference_elastic_mask(
        self,
        comp_mask_np,
        defect_token,
        component_token,
        alpha: float = 10.0,
        sigma: float = 4.0,
    ):
        """Warp a real training mask for the w/o-CDME inference ablation."""
        if alpha < 0 or sigma <= 0:
            raise ValueError("Elastic deformation requires alpha >= 0 and sigma > 0.")

        self._load_reference_masks()
        candidates = self._reference_masks.get((defect_token, component_token), [])
        if not candidates:
            raise KeyError(
                f"No reference masks for {defect_token} on {component_token}."
            )

        mask_path = random.choice(candidates)
        reference = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise ValueError(f"Cannot read reference defect mask: {mask_path}")
        reference = cv2.resize(
            reference,
            (self.shape[1], self.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        reference = (reference > 0).astype(np.uint8) * 255

        height, width = reference.shape
        dx = cv2.GaussianBlur(
            np.random.uniform(-1.0, 1.0, (height, width)).astype(np.float32),
            (0, 0),
            sigmaX=sigma,
        ) * alpha
        dy = cv2.GaussianBlur(
            np.random.uniform(-1.0, 1.0, (height, width)).astype(np.float32),
            (0, 0),
            sigmaX=sigma,
        ) * alpha
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        warped = cv2.remap(
            reference,
            grid_x + dx,
            grid_y + dy,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        component_binary = (comp_mask_np > 0).astype(np.uint8) * 255
        return cv2.bitwise_and(warped, component_binary)

    def _build_default_stats(self, defect_token):
        if "particle" in defect_token:
            return {
                "kind": "particle",
                "radius": {"p10": 3, "p90": 8},
                "count": {"p10": 1, "p90": 3},
            }

        if "crack" in defect_token or "tear" in defect_token:
            return {
                "kind": "tear",
                "length": {"p10": 50, "p90": 150},
                "width": {"p10": 5, "p90": 15},
            }

            return {
                "kind": "scratch",
                "length": {"p10": 50, "p90": 150},
                "thickness": {"p10": 5, "p90": 15},
            }

    def _is_structured_stats(self, stats):
        return isinstance(stats, dict) and "kind" in stats

    def get_defect_kind(self, defect_token):
        defect_stats = self.stats_cache.get(defect_token, self._build_default_stats(defect_token))
        return defect_stats["kind"]

    def _percentile_range(self, values, minimum, minimum_gap=0):
        if not values:
            return {"p10": minimum, "p90": max(minimum, minimum + minimum_gap)}

        p10 = int(np.percentile(values, 10))
        p90 = int(np.percentile(values, 90))
        p10 = max(minimum, p10)
        p90 = max(p10 + minimum_gap, p90)
        return {"p10": p10, "p90": p90}

    def _extract_valid_components(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        components = []

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < 4:
                continue

            component_mask = np.zeros_like(mask, dtype=np.uint8)
            component_mask[labels == label_idx] = 255
            components.append(component_mask)

        return components

    def _skeletonize(self, mask):
        skeleton = np.zeros_like(mask, dtype=np.uint8)
        work_mask = (mask > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while cv2.countNonZero(work_mask) > 0:
            eroded = cv2.erode(work_mask, kernel)
            temp = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work_mask, temp))
            work_mask = eroded

        return skeleton

    def _measure_skeleton_length_and_widths(self, component_mask):
        skeleton = self._skeletonize(component_mask)
        skeleton_coords = np.column_stack(np.where(skeleton > 0))
        if len(skeleton_coords) == 0:
            return 0.0, []

        dist_map = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        local_widths = [float(2.0 * dist_map[y, x]) for y, x in skeleton_coords]
        length = float(len(skeleton_coords))
        return length, local_widths

    def _measure_projected_length(self, component_mask):
        ys, xs = np.where(component_mask > 0)
        if len(xs) < 2:
            return 0.0

        points = np.stack([xs, ys], axis=1).astype(np.float32)
        center = points.mean(axis=0, keepdims=True)
        centered = points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal_axis = vh[0]
        projections = centered @ principal_axis
        return float(projections.max() - projections.min())

    def _print_stats_summary(self, defect_token, stats, sample_count):
        kind = stats["kind"]
        if kind == "scratch":
            print(
                f"\n📈 [{defect_token}] Scratch 统计: 实例数 {sample_count} | "
                f"PathLength [{stats['length']['p10']}, {stats['length']['p90']}] | "
                f"LineWidth [{stats['thickness']['p10']}, {stats['thickness']['p90']}]"
            )
        elif kind == "tear":
            print(
                f"\n📈 [{defect_token}] Tear/Crack 统计: 实例数 {sample_count} | "
                f"PenetrationLength [{stats['length']['p10']}, {stats['length']['p90']}] | "
                f"RootWidth [{stats['width']['p10']}, {stats['width']['p90']}]"
            )
        else:
            print(
                f"\n📈 [{defect_token}] Particle 统计: 图像数 {sample_count} | "
                f"Radius [{stats['radius']['p10']}, {stats['radius']['p90']}] | "
                f"Count [{stats['count']['p10']}, {stats['count']['p90']}]"
            )

    def _compute_single_defect_stats(self, defect_token):
        metadata_path = self.train_dir / "metadata.jsonl"
        if not metadata_path.exists():
            return self._build_default_stats(defect_token)

        if "particle" in defect_token:
            radii, counts = [], []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get("defect_token") != defect_token or "defect_mask_path" not in item:
                        continue

                    mask_path = self.train_dir / item["defect_mask_path"]
                    if not mask_path.exists():
                        continue

                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None or cv2.countNonZero(mask) == 0:
                        continue

                    components = self._extract_valid_components(mask)
                    if not components:
                        continue

                    counts.append(len(components))
                    for component_mask in components:
                        area = cv2.countNonZero(component_mask)
                        radii.append(max(2.0, float(np.sqrt(area / np.pi))))

            stats = {
                "kind": "particle",
                "radius": self._percentile_range(radii, minimum=2, minimum_gap=1),
                "count": self._percentile_range(counts, minimum=1, minimum_gap=0),
            }
            self._print_stats_summary(defect_token, stats, sample_count=len(counts))
            return stats

        lengths, secondary_values = [], []
        is_tear_like = ("crack" in defect_token or "tear" in defect_token)

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("defect_token") != defect_token or "defect_mask_path" not in item:
                    continue

                mask_path = self.train_dir / item["defect_mask_path"]
                if not mask_path.exists():
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or cv2.countNonZero(mask) == 0:
                    continue

                for component_mask in self._extract_valid_components(mask):
                    if is_tear_like:
                        length = self._measure_projected_length(component_mask)
                        _, local_widths = self._measure_skeleton_length_and_widths(component_mask)
                        if length <= 0 or not local_widths:
                            continue
                        lengths.append(length)
                        secondary_values.append(float(np.percentile(local_widths, 75)))
                    else:
                        length, local_widths = self._measure_skeleton_length_and_widths(component_mask)
                        if length <= 0 or not local_widths:
                            continue
                        lengths.append(length)
                        secondary_values.append(float(np.median(local_widths)))

        if not lengths:
            return self._build_default_stats(defect_token)

        if is_tear_like:
            stats = {
                "kind": "tear",
                "length": self._percentile_range(lengths, minimum=5, minimum_gap=10),
                "width": self._percentile_range(secondary_values, minimum=2, minimum_gap=2),
            }
        else:
            stats = {
                "kind": "scratch",
                "length": self._percentile_range(lengths, minimum=5, minimum_gap=10),
                "thickness": self._percentile_range(secondary_values, minimum=2, minimum_gap=2),
            }

        self._print_stats_summary(defect_token, stats, sample_count=len(lengths))
        return stats

    def load_or_compute_stats(self, tasks):
        unique_defects = {t["defect"] for t in tasks}

        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                self.stats_cache = json.load(f)
            missing_or_legacy = [
                token for token in unique_defects
                if token not in self.stats_cache or not self._is_structured_stats(self.stats_cache[token])
            ]
            if not missing_or_legacy:
                return

        for token in unique_defects:
            if token not in self.stats_cache or not self._is_structured_stats(self.stats_cache[token]):
                self.stats_cache[token] = self._compute_single_defect_stats(token)

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.stats_cache, f, indent=4, ensure_ascii=False)

    def _sample_stat(self, defect_stats, field_name, minimum):
        field_stats = defect_stats[field_name]
        low = max(minimum, int(field_stats["p10"]))
        high = max(low, int(field_stats["p90"]))
        return random.randint(low, high)

    def get_default_param_values(self, defect_token):
        defect_stats = self.stats_cache.get(defect_token, self._build_default_stats(defect_token))
        kind = defect_stats["kind"]

        if kind == "scratch":
            return {
                "length": int((defect_stats["length"]["p10"] + defect_stats["length"]["p90"]) / 2),
                "thickness": int((defect_stats["thickness"]["p10"] + defect_stats["thickness"]["p90"]) / 2),
            }
        if kind == "tear":
            return {
                "length": int((defect_stats["length"]["p10"] + defect_stats["length"]["p90"]) / 2),
                "width": int((defect_stats["width"]["p10"] + defect_stats["width"]["p90"]) / 2),
            }
        return {
            "radius": int((defect_stats["radius"]["p10"] + defect_stats["radius"]["p90"]) / 2),
            "count": int((defect_stats["count"]["p10"] + defect_stats["count"]["p90"]) / 2),
        }

    def sample_generation_params(self, defect_token):
        defect_stats = self.stats_cache.get(defect_token, self._build_default_stats(defect_token))
        kind = defect_stats["kind"]

        if kind == "tear":
            return {
                "length": self._sample_stat(defect_stats, "length", minimum=5),
                "width": self._sample_stat(defect_stats, "width", minimum=2),
            }
        if kind == "scratch":
            return {
                "length": self._sample_stat(defect_stats, "length", minimum=5),
                "thickness": self._sample_stat(defect_stats, "thickness", minimum=2),
            }
        return {
            "radius": self._sample_stat(defect_stats, "radius", minimum=2),
            "count": self._sample_stat(defect_stats, "count", minimum=1),
        }

    def generate_dynamic_mask_with_params(self, comp_mask_np, defect_token, params, return_details=False):
        defect_stats = self.stats_cache.get(defect_token, self._build_default_stats(defect_token))
        kind = defect_stats["kind"]
        defect_mask = None
        details = {"kind": kind}

        if kind == "tear":
            defect_mask = self._generate_flexible_printed_circuit_tear(
                comp_mask_np,
                length=int(params["length"]),
                width=int(params["width"]),
            )
        elif kind == "scratch":
            defect_mask, scratch_details = self._generate_scratch(
                comp_mask_np,
                length=int(params["length"]),
                thickness=int(params["thickness"]),
                curvature_min=float(params.get("curvature_min", 0.03)),
                curvature_max=float(params.get("curvature_max", 0.65)),
            )
            details.update(scratch_details)
        elif kind == "particle":
            defect_mask = self._generate_particle(
                comp_mask_np,
                radius=int(params["radius"]),
                count=int(params["count"]),
            )

        if defect_mask is None or cv2.countNonZero(defect_mask) == 0:
            defect_mask = np.zeros(self.shape, dtype=np.uint8)
            ys, xs = np.where(comp_mask_np > 0)
            if len(ys) > 0:
                idx = random.randint(0, len(ys) - 1)
                if kind == "particle":
                    fallback_radius = max(2, int(params.get("radius", 2)))
                elif kind == "scratch":
                    fallback_radius = max(2, int(params.get("thickness", 2)))
                else:
                    fallback_radius = max(2, int(params.get("width", 2)))
                cv2.circle(defect_mask, (xs[idx], ys[idx]), max(3, fallback_radius), 255, -1)
                if kind == "scratch":
                    fallback_length = float(max(3, fallback_radius) * 2)
                    details.update({
                        "requested_length": float(params.get("length", 0)),
                        "actual_length": fallback_length,
                        "actual_arc_length": fallback_length,
                        "length_control_mode": "endpoint_distance",
                        "degraded": True,
                        "fallback": True,
                    })

        if return_details:
            return defect_mask, details
        return defect_mask

    def generate_dynamic_mask(self, comp_mask_np, defect_token):
        params = self.sample_generation_params(defect_token)
        return self.generate_dynamic_mask_with_params(comp_mask_np, defect_token, params)

    def _bezier_curve(self, points, num_points=100):
        n, t = len(points) - 1, np.linspace(0.0, 1.0, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            curve += np.outer(comb(n, i) * (1 - t) ** (n - i) * t ** i, points[i])
        return curve.astype(np.float32)

    def _compute_polyline_arc_length(self, points):
        if len(points) < 2:
            return 0.0
        deltas = np.diff(points.astype(np.float32), axis=0)
        return float(np.linalg.norm(deltas, axis=1).sum())

    def _compute_bezier_arc_length(self, control_points, num_points=120):
        curve = self._bezier_curve(control_points, num_points=num_points)
        return self._compute_polyline_arc_length(curve), curve

    def _is_curve_inside_component(self, curve_points, comp_mask, thickness, dist_map=None):
        if dist_map is None:
            dist_map = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

        clearance = max(1.5, float(thickness) / 2.0 + 1.0)
        height, width = comp_mask.shape[:2]
        for point in curve_points:
            x = int(round(point[0]))
            y = int(round(point[1]))
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
            if comp_mask[y, x] <= 0:
                return False
            if dist_map[y, x] < clearance:
                return False
        return True

    def _sample_scratch_start_point(self, dist_map, thickness):
        clearance = max(2.0, float(thickness) / 2.0 + 2.0)
        safe_ys, safe_xs = np.where(dist_map >= clearance)
        if len(safe_xs) == 0:
            return None

        weights = dist_map[safe_ys, safe_xs].astype(np.float64)
        probs = (weights / weights.sum()) if weights.sum() > 0 else None
        idx0 = np.random.choice(len(safe_xs), p=probs) if probs is not None else random.randint(0, len(safe_xs) - 1)
        return np.array([safe_xs[idx0], safe_ys[idx0]], dtype=np.float32)

    def _scan_scratch_directions_from_point(
        self,
        p0,
        comp_mask,
        dist_map,
        thickness,
        num_directions=32,
        step_size=1.0,
    ):
        clearance = max(2.0, float(thickness) / 2.0 + 2.0)
        height, width = comp_mask.shape[:2]
        direction_results = []

        for angle in np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            last_valid_point = p0.copy()
            traveled = 0.0

            while True:
                candidate = p0 + direction * (traveled + step_size)
                x = int(round(candidate[0]))
                y = int(round(candidate[1]))
                if x < 0 or x >= width or y < 0 or y >= height:
                    break
                if comp_mask[y, x] <= 0 or dist_map[y, x] < clearance:
                    break
                traveled += step_size
                last_valid_point = candidate.copy()

            if traveled >= max(4.0, float(thickness) * 1.5):
                direction_results.append(
                    {
                        "direction": direction,
                        "max_length": float(traveled),
                        "endpoint": last_valid_point.astype(np.float32),
                    }
                )

        return direction_results

    def _select_scratch_endpoint(self, p0, target_length, direction_results):
        if not direction_results:
            return None

        target_length = float(target_length)
        exact_candidates = [item for item in direction_results if item["max_length"] >= target_length]
        degraded = len(exact_candidates) == 0
        candidates = exact_candidates if exact_candidates else direction_results
        chosen = random.choice(candidates)

        chord_target = min(chosen["max_length"], target_length)
        p2 = p0 + chosen["direction"] * chord_target
        return p2.astype(np.float32), chosen, degraded

    def _build_scratch_curve_with_random_curvature(
        self,
        p0,
        p2,
        comp_mask,
        thickness,
        dist_map,
        min_curvature_ratio=0.03,
        max_curvature_ratio=0.65,
        max_curve_attempts=14,
    ):
        chord = float(np.linalg.norm(p2 - p0))
        if chord <= 1e-6:
            return None

        midpoint = (p0 + p2) / 2.0
        tangent = p2 - p0
        perp = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        perp_norm = np.linalg.norm(perp)
        if perp_norm <= 1e-6:
            return None
        perp /= perp_norm
        curvature_candidates = [0.0]
        curvature_candidates.extend(
            random.uniform(float(min_curvature_ratio), float(max_curvature_ratio))
            for _ in range(max_curve_attempts)
        )
        random.shuffle(curvature_candidates)

        inside_candidates = []
        for curvature_ratio in curvature_candidates:
            signs = [random.choice([-1, 1])] if curvature_ratio == 0.0 else random.sample([-1, 1], k=2)
            for curvature_sign in signs:
                offset = float(curvature_ratio) * chord
                p1 = midpoint + perp * (float(curvature_sign) * offset)
                arc_length, curve = self._compute_bezier_arc_length([p0, p1, p2])
                if not self._is_curve_inside_component(curve, comp_mask, thickness, dist_map=dist_map):
                    continue
                inside_candidates.append(
                    {
                        "control_point": p1,
                        "curve": curve,
                        "arc_length": arc_length,
                        "curvature_sign": int(curvature_sign),
                        "curvature_ratio": float(curvature_ratio),
                    }
                )

        if not inside_candidates:
            return None

        curved_candidates = [candidate for candidate in inside_candidates if candidate["curvature_ratio"] > 0.0]
        pool = curved_candidates if curved_candidates else inside_candidates
        return random.choice(pool)

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

    def _generate_scratch(
        self,
        comp_mask,
        length=200,
        thickness=5,
        margin=10,
        curvature=50,
        curvature_min=0.03,
        curvature_max=0.65,
    ):
        dist_map = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        target_length = max(5.0, float(length))
        thickness = max(1, int(thickness))
        curvature_min = max(0.0, float(curvature_min))
        curvature_max = max(curvature_min, float(curvature_max))
        best_candidate = None
        max_attempts = 32

        for _ in range(max_attempts):
            p0 = self._sample_scratch_start_point(dist_map, thickness=thickness)
            if p0 is None:
                break

            direction_results = self._scan_scratch_directions_from_point(
                p0=p0,
                comp_mask=comp_mask,
                dist_map=dist_map,
                thickness=thickness,
                num_directions=32,
                step_size=1.0,
            )
            endpoint_result = self._select_scratch_endpoint(p0, target_length, direction_results)
            if endpoint_result is None:
                continue

            p2, direction_info, degraded_length = endpoint_result
            candidate = self._build_scratch_curve_with_random_curvature(
                p0=p0,
                p2=p2,
                comp_mask=comp_mask,
                thickness=thickness,
                dist_map=dist_map,
                min_curvature_ratio=curvature_min,
                max_curvature_ratio=curvature_max,
                max_curve_attempts=14,
            )
            if candidate is None:
                continue

            candidate["direction_angle_deg"] = float(
                np.degrees(np.arctan2(direction_info["direction"][1], direction_info["direction"][0]))
            )
            candidate["chord_length"] = float(np.linalg.norm(p2 - p0))
            candidate["degraded"] = bool(degraded_length)

            if best_candidate is None:
                best_candidate = candidate
            elif best_candidate["degraded"] and not candidate["degraded"]:
                best_candidate = candidate
            elif best_candidate["degraded"] == candidate["degraded"] and candidate["curvature_ratio"] > best_candidate["curvature_ratio"]:
                best_candidate = candidate

            if not candidate["degraded"]:
                break

        if best_candidate is None:
            return np.zeros(self.shape, dtype=np.uint8), {
                "requested_length": target_length,
                "actual_length": 0.0,
                "actual_arc_length": 0.0,
                "length_control_mode": "endpoint_distance",
                "degraded": True,
                "fallback": True,
            }

        defect_mask = cv2.bitwise_and(
            self._draw_variable_thickness_poly(best_candidate["curve"], thickness),
            comp_mask,
        )
        start_point = best_candidate["curve"][0]
        end_point = best_candidate["curve"][-1]
        details = {
            "requested_length": target_length,
            "actual_length": float(best_candidate["chord_length"]),
            "actual_arc_length": float(best_candidate["arc_length"]),
            "length_control_mode": "endpoint_distance",
            "degraded": bool(best_candidate["degraded"]),
            "fallback": False,
            "start_point": [float(start_point[0]), float(start_point[1])],
            "end_point": [float(end_point[0]), float(end_point[1])],
            "curvature_sign": int(best_candidate["curvature_sign"]),
            "curvature_ratio": float(best_candidate["curvature_ratio"]),
            "requested_curvature_min": float(curvature_min),
            "requested_curvature_max": float(curvature_max),
            "direction_angle_deg": float(best_candidate["direction_angle_deg"]),
            "chord_length": float(best_candidate["chord_length"]),
        }
        return defect_mask, details

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
