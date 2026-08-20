"""Pixel-based heart control detection for mirrored dating apps."""

import math


# Normalized from the supplied Hinge control: a white heart cutout inside a
# solid black circular button. Keeping this as a compact binary signature makes
# detection independent of screen scale and avoids a fragile external asset.
HINGE_HEART_BUTTON_TEMPLATE = (
    "........########........",
    "......############......",
    ".....##############.....",
    "....################....",
    "...##################...",
    "..####################..",
    ".######################.",
    ".#######........#######.",
    ".######..........######.",
    "#######.########.#######",
    "#######.########..######",
    "#######.########.#######",
    "#######..######..#######",
    "#######...####...#######",
    "########...#############",
    "#########...############",
    ".#########...##########.",
    ".##########.###########.",
    "..####################..",
    "..####################..",
    "...##################...",
    "....################....",
    ".....##############.....",
    ".......##########.......",
)



class HeartIconDetector:
    """Find heart silhouettes without relying on a fixed screen position."""

    GRID_SIZE = 24

    def __init__(self):
        self.expected_fill, self.expected_boundary = self._expected_heart()

    @classmethod
    def _expected_heart(cls):
        raw_fill = set()
        raw_size = 96
        for row in range(raw_size):
            for column in range(raw_size):
                x = ((column + 0.5) / raw_size - 0.5) * 2.7
                y = 1.28 - (row + 0.5) / raw_size * 2.7
                if (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0:
                    raw_fill.add((column, row))

        raw_left = min(x for x, _ in raw_fill)
        raw_right = max(x for x, _ in raw_fill)
        raw_top = min(y for _, y in raw_fill)
        raw_bottom = max(y for _, y in raw_fill)
        size = cls.GRID_SIZE
        fill = {
            (
                round((x - raw_left) / (raw_right - raw_left) * (size - 1)),
                round((y - raw_top) / (raw_bottom - raw_top) * (size - 1)),
            )
            for x, y in raw_fill
        }

        boundary = {
            point
            for point in fill
            if any(
                (point[0] + dx, point[1] + dy) not in fill
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        }
        return fill, boundary

    @staticmethod
    def _near(point, targets, radius=1):
        x, y = point
        return any(
            (x + dx, y + dy) in targets
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        )

    def shape_score(self, points, bounds):
        """Score a connected component as either a filled or outlined heart."""
        left, top, width, height = bounds
        if not points or width <= 0 or height <= 0:
            return 0.0

        observed = set()
        size = self.GRID_SIZE
        for x, y in points:
            column = min(size - 1, int((x - left) / width * size))
            row = min(size - 1, int((y - top) / height * size))
            observed.add((column, row))

        intersection = len(observed & self.expected_fill)
        union = len(observed | self.expected_fill)
        filled_score = intersection / union if union else 0.0

        boundary_precision = sum(
            self._near(point, self.expected_boundary) for point in observed
        ) / len(observed)
        boundary_recall = sum(
            self._near(point, observed) for point in self.expected_boundary
        ) / len(self.expected_boundary)
        boundary_score = (
            2 * boundary_precision * boundary_recall
            / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall
            else 0.0
        )

        mirrored = {(size - 1 - x, y) for x, y in observed}
        symmetry_union = len(observed | mirrored)
        symmetry = len(observed & mirrored) / symmetry_union if symmetry_union else 0.0

        bottom = [x for x, y in observed if y >= size * 0.82]
        bottom_centered = bool(bottom) and abs(sum(bottom) / len(bottom) - (size - 1) / 2) < size * 0.16
        top_left = any(x < size * 0.45 and y < size * 0.42 for x, y in observed)
        top_right = any(x > size * 0.55 and y < size * 0.42 for x, y in observed)
        landmarks = (bottom_centered + top_left + top_right) / 3

        geometry = max(filled_score, boundary_score)
        return 0.72 * geometry + 0.18 * symmetry + 0.10 * landmarks

    @staticmethod
    def hinge_patch_score(frame, left, top, size):
        """Score a raw image patch, independent of connected dark regions."""
        matched_weight = 0.0
        total_weight = 0.0
        grid_size = len(HINGE_HEART_BUTTON_TEMPLATE)
        for row, template_row in enumerate(HINGE_HEART_BUTTON_TEMPLATE):
            for column, expected in enumerate(template_row):
                normalized_x = (column + 0.5) / grid_size - 0.5
                normalized_y = (row + 0.5) / grid_size - 0.5
                inside_circle = normalized_x ** 2 + normalized_y ** 2 < 0.19

                # Ignore the area outside the button because the underlying
                # profile image may be light or dark.
                if expected == "." and not inside_circle:
                    continue
                weight = 5.0 if expected == "." else 1.0
                x = left + (column + 0.5) / grid_size * size
                y = top + (row + 0.5) / grid_size * size
                observed_dark = HeartIconDetector._is_candidate_color(
                    frame.color_at(x, y),
                    "Hinge",
                )
                total_weight += weight
                if observed_dark == (expected == "#"):
                    matched_weight += weight
        return matched_weight / total_weight if total_weight else 0.0

    def find_hinge_template(
        self,
        frame,
        mask,
        mask_width,
        mask_height,
        x_start,
        y_start,
        expected_button_size,
    ):
        """Slide the Hinge signature over the right side when dark pixels merge with a photo."""
        integral_width = mask_width + 1
        integral = [0] * (integral_width * (mask_height + 1))
        for y in range(mask_height):
            running = 0
            source_row = y * mask_width
            integral_row = (y + 1) * integral_width
            previous_row = y * integral_width
            for x in range(mask_width):
                running += mask[source_row + x]
                integral[integral_row + x + 1] = integral[previous_row + x + 1] + running

        def dark_count(left, top, size):
            right, bottom = left + size, top + size
            return (
                integral[bottom * integral_width + right]
                - integral[top * integral_width + right]
                - integral[bottom * integral_width + left]
                + integral[top * integral_width + left]
            )

        best_score = 0.0
        best_center = None
        scan_left = max(0, int(frame.width * 0.68) - x_start)
        ring_angles = tuple(index * math.pi / 6 for index in range(12))
        internal_white_cells = []
        template_size = len(HINGE_HEART_BUTTON_TEMPLATE)
        for row, template_row in enumerate(HINGE_HEART_BUTTON_TEMPLATE):
            for column, expected in enumerate(template_row):
                nx = (column + 0.5) / template_size - 0.5
                ny = (row + 0.5) / template_size - 0.5
                if expected == "." and nx * nx + ny * ny < 0.19:
                    internal_white_cells.append((column, row))
        probe_step = max(1, len(internal_white_cells) // 12)
        white_probes = internal_white_cells[::probe_step][:12]

        if expected_button_size < 55:
            # Preserve support for 1x/external displays, where screenshots may
            # still contain a larger rendered phone surface.
            minimum_size, maximum_size = 28, 88
        else:
            minimum_size = max(24, round(expected_button_size * 0.82 / 4) * 4)
            maximum_size = round(expected_button_size * 1.18 / 4) * 4
        maximum_size = min(maximum_size, mask_width, mask_height)
        for size in range(minimum_size, maximum_size + 1, 4):
            radius = size * 0.39
            center_offset = size / 2
            for top in range(0, mask_height - size + 1, 4):
                for left in range(scan_left, mask_width - size + 1, 4):
                    density = dark_count(left, top, size) / (size * size)
                    if not 0.42 <= density <= 0.90:
                        continue

                    center_x = left + center_offset
                    center_y = top + center_offset
                    dark_ring_points = sum(
                        bool(mask[
                            min(mask_height - 1, max(0, round(center_y + math.sin(angle) * radius)))
                            * mask_width
                            + min(mask_width - 1, max(0, round(center_x + math.cos(angle) * radius)))
                        ])
                        for angle in ring_angles
                    )
                    if dark_ring_points < 9:
                        continue

                    global_left = x_start + left
                    global_top = y_start + top
                    light_probe_matches = sum(
                        not self._is_candidate_color(
                            frame.color_at(
                                global_left + (column + 0.5) / template_size * size,
                                global_top + (row + 0.5) / template_size * size,
                            ),
                            "Hinge",
                        )
                        for column, row in white_probes
                    )
                    if light_probe_matches < len(white_probes) * 0.58:
                        continue

                    score = self.hinge_patch_score(frame, global_left, global_top, size)
                    if score > best_score:
                        best_score = score
                        best_center = (
                            global_left + center_offset,
                            global_top + center_offset,
                        )

        return best_center, best_score

    @staticmethod
    def _is_candidate_color(color, platform):
        first, green, third = color
        if platform == "Tinder":
            # Tinder's heart is strongly green; Quartz may expose RGB or BGRA.
            return green > 85 and green - first > 22 and green - third > 10
        # Hinge's target is a dark circular button with a white heart cutout.
        return max(color) < 112 and max(color) - min(color) < 55

    def find_all(self, capture, platform):
        frame = capture.frame
        capture_width = frame.width
        capture_height = frame.height

        if platform == "Tinder":
            x_start, x_end = int(capture_width * 0.05), int(capture_width * 0.95)
            y_start, y_end = int(capture_height * 0.42), int(capture_height * 0.96)
            threshold = 0.66
        else:
            # Hinge places the action heart against the lower-right safe-area
            # edge on some profile layouts, so scan all the way to both edges.
            x_start, x_end = int(capture_width * 0.68), capture_width
            y_start, y_end = int(capture_height * 0.08), capture_height
            threshold = 0.84

        mask_width = x_end - x_start
        mask_height = y_end - y_start
        mask = bytearray(mask_width * mask_height)
        for local_y in range(mask_height):
            row_offset = local_y * mask_width
            for local_x in range(mask_width):
                capture_x = x_start + local_x
                capture_y = y_start + local_y
                # Detect at native capture resolution. Downsampling a Retina
                # frame can disconnect Hinge's thin white heart stroke.
                if self._is_candidate_color(frame.color_at(capture_x, capture_y), platform):
                    mask[row_offset + local_x] = 1

        candidates = []
        if platform == "Tinder":
            for local_y in range(mask_height):
                for local_x in range(mask_width):
                    start_index = local_y * mask_width + local_x
                    if not mask[start_index]:
                        continue

                    stack = [(local_x, local_y)]
                    mask[start_index] = 0
                    points = []
                    min_x = max_x = local_x
                    min_y = max_y = local_y

                    while stack:
                        x, y = stack.pop()
                        points.append((x, y))
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        for dx, dy in (
                            (-1, -1), (0, -1), (1, -1),
                            (-1, 0), (1, 0),
                            (-1, 1), (0, 1), (1, 1),
                        ):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < mask_width and 0 <= ny < mask_height:
                                index = ny * mask_width + nx
                                if mask[index]:
                                    mask[index] = 0
                                    stack.append((nx, ny))

                    width = max_x - min_x + 1
                    height = max_y - min_y + 1
                    aspect = width / height
                    if (
                        width < 15 or height < 15
                        or width > 180 or height > 180
                        or not 0.62 <= aspect <= 1.48
                        or len(points) < width + height
                    ):
                        continue

                    score = self.shape_score(points, (min_x, min_y, width, height))
                    if score < threshold:
                        continue

                    center_x = x_start + min_x + width / 2
                    center_y = y_start + min_y + height / 2
                    # Prefer the familiar lower-center action row when shapes tie.
                    score += max(0, 0.08 - abs(center_x / capture_width - 0.5) * 0.08)
                    candidates.append((score, center_x, center_y, None))
        else:
            template_center, template_score = self.find_hinge_template(
                frame,
                mask,
                mask_width,
                mask_height,
                x_start,
                y_start,
                36 * capture.scale_x,
            )
            if template_center is not None and template_score >= threshold:
                candidates.append((
                    template_score,
                    template_center[0],
                    template_center[1],
                    None,
                ))

        if not candidates:
            return []
        candidates.sort(reverse=True)
        results = []
        for score, center_x, center_y, _ in candidates:
            point = capture.desktop_point(center_x, center_y)
            if any(
                abs(point[0] - current_point[0]) < 8
                and abs(point[1] - current_point[1]) < 8
                for current_point, _ in results
            ):
                continue
            results.append((point, min(score, 1.0)))
        return results

    def find(self, capture, platform):
        candidates = self.find_all(capture, platform)
        return candidates[0] if candidates else (None, 0.0)
