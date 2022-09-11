from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np
from skimage.draw import line

__all__ = [
    "LineSegment",
    "detect_line_segment",
    "detect_and_compute_line_segment",
]


@dataclass
class LineSegment:
    start: np.ndarray
    end: np.ndarray
    lsr_width: int = 10  # line support region width

    def _validate(self):
        assert self.start.shape[0] == 2
        assert self.end.shape[0] == 2
        assert self.lsr_width > 0

    def __post_init__(self):
        self._validate()

    @cached_property
    def middle(self):
        return ((self.start + self.end) // 2).astype(np.uint32)

    @cached_property
    def length(self):
        return np.linalg.norm(self.end - self.start)

    @cached_property
    def angle(self):
        delta = self.end - self.start
        return np.arctan2(delta[1], delta[0])

    @cached_property
    def line_support_region(self):
        dx = self.lsr_width * np.sin(self.angle)
        dy = self.lsr_width * np.cos(self.angle)

        x1, y1 = self.start
        x2, y2 = self.end
        p1 = (int(x1 - dx), int(y1 + dy))
        p2 = (int(x1 + dx), int(y1 - dy))
        p3 = (int(x2 + dx), int(y2 - dy))
        p4 = (int(x2 - dx), int(y2 + dy))

        return np.array([p1, p2, p3, p4], dtype=np.int32)

    def draw_line(
        self,
        image: np.ndarray,
        color=(255, 0, 0),
        thickness=1,
    ):
        cv2.line(
            image,
            tuple(self.start),
            tuple(self.end),
            color=[int(e) for e in color],
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_line_support_region(
        self,
        image: np.ndarray,
        color=(0, 255, 0),
        thickness=2,
    ):
        cv2.polylines(
            image,
            [self.line_support_region.reshape((-1, 1, 2))],
            isClosed=True,
            color=color,
            thickness=thickness,
        )

    def estimate_lbd_descriptor(
        self,
        gray: np.ndarray,
        num_bands: int,
    ):
        band_descriptors = []
        for i in range(num_bands):
            offset = (i - num_bands / 2) * (self.length / num_bands)
            cx = self.start[0] + offset * np.cos(self.angle)
            cy = self.start[1] + offset * np.sin(self.angle)

            perp_angle = self.angle + np.pi / 2
            band_start = (
                int(cx - self.lsr_width / 2 * np.cos(perp_angle)),
                int(cy - self.lsr_width / 2 * np.sin(perp_angle)),
            )
            band_end = (
                int(cx + self.lsr_width / 2 * np.cos(perp_angle)),
                int(cy + self.lsr_width / 2 * np.sin(perp_angle)),
            )

            rr, cc = line(band_start[1], band_start[0], band_end[1], band_end[0])
            valid = (rr >= 0) & (rr < gray.shape[0]) & (cc >= 0) & (cc < gray.shape[1])
            intensities = gray[rr[valid], cc[valid]]

            if len(intensities) > 0:
                band_descriptors.extend([np.mean(intensities), np.std(intensities)])
            else:
                band_descriptors.extend([0, 0])

        band_descriptors = np.array(band_descriptors).reshape(1, -1)
        return cv2.normalize(band_descriptors, None, norm_type=cv2.NORM_L2).flatten()


def detect_line_segment(
    gray: np.ndarray,
    detector=cv2.ximgproc.createFastLineDetector(),
):
    cv2_lines = detector.detect(gray)

    def _get_segment(cv2_line):
        x0, y0, x1, y1 = cv2_line.flatten()
        return LineSegment(
            np.asarray((x0, y0)),
            np.asarray((x1, y1)),
        )

    return [_get_segment(_line) for _line in cv2_lines]


def detect_and_compute_line_segment(
    gray: np.ndarray,
    detector=cv2.ximgproc.createFastLineDetector(),
    num_bands=16,
):
    line_segments = detect_line_segment(
        gray,
        detector,
    )
    descriptors = [
        line_segment.estimate_lbd_descriptor(gray, num_bands).reshape(1, -1) for line_segment in line_segments
    ]
    return line_segments, np.concatenate(descriptors).astype(np.float32)
