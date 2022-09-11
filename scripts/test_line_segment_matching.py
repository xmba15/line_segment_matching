import argparse
from typing import List

import cv2
import numpy as np
from custom_types import LineSegment, detect_and_compute_line_segment


def _get_args():
    parser = argparse.ArgumentParser("test line segment detection and matching")
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--ref_path", required=True)
    parser.add_argument("--num_bands", type=int, default=16)
    parser.add_argument("--num_draw", type=int, default=20)

    return parser.parse_args()


def generate_random_colors(
    num_samples: int,
    seed=42,
):
    assert num_samples >= 0
    if num_samples == 0:
        return []

    np.random.seed(seed)

    b_values = np.random.randint(0, 256, size=num_samples, dtype=np.uint8)
    g_values = np.random.randint(0, 256, size=num_samples, dtype=np.uint8)
    r_values = np.random.randint(0, 256, size=num_samples, dtype=np.uint8)

    return list(zip(b_values, g_values, r_values))


def draw_matched_line_segments(  # pylint: disable=R0913
    query_image: np.ndarray,
    ref_image: np.ndarray,
    query_segments: List[LineSegment],
    ref_segments: List[LineSegment],
    matches: List[cv2.DMatch],
    thickness: int = 2,
):
    query_height, query_width = query_image.shape[:2]
    ref_height, ref_width = query_image.shape[:2]
    output = np.zeros((max(query_height, ref_height), query_width + ref_width, 3), dtype=np.uint8)

    colors = generate_random_colors(len(matches))

    for _color, _match in zip(colors, matches):
        query_segments[_match.queryIdx].draw_line(
            query_image,
            color=_color,
            thickness=thickness,
        )
        ref_segments[_match.trainIdx].draw_line(
            ref_image,
            color=_color,
            thickness=thickness,
        )

    output[:query_height, :query_width, :] = query_image
    output[:ref_height, query_width : (query_width + ref_width), :] = ref_image  # noqa: E203

    for _color, _match in zip(colors, matches):
        cv2.line(
            output,
            tuple(query_segments[_match.queryIdx].middle),
            tuple(ref_segments[_match.trainIdx].middle + np.array((query_width, 0))),
            color=[int(e) for e in _color],
            thickness=thickness,
        )

    return output


def main():
    args = _get_args()
    image_paths = [args.query_path, args.ref_path]
    images = [cv2.imread(input_path) for input_path in [args.query_path, args.ref_path]]
    for image, image_path in zip(images, image_paths):
        assert image is not None, f"failed to load {image_path}"
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    line_segment_detector = cv2.ximgproc.createFastLineDetector()

    query_segments, query_descs = detect_and_compute_line_segment(
        grays[0],
        line_segment_detector,
        args.num_bands,
    )

    ref_segments, ref_descs = detect_and_compute_line_segment(
        grays[1],
        line_segment_detector,
        args.num_bands,
    )

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(query_descs, ref_descs)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_image = draw_matched_line_segments(
        images[0],
        images[1],
        query_segments,
        ref_segments,
        matches[: args.num_draw],
    )
    cv2.imwrite("matched_lines.jpg", matched_image)


if __name__ == "__main__":
    main()
