import argparse

import cv2
from custom_types import detect_line_segment


def _get_args():
    parser = argparse.ArgumentParser("test line segment detection")
    parser.add_argument("--input", "-i", type=str, help="path to input image", required=True)
    parser.add_argument("--num_bands", type=int, default=16)

    return parser.parse_args()


def main():
    args = _get_args()
    image = cv2.imread(args.input)
    assert image is not None, f"failed to load {args.input}"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    line_segment_detector = cv2.ximgproc.createFastLineDetector()

    line_segments = detect_line_segment(gray, line_segment_detector)
    for line_segment in line_segments:
        line_segment.draw_line(image)
        line_segment.draw_line_support_region(image)

    cv2.imwrite("detected_lines.jpg", image)


if __name__ == "__main__":
    main()
