import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.tracker import FeatureTracker
from src.vision import preprocess_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Pyramidal Lucas-Kanade object tracker")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output-video", default="outputs/annotated.mp4", help="Path for annotated output video")
    parser.add_argument("--output-csv", default="outputs/metrics.csv", help="Path for CSV metrics export")
    parser.add_argument("--output-plot", default="outputs/speed.png", help="Path for speed plot")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="Initial ROI (x y w h)")
    parser.add_argument("--meters-per-pixel", type=float, default=None, help="Direct pixel-to-meter scale")
    parser.add_argument("--reference-length-m", type=float, default=None, help="Known length of a reference object (meters)")
    parser.add_argument("--reference-pixels", type=float, default=None, help="Length of the reference object in pixels")
    parser.add_argument("--max-corners", type=int, default=300)
    parser.add_argument("--quality-level", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--lk-window", type=int, default=9)
    parser.add_argument("--lk-levels", type=int, default=3)
    parser.add_argument("--lk-iters", type=int, default=10)
    parser.add_argument("--lk-epsilon", type=float, default=0.01)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--ransac-iters", type=int, default=100)
    parser.add_argument("--min-features", type=int, default=80)
    parser.add_argument("--detect-interval", type=int, default=10)
    parser.add_argument("--intensity-thresh", type=float, default=None, help="Only detect features on pixels above this grayscale intensity (0-255)")
    parser.add_argument("--intensity-blur", type=int, default=3, help="Gaussian blur size before intensity thresholding (odd int, 0 to disable)")
    parser.add_argument("--auto-intensity-percentile", type=float, default=None, help="Compute intensity threshold automatically from this percentile (e.g., 99.5)")
    parser.add_argument("--exclude-rect", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="Region to exclude from tracking/detection (e.g., watermark box)")
    parser.add_argument("--bright-spot", action="store_true", help="Enable brightest-spot tracking (good for tiny bright balls)")
    parser.add_argument("--bright-spot-radius", type=float, default=60, help="Search radius around last center for bright spot (pixels)")
    parser.add_argument("--bright-spot-blur", type=int, default=3, help="Blur before bright-spot search (odd int, 0 to disable)")
    parser.add_argument("--bright-spot-percentile", type=float, default=99.5, help="Percentile threshold for bright-spot blob extraction")
    parser.add_argument("--bright-min-area", type=int, default=1, help="Minimum blob area for bright-spot selection")
    parser.add_argument("--bright-max-area", type=int, default=120, help="Maximum blob area for bright-spot selection (filter out lines)")
    parser.add_argument("--bright-max-aspect", type=float, default=3.0, help="Reject bright blobs with aspect ratio above this (filters line-like shapes)")
    parser.add_argument("--bright-dist-weight", type=float, default=1.0, help="Distance penalty weight for bright-spot scoring")
    parser.add_argument("--bright-max-jump", type=float, default=60.0, help="Reject bright-spot jumps larger than this (pixels)")
    parser.add_argument("--bright-min-fill", type=float, default=0.2, help="Minimum fill ratio (area/(w*h)) for bright spots")
    parser.add_argument("--bright-max-dim", type=float, default=120, help="Maximum width/height for bright blobs")
    return parser.parse_args()


def ensure_output_paths(args):
    for path_str in [args.output_video, args.output_csv, args.output_plot]:
        if path_str:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def default_roi(frame_shape):
    h, w = frame_shape[:2]
    return (
        w * 0.25,
        h * 0.25,
        w * 0.5,
        h * 0.5,
    )


def compute_scale(args):
    if args.meters_per_pixel is not None:
        return args.meters_per_pixel
    if args.reference_length_m is not None and args.reference_pixels is not None and args.reference_pixels > 0:
        return args.reference_length_m / args.reference_pixels
    return 1.0


def annotate(frame, roi, points, trajectory, center, speed_m_s, fps):
    vis = frame.copy()

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

    for pt in points:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

    if len(trajectory) > 1:
        pts = np.int32(trajectory)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)

    if center is not None:
        cv2.circle(vis, (int(center[0]), int(center[1])), 4, (255, 0, 0), -1)

    if speed_m_s is not None and fps > 0:
        text = f"Speed: {speed_m_s:.2f} m/s"
        cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return vis


def run():
    args = parse_args()
    ensure_output_paths(args)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0
    dt = 1.0 / fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame from video.")

    prev_gray = preprocess_frame(frame)
    roi = tuple(args.roi) if args.roi else default_roi(frame.shape)

    # Compute scale (pixels -> meters)
    scale = compute_scale(args)

    tracker = FeatureTracker(
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
        block_size=args.block_size,
        lk_window=args.lk_window,
        lk_levels=args.lk_levels,
        lk_iters=args.lk_iters,
        lk_epsilon=args.lk_epsilon,
        ransac_thresh=args.ransac_thresh,
        ransac_iters=args.ransac_iters,
        min_features=args.min_features,
        detect_interval=args.detect_interval,
        intensity_thresh=args.intensity_thresh,
        intensity_blur=args.intensity_blur,
        auto_intensity_percentile=args.auto_intensity_percentile,
        exclude_rect=tuple(args.exclude_rect) if args.exclude_rect else None,
        bright_spot=args.bright_spot,
        bright_spot_radius=args.bright_spot_radius,
        bright_spot_blur=args.bright_spot_blur,
        bright_spot_percentile=args.bright_spot_percentile,
        bright_min_area=args.bright_min_area,
        bright_max_area=args.bright_max_area,
        bright_max_aspect=args.bright_max_aspect,
        bright_dist_weight=args.bright_dist_weight,
        bright_max_jump=args.bright_max_jump,
        bright_min_fill=args.bright_min_fill,
        bright_max_dim=args.bright_max_dim,
    )
    tracker.initialize(prev_gray, roi)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    centers = []
    speeds = []
    trajectory = []
    prev_center = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = preprocess_frame(frame)

        update = tracker.update(prev_gray, curr_gray)
        prev_gray = curr_gray

        center = update["center"]
        if center is not None:
            trajectory.append(center)
            centers.append(center)
        else:
            centers.append(None)

        speed = None
        if prev_center is not None and center is not None:
            dist_px = np.linalg.norm(center - prev_center)
            speed = (dist_px * scale) / dt
        speeds.append(speed if speed is not None else 0.0)
        prev_center = center if center is not None else prev_center

        annotated = annotate(
            frame,
            tracker.roi,
            update["curr_points"],
            trajectory,
            center,
            speed,
            fps,
        )

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1
        if total_frames > 0 and frame_idx % max(1, total_frames // 10) == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    if writer is not None:
        writer.release()

    save_metrics(args.output_csv, centers, speeds)
    save_speed_plot(args.output_plot, speeds, fps)
    print("Tracking complete.")
    print(f"Annotated video: {args.output_video}")
    print(f"Metrics CSV:    {args.output_csv}")
    print(f"Speed plot:     {args.output_plot}")


def save_metrics(csv_path, centers, speeds):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "center_x", "center_y", "speed_m_s"])
        for idx, (center, speed) in enumerate(zip(centers, speeds)):
            if center is None:
                writer.writerow([idx, "", "", speed])
            else:
                writer.writerow([idx, float(center[0]), float(center[1]), speed])


def save_speed_plot(plot_path, speeds, fps):
    if not speeds:
        return
    times = np.arange(len(speeds)) / fps
    plt.figure(figsize=(8, 4))
    plt.plot(times, speeds, label="Speed (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Object speed over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    run()
