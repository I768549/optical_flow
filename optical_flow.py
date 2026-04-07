import argparse
import time
import json
import sys
import os
import cv2
import numpy as np
import yaml
from messenger_async import DDSMessenger


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class OpticalFlowSender:
    def __init__(self, device, partition, domain_id: int, config_path=None):
        self._device = device
        self._domain_id = domain_id
        self._partition = partition

        cfg = load_config(config_path)

        self._lk_params = dict(
            winSize=(cfg["lk_win_size"], cfg["lk_win_size"]),
            maxLevel=cfg["lk_max_level"],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      cfg["lk_max_count"], cfg["lk_epsilon"]),
        )
        self._feature_params = dict(
            maxCorners=cfg["feature_max_corners"],
            qualityLevel=cfg["feature_quality_level"],
            minDistance=cfg["feature_min_distance"],
            blockSize=cfg["feature_block_size"],
        )
        self._min_features = cfg["min_features"]
        self._frame_width = cfg["frame_width"]
        self._frame_height = cfg["frame_height"]

        self._cap = None
        self._messenger = DDSMessenger(domain_id=domain_id)
        self._prev_gray = None
        self._prev_points = None
        self._prev_time = None

        self._messenger_ready = False

    def start(self):
        self._connect_camera()
        self._connect_messenger()

        print("Optical flow sender running. Press Ctrl+C to stop.")
        try:
            while True:
                self._process_frame()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._cleanup()

    def _connect_camera(self):
        if isinstance(self._device, int) or self._device.isdigit():
            self._cap = cv2.VideoCapture(int(self._device))
        else:
            self._cap = cv2.VideoCapture(self._device)

        if not self._cap.isOpened():
            print(f"Failed to open camera device: {self._device}")
            sys.exit(1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        print(f"Camera opened: {self._device} ({self._frame_width}x{self._frame_height})")

    def _connect_messenger(self):
        try:
            self._messenger.init()
            self._messenger_ready = True
            print(f"DDS messenger initialized (domain_id={self._domain_id})")
        except Exception as e:
            self._messenger_ready = False
            print(f"Failed to initialize DDS messenger: {e}")
            print("Running without publisher connection (will skip sends).")

    def _detect_features(self, gray):
        points = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        return points

    def _process_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            return

        now = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame -- just detect features
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return

        dt = now - self._prev_time
        if dt < 1e-4:
            return

        # No features to track -- re-detect
        if self._prev_points is None or len(self._prev_points) < self._min_features:
            self._prev_points = self._detect_features(gray)
            if self._prev_points is None:
                self._prev_gray = gray
                self._prev_time = now
                self._send_flow(0.0, 0.0, dt, 0.0)
                return

        # Lucas-Kanade optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            self._send_flow(0.0, 0.0, dt, 0.0)
            return

        # Filter good points
        good_mask = status.flatten() == 1
        total_points = len(status)
        good_count = int(np.sum(good_mask))

        if good_count == 0:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            self._send_flow(0.0, 0.0, dt, 0.0)
            return

        quality = good_count / total_points

        prev_good = self._prev_points[good_mask]
        next_good = next_points[good_mask]

        # Flow vectors per point
        flow_vectors = next_good - prev_good

        # Median -- robust to outliers (shadows, moving objects)
        dx = float(np.median(flow_vectors[:, 0]))
        dy = float(np.median(flow_vectors[:, 1]))

        self._send_flow(dx, dy, dt, quality)

        # Prepare next iteration
        self._prev_gray = gray
        self._prev_time = now

        # Keep tracked points, re-detect if too few
        if good_count < self._min_features:
            self._prev_points = self._detect_features(gray)
        else:
            self._prev_points = next_good.reshape(-1, 1, 2)

    def _send_flow(self, dx, dy, dt, quality):
        payload = json.dumps({
            "dx": round(dx, 4),
            "dy": round(dy, 4),
            "dt": round(dt, 6),
            "quality": round(quality, 3),
        })

        if self._messenger_ready:
            try:
                # New tracker dev format: topic-based publish with payload only
                self._messenger.send("FLOW_DATA", payload)
            except Exception as e:
                print(f"Send error: {e}")

    def _cleanup(self):
        if self._cap is not None:
            self._cap.release()
        if self._messenger_ready:
            self._messenger.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Optical flow sender for FPV drop mission")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path (default: 0)")
    
    parser.add_argument("--partition", default="", help="DDs partition (must match handler)")
    
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain id (default: 0)")
    
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml next to this script)")
    args = parser.parse_args()

    sender = OpticalFlowSender(args.device, args.partition, args.domain_id,  args.config)
    sender.start()


if __name__ == "__main__":
    main()
