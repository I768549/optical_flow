"""
optical flow computation 

Captures frames from the bottom fpv camera computes Lucas-Kanade
optical flow, and sends results to huha_handler via DDS messenger topic

Usage example:

python optical_flow_sender.py --device 0 --domain-id 0

The huha_handler receives data in OpticalFlowReceiver via "FLOW_DATA" topic.
"""

import argparse
import time
import json
import sys

import cv2
import numpy as np
from messenger_async import DDSMessenger


# Lucas-Kanade params
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Feature detection params
FEATURE_PARAMS = dict(
    maxCorners=80,
    qualityLevel=0.05,
    minDistance=10,
    blockSize=7,
)

# Re-detect features when count drops below this
MIN_FEATURES = 15

# Frame processing resolution
FRAME_WIDTH = 320
FRAME_HEIGHT = 240


class OpticalFlowSender:
    def __init__(self, device, domain_id: int):
        self._device = device
        self._domain_id = domain_id

        self._cap = None
        self._messenger = DDSMessenger(domain_id)
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

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Camera opened: {self._device} ({FRAME_WIDTH}x{FRAME_HEIGHT})")

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
        points = cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)
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
        if self._prev_points is None or len(self._prev_points) < MIN_FEATURES:
            self._prev_points = self._detect_features(gray)
            if self._prev_points is None:
                self._prev_gray = gray
                self._prev_time = now
                self._send_flow(0.0, 0.0, dt, 0.0)
                return

        # Lucas-Kanade optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **LK_PARAMS
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
        if good_count < MIN_FEATURES:
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
    parser = argparse.ArgumentParser(description="Optical flow sender for FPV drone")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path (default: 0)")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain id (default: 0)")
    args = parser.parse_args()

    sender = OpticalFlowSender(args.device, args.domain_id)
    sender.start()


if __name__ == "__main__":
    main()
