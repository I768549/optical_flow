import time
import cv2
import numpy as np

class OpticalFlow:
    def __init__(self, cfg):
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

        self._prev_gray = None
        self._prev_points = None
        self._prev_time = None

    def _detect_features(self, gray):
        points = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        return points

    def process_frame(self, frame):

        if frame is None:
            return None

        now = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame -- just detect features
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return None

        dt = now - self._prev_time
        if dt < 1e-4:
            return None

        zero_result = (0.0, 0.0, dt, 0.0)

        # No features to track -- re-detect
        if self._prev_points is None or len(self._prev_points) < self._min_features:
            self._prev_points = self._detect_features(gray)
            self._prev_gray = gray
            self._prev_time = now
            if self._prev_points is None:
                return zero_result
            return None

        # Lucas-Kanade optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return zero_result

        # Filter good points
        good_mask = status.flatten() == 1
        total_points = len(status)
        good_count = int(np.sum(good_mask))

        if good_count == 0:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return zero_result

        quality = good_count / total_points

        prev_good = self._prev_points[good_mask].reshape(-1, 2)
        next_good = next_points[good_mask].reshape(-1, 2)

        # Flow vectors per point
        flow_vectors = next_good - prev_good

        # using median to be more stable for outliers
        dx = float(np.median(flow_vectors[:, 0]))
        dy = float(np.median(flow_vectors[:, 1]))

        # Prepare next iteration
        self._prev_gray = gray
        self._prev_time = now

        # Keep tracked points, re-detect if too few
        if good_count < self._min_features:
            self._prev_points = self._detect_features(gray)
        else:
            self._prev_points = next_good.reshape(-1, 1, 2)

        return dx, dy, dt, quality

    def reset_state(self):
        self._prev_gray = None
        self._prev_points = None
        self._prev_time = None
