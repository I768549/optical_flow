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
        self._fb_threshold = cfg["fb_threshold"]

        self._prev_gray = None
        self._prev_points = None
        self._prev_time = None

    def _detect_features(self, gray):
        # (N, 1, 2)
        points = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        return points

    def process_frame(self, frame, frame_ts):

        if frame is None:
            return None

        now = frame_ts # time потрібен для нормування в швидкість
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
        #Кадр був, але нічого не знайдено, тому повертаємо пустий результат з довірою до точок 0
        zero_result = (0.0, 0.0, dt, 0.0)
        # No features to track -- re-detect
        if self._prev_points is None or len(self._prev_points) < self._min_features:
            self._prev_points = self._detect_features(gray)
            self._prev_gray = gray
            self._prev_time = now
            if self._prev_points is None:
                return zero_result
            return None

        # Lucas-Kanade forward: prev -> curr
        next_points, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        if next_points is None or status_fwd is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return zero_result

        # Lucas-Kanade backward: curr -> prev (FB sanity check)
        back_points, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, next_points, None, **self._lk_params
        )

        if back_points is None or status_bwd is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return zero_result

        # Round-trip error: how far a point lands from where it started
        fb_error = np.linalg.norm(
            (self._prev_points - back_points).reshape(-1, 2), axis=1
        )

        # Good = forward ok, backward ok, round-trip tight
        good_mask = (
            (status_fwd.flatten() == 1)
            & (status_bwd.flatten() == 1)
            & (fb_error < self._fb_threshold)
        )
        total_points = len(status_fwd)
        good_count = int(np.sum(good_mask))

        if good_count == 0:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            self._prev_time = now
            return zero_result

        prev_good = self._prev_points[good_mask].reshape(-1, 2)
        next_good = next_points[good_mask].reshape(-1, 2)

        # Flow vectors per point
        flow_vectors = next_good - prev_good

        # MAD-based outlier rejection (euclidean: круг вокруг медианы)
        med = np.median(flow_vectors, axis=0)
        dist = np.linalg.norm(flow_vectors - med, axis=1)
        scale = 1.4826 * np.median(dist) + 1e-6
        inlier_mask = dist < 3.0 * scale
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count >= self._min_features:
            dx, dy = np.mean(flow_vectors[inlier_mask], axis=0)
        else:
            dx, dy = med
        dx = float(dx)
        dy = float(dy)

        # Fraction of originally tracked points that survived FB + MAD
        quality = inlier_count / total_points

        # Prepare next iteration
        self._prev_gray = gray
        self._prev_time = now

        # Keep tracked points, re-detect if too few
        if good_count < self._min_features:
            self._prev_points = self._detect_features(gray)
        else:
            self._prev_points = next_good.reshape(-1, 1, 2)

        return dx, dy, dt, quality

    def draw_overlay(self, frame, last_result=None, tracked_count=None):
        """Debug overlay: tracked points + median flow vector + stats text."""
        if frame is None:
            return frame

        if self._prev_points is not None:
            for p in self._prev_points.reshape(-1, 2):
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), 1)

        if last_result is not None:
            dx, dy, dt, quality = last_result
            arrow_scale = 20
            end_pt = (int(cx + dx * arrow_scale), int(cy + dy * arrow_scale))
            cv2.arrowedLine(frame, (cx, cy), end_pt, (0, 0, 255), 2, tipLength=0.3)
            text = (f"dx={dx:+.2f} dy={dy:+.2f} "
                    f"dt={dt*1000:.1f}ms q={quality:.2f}")
        else:
            text = "dx=---- dy=---- dt=---- q=----"

        if tracked_count is not None:
            text += f" pts={tracked_count}"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    @property
    def tracked_count(self):
        if self._prev_points is None:
            return 0
        return len(self._prev_points)

    def reset_state(self):
        self._prev_gray = None
        self._prev_points = None
        self._prev_time = None
