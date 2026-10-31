"""Multi-egg tracker (SORT-style, classical): Kalman filter + Hungarian assignment + track lifecycle.

For every frame the segmenter gives a list of egg masks (detections). The tracker keeps one Track per egg:
  1. predict  - each track's Kalman filter (constant velocity: x, y, vx, vy) predicts the egg centre
  2. cost     - for each (track, detection) pair:
                  d   = distance(predicted centre, detection centre) / egg diameter
                  iou = overlap of the track's last mask (shifted to the predicted centre) with the detection
                  cost = d + (1 - iou)            (small = similar)
                pairs with d > gate are forbidden; the gate grows while a track is unseen (it may have moved)
  3. assign   - Hungarian algorithm (globally optimal one-to-one matching)
  4. update   - matched: Kalman correction with the measured centre, store new mask
                unmatched track: "coasting" (occluded by the hand, missed detection); deleted after max_age frames
  4b. re-ID    - a track unseen for more than reid_after frames may claim a detection anywhere in the frame
                ("re-identification by elimination": the eggs look identical, but if the other egg is still
                tracked, an egg that reappears must be the lost one). Its Kalman state is reset to the new position.
                unmatched detection: new tentative track
  5. confirm  - a track gets a permanent ID only after min_hits consecutive matches, so one-frame false
                positives never get an ID.
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def mask_props(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()], np.float32), 2 * np.sqrt(len(xs) / np.pi)   # centre, equiv. diameter


class Track:
    def __init__(self, mask, frame):
        c, self.diam = mask_props(mask)
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32)      # eggs can be flicked: allow accel.
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
        kf.statePost = np.array([c[0], c[1], 0, 0], np.float32).reshape(4, 1)
        self.kf, self.mask, self.hits, self.miss, self.id = kf, mask, 1, 0, None
        self.first, self.trail = frame, [tuple(c)]

    @property
    def centre(self):
        return self.kf.statePost[:2, 0].copy()

    def predict(self):
        if self.miss:            # while unseen, damp the velocity so a coasting track does not fly away
            self.kf.statePost[2:] *= 0.7
        return self.kf.predict()[:2, 0].copy()

    def update(self, mask, reset=False):
        c, d = mask_props(mask)
        if reset:                # re-identified far away: restart the filter and the trail at the new position
            self.kf.statePost = np.array([c[0], c[1], 0, 0], np.float32).reshape(4, 1)
            self.kf.errorCovPost = np.diag([10, 10, 100, 100]).astype(np.float32)
            self.trail = []
        else:
            self.kf.correct(c.reshape(2, 1))
        self.mask, self.diam = mask, 0.7 * self.diam + 0.3 * d
        self.hits += 1; self.miss = 0
        self.trail.append(tuple(self.centre))


class EggTracker:
    def __init__(self, gate=1.0, gate_growth=0.05, max_age=48, min_hits=3, reid_after=None):
        self.gate, self.gate_growth, self.max_age, self.min_hits = gate, gate_growth, max_age, min_hits
        self.reid_after = reid_after
        self.tracks, self.next_id, self.frame = [], 1, 0

    def _cost(self, tr, pred, det):
        c, _ = mask_props(det)
        d = np.linalg.norm(pred - c) / max(tr.diam, 1)
        if self.reid_after is not None and tr.id is not None and tr.miss > self.reid_after:
            return 10.0 + d / 100          # long-lost: allowed anywhere, but always worse than a normal match
        if d > self.gate * (1 + self.gate_growth * tr.miss) + 0.5 * min(tr.miss, 10):
            return np.inf
        shift = (pred - mask_props(tr.mask)[0]).round().astype(int)
        moved = np.roll(np.roll(tr.mask, shift[1], 0), shift[0], 1)
        inter = np.logical_and(moved, det).sum()
        iou = inter / max(np.logical_or(moved, det).sum(), 1)
        return d + (1 - iou)

    def step(self, dets):
        """dets: list of boolean masks for this frame. Returns list of (track, matched_this_frame)."""
        preds = [t.predict() for t in self.tracks]
        C = np.full((len(self.tracks), len(dets)), 1e6)
        for i, (t, p) in enumerate(zip(self.tracks, preds)):
            for j, d in enumerate(dets):
                c = self._cost(t, p, d)
                if np.isfinite(c):
                    C[i, j] = c
        matched_t, matched_d = set(), set()
        if len(self.tracks) and len(dets):
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6:
                    t = self.tracks[i]
                    reset = self.reid_after is not None and t.miss > self.reid_after
                    t.update(dets[j], reset); matched_t.add(i); matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.miss += 1
                if t.id is None:
                    t.hits = 0                     # tentative tracks must be seen in consecutive frames
        self.tracks = [t for t in self.tracks if t.miss <= (self.max_age if t.id else 1)]
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(d, self.frame))
        for t in self.tracks:
            if t.id is None and t.hits >= self.min_hits:
                # ID inheritance (with re-ID enabled): a newly confirmed egg takes over the ID of an egg that is
                # currently missing (nearest last position) instead of getting a new ID
                lost = [o for o in self.tracks if o.id is not None and o.miss > 0]
                if self.reid_after is not None and lost:
                    o = min(lost, key=lambda o: np.linalg.norm(o.centre - t.centre))
                    t.id, o.miss = o.id, self.max_age + 1        # old track is dropped below
                else:
                    t.id = self.next_id; self.next_id += 1
        self.tracks = [t for t in self.tracks if t.id is None or t.miss <= self.max_age]
        self.frame += 1
        return [(t, t.miss == 0) for t in self.tracks if t.id is not None]
