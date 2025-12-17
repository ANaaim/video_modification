
import cv2
from pathlib import Path
import mediapipe as mp
import numpy as np
# ================= CONFIG =================
INPUT_VIDEO = "test_video.avi"
OUTPUT_DIR = Path("blurred_frames_upright")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Blur options
BLUR_TYPE = "black"   # "gaussian" or "pixelate" or "black"
GAUSSIAN_SCALE = 7       # larger => stronger; kernel size scales with face size
PIXELATE_BLOCK = 16      # larger => coarser pixelation
PAD_FRAC = 0.10          # expand bbox 10% to include edges/hairline

# Face detector settings
MODEL_SELECTION = 1      # 0: short-range (close faces), 1: full-range (far/small faces)
MIN_CONF = 0.1

# Heuristic orientation probe
PROBE_FRAMES = 40         # number of frames to sample across the video
ROT_CANDIDATES = [0, 90, 180, 270]  # allowed rotations (multiples of 90°)

# ================= HELPERS =================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def rotate_image_90s(img, angle):
    """Rotate image by 0/90/180/270 degrees using fast OpenCV ops."""
    angle = angle % 360
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only multiples of 90 degrees supported.")


def blur_roi(img, x1, y1, x2, y2):
    """Apply configured blur/blackout to an ROI within the image."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return

    if BLUR_TYPE == "gaussian":
        # Kernel sizes must be odd; scale with face box size
        kx = max(3, ((x2 - x1) // GAUSSIAN_SCALE) | 1)
        ky = max(3, ((y2 - y1) // GAUSSIAN_SCALE) | 1)
        blurred = cv2.GaussianBlur(roi, (kx, ky), 0)

    elif BLUR_TYPE == "pixelate":
        # Pixelation via downscale/upscale
        w_small = max(1, (x2 - x1) // PIXELATE_BLOCK)
        h_small = max(1, (y2 - y1) // PIXELATE_BLOCK)
        small = cv2.resize(roi, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    elif BLUR_TYPE == "black":
        # Fast blackout: fill ROI with black. Works for grayscale and BGR.
        # If you ever have 4-channel (BGRA), keep alpha opaque.
        if roi.ndim == 2:  # grayscale
            blurred = np.zeros_like(roi)
        else:
            blurred = np.zeros_like(roi)  # BGR or BGRA -> all zeros => black
            if roi.shape[2] == 4:
                blurred[:, :, 3] = 255  # optional: keep alpha opaque instead of transparent
    else:
        raise ValueError(f"Unknown BLUR_TYPE '{BLUR_TYPE}'. Use 'gaussian', 'pixelate', or 'black'.")

    img[y1:y2, x1:x2] = blurred


def choose_rotation_by_probe(cap, face_detector, probe_frames=PROBE_FRAMES):
    """
    Heuristic: sample a few frames evenly across the video, try each candidate rotation,
    score by sum of detection confidences, and pick the rotation with the highest score.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Select sample indices across the video
    if total_frames > 0:
        idxs = [int((i + 0.5) * total_frames / probe_frames) for i in range(probe_frames)]
    else:
        # If frame count unknown, sample first N frames
        idxs = list(range(probe_frames))

    scores = {angle: 0.0 for angle in ROT_CANDIDATES}
    # Save current position to restore later
    cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    for target_idx in idxs:
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        for angle in ROT_CANDIDATES:
            fr = rotate_image_90s(frame_bgr, angle)
            fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = face_detector.process(fr_rgb)
            if res.detections:
                # Add confidences (score[0] holds detection confidence)
                scores[angle] += sum(d.score[0] for d in res.detections)

    # Restore original read position
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)

    # Pick the rotation with the highest score; default to 0 if tie/all zero
    best_angle = max(scores, key=scores.get)
    # If all scores are zero, it's possible faces are too hard to detect; return 0
    if all(s == 0.0 for s in scores.values()):
        best_angle = 0
    return best_angle, scores

# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=MODEL_SELECTION,
                               min_detection_confidence=MIN_CONF) as face_det:
        # 1) Heuristically pick the upright rotation
        angle, scores = choose_rotation_by_probe(cap, face_det, PROBE_FRAMES)
        print(f"Heuristic rotation scores: {scores} -> Using {angle}°")

        # 2) Process all frames: rotate to upright, detect+blur, rotate back
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            # Rotate to chosen upright orientation
            upright = rotate_image_90s(frame_bgr, angle)
            h, w = upright.shape[:2]

            # Detect faces on upright frame
            upright_rgb = cv2.cvtColor(upright, cv2.COLOR_BGR2RGB)
            results = face_det.process(upright_rgb)

            # Blur all detected boxes in upright coordinates
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)

                    pad = int(PAD_FRAC * max(bw, bh))
                    x1 = clamp(x - pad, 0, w - 1)
                    y1 = clamp(y - pad, 0, h - 1)
                    x2 = clamp(x + bw + pad, 1, w)
                    y2 = clamp(y + bh + pad, 1, h)

                    blur_roi(upright, x1, y1, x2, y2)

            # Rotate back to original orientation for output
            processed = rotate_image_90s(upright, (-angle) % 360)

            # Save lossless PNG frame (compression level 0 = fastest, still lossless)
            out_path = OUTPUT_DIR / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), processed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} PNG frames to {OUTPUT_DIR} at ~{fps:.3f} FPS")

if __name__ == "__main__":
    main()
