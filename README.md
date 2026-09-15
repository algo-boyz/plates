# [Python Vehicle-Plates-Detection-and-Reading-System](https://github.com/kemalkilicaslan/Vehicle-Plates-Detection-and-Reading-System)

```python
# Vehicle Plates Detection and Reading System
# Import the necessary libraries
import cv2
import numpy as np
import supervision as sv
import re
from pathlib import Path
from rfdetr import RFDETRLarge
from fast_plate_ocr import LicensePlateRecognizer
# Paths (resolved relative to this script's folder)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "checkpoint_best_total.pth")  # RF-DETR weights
video_path = str(BASE_DIR / "Vehicle-Plates.mp4")  # write the name of the video file here
output_file = str(BASE_DIR / "Vehicle-Plates-Detection-and-Reading.mp4")  # extension the name and extension of the video file to be recorded
# Compute device for RF-DETR ("mps" = Apple Silicon GPU, "cuda" = NVIDIA GPU, "cpu" = fallback)
DEVICE = "mps"
# Fast Plate OCR settings (cct-s-v2-global-model = accurate, cct-xs-v2-global-model = fastest)
OCR_MODEL_NAME = "cct-s-v2-global-model"
OCR_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
# Minimum OCR confidence to accept a plate reading
OCR_CONFIDENCE_THRESHOLD = 0.95
# Frame size (width, height) each frame is resized to before RF-DETR detection
RFDETR_INPUT_SIZE = (1280, 720)
# Class ID(s) for plates and detection confidence threshold
PLATE_CLASS_IDS = [1]
DETECTION_THRESHOLD = 0.3
# Display window scale (1.0 = original, 0.5 = half size)
DISPLAY_SCALE = 0.5
# Fixed size (in pixels) of the white label block that shows the plate reading
LABEL_BOX_WIDTH = 180
LABEL_BOX_HEIGHT = 40
# Minimum box overlap (IoU) to treat a plate as the same one seen before
PLATE_MATCH_IOU = 0.2
# Frames a plate may stay undetected before its track (and locked text) is dropped
PLATE_MAX_MISSES = 60
# Turkish plate character correction maps
CHAR_TO_INT = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
INT_TO_CHAR = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}
# Function to format and validate a plate reading against the Turkish plate format
def format_plate_text(text: str):
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    cleaned = re.sub(r"[XQW]", "", cleaned)
    if not (7 <= len(cleaned) <= 10):
        return None
    part1 = "".join(CHAR_TO_INT.get(c, c) for c in cleaned[:2])
    if not part1.isdigit():
        return None
    part2, part3, found_letters = "", "", False
    for ch in cleaned[2:]:
        if ch.isalpha():
            if found_letters:
                return None
            part2 += INT_TO_CHAR.get(ch, ch)
        elif ch.isdigit():
            if not found_letters and part2:
                found_letters = True
            part3 += CHAR_TO_INT.get(ch, ch)
        else:
            return None
    if (1 <= len(part2) <= 3 and 1 <= len(part3) <= 4
            and part2.isalpha() and part3.isdigit()):
        return f"{part1} {part2} {part3}"
    return None
# Function to compute the lowest per-character probability across the real (non-pad) characters
def reading_confidence(padded_text: str, char_probs, pad_char: str) -> float:
    if char_probs is None:
        return 1.0
    probs = np.asarray(char_probs).ravel()
    real = [float(p) for ch, p in zip(padded_text, probs) if ch != pad_char]
    return min(real) if real else 0.0
# Function to read a cropped plate and return (formatted_text, confidence) pairs sorted by confidence
def plate_ocr(plate_bgr: np.ndarray, ocr_model) -> list:
    try:
        # The v2 global CCT models expect RGB crops; the library resizes internally
        rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        pad_char = getattr(getattr(ocr_model, "config", None), "pad_char", "_")
        # Keep text aligned 1:1 with char_probs
        preds = ocr_model.run(rgb, return_confidence=True, remove_pad_char=False)
        results = []
        for pred in preds:
            padded = getattr(pred, "plate", str(pred))
            conf = reading_confidence(padded, getattr(pred, "char_probs", None), pad_char)
            formatted = format_plate_text(padded.replace(pad_char, ""))
            if formatted:
                results.append((formatted, conf))
        results.sort(key=lambda r: r[1], reverse=True)
        return results
    except Exception as e:
        print(f"OCR error: {e}")
        return []
# Function to run RF-DETR inference and scale detections back to the original frame size
def run_rfdetr(rfdetr_model, frame_bgr: np.ndarray) -> sv.Detections:
    orig_h, orig_w = frame_bgr.shape[:2]
    target_w, target_h = RFDETR_INPUT_SIZE
    resized = cv2.resize(frame_bgr, (target_w, target_h))
    detections = rfdetr_model.predict(resized, threshold=DETECTION_THRESHOLD)
    if len(detections) == 0:
        return detections
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    scaled = detections.xyxy.copy()
    scaled[:, 0] = np.clip(scaled[:, 0] * scale_x, 0, orig_w)
    scaled[:, 1] = np.clip(scaled[:, 1] * scale_y, 0, orig_h)
    scaled[:, 2] = np.clip(scaled[:, 2] * scale_x, 0, orig_w)
    scaled[:, 3] = np.clip(scaled[:, 3] * scale_y, 0, orig_h)
    detections.xyxy = scaled
    return detections
# Function to draw the plate box and the recognized plate label onto the frame
def annotate_plate(frame: np.ndarray, x1, y1, x2, y2, plate_text: str) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    pad = 6
    # Draw the plate detection box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    # The white label block is always the same fixed size on every frame
    box_w = LABEL_BOX_WIDTH
    box_h = LABEL_BOX_HEIGHT
    # Pick the largest font scale whose text fits inside the fixed block width/height
    font_scale = 0.3
    for scale in np.linspace(1.2, 0.3, 20):
        (tw, th), baseline = cv2.getTextSize(plate_text, font, scale, thickness)
        if tw <= box_w - 2 * pad and th + baseline <= box_h - 2 * pad:
            font_scale = scale
            break
    (text_width, text_height), baseline = cv2.getTextSize(plate_text, font, font_scale, thickness)
    # Center the block on the plate, place it above (or below) the box, and clamp it inside the frame
    center_x = (x1 + x2) // 2
    bg_x1 = int(np.clip(center_x - box_w // 2, 0, max(0, frame_w - box_w)))
    bg_y1 = y1 - box_h if y1 - box_h >= 0 else min(y2, frame_h - box_h)
    bg_y1 = max(0, bg_y1)
    bg_x2 = bg_x1 + box_w
    bg_y2 = bg_y1 + box_h
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
    # Center the text within the fixed block
    text_x = bg_x1 + (box_w - text_width) // 2
    text_y = bg_y1 + (box_h + text_height) // 2
    cv2.putText(frame, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return frame
# Function to compute Intersection-over-Union between two (x1, y1, x2, y2) boxes
def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)
# Function to detect plates in a single frame and read their text
def process_frame(frame, rfdetr_model, ocr_model, state):
    annotated_frame = frame.copy()
    all_detections = run_rfdetr(rfdetr_model, frame)
    # Keep only plate detections
    if len(all_detections) > 0:
        plate_dets = all_detections[np.isin(all_detections.class_id, PLATE_CLASS_IDS)]
    else:
        plate_dets = all_detections
    # Persistent tracks that carry each plate's locked-in text across frames
    tracks = state.get("tracks", [])
    used = [False] * len(tracks)
    # Plate OCR for detected plates
    for box in plate_dets.xyxy:
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            continue
        # Match this plate to the best-overlapping existing track not used yet
        best_iou, best_idx = 0.0, -1
        for i, tr in enumerate(tracks):
            if used[i]:
                continue
            score = box_iou((x1, y1, x2, y2), tr["box"])
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_idx != -1 and best_iou >= PLATE_MATCH_IOU:
            # Same plate as before: keep its stored (locked) text, refresh its position
            track = tracks[best_idx]
            used[best_idx] = True
            track["box"] = (x1, y1, x2, y2)
            track["misses"] = 0
            plate_text = track["text"]
        else:
            # A newly seen plate gets its own track
            track = {"box": (x1, y1, x2, y2), "text": None, "conf": 0.0, "misses": 0}
            tracks.append(track)
            used.append(True)
            plate_text = None
        # Run OCR only until the first reading above the confidence threshold is locked into the track
        if plate_text is None:
            plate_region = frame[y1:y2, x1:x2]
            if plate_region.size == 0:
                continue
            ocr_results = plate_ocr(plate_region, ocr_model)
            if ocr_results:
                best_text, best_conf = ocr_results[0]
                if best_conf >= OCR_CONFIDENCE_THRESHOLD:
                    plate_text = best_text
                    track["text"] = best_text
                    track["conf"] = best_conf
        if plate_text:
            annotated_frame = annotate_plate(annotated_frame, x1, y1, x2, y2, plate_text)
    # Age out tracks not matched this frame, keeping them briefly to bridge detection gaps
    surviving = []
    for i, track in enumerate(tracks):
        if not used[i]:
            track["misses"] += 1
        if track["misses"] <= PLATE_MAX_MISSES:
            surviving.append(track)
    state["tracks"] = surviving
    return annotated_frame
# Load RF-DETR and Fast Plate OCR models
print("Initializing models...")
rfdetr_model = RFDETRLarge(pretrain_weights=MODEL_PATH, device=DEVICE)
try:
    ocr_model = LicensePlateRecognizer(OCR_MODEL_NAME, providers=OCR_PROVIDERS)
except Exception as e:
    # Fall back to plain CPU if CoreML is not available
    print(f"CoreML provider unavailable ({e}); falling back to CPU.")
    ocr_model = LicensePlateRecognizer(OCR_MODEL_NAME)
print("All models ready!")
# Initialize video processing
video_info = sv.VideoInfo.from_video_path(video_path=video_path)
frames_generator = sv.get_video_frames_generator(source_path=video_path)
# Create the VideoWriter object to save the video file
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_file, fourcc, video_info.fps, video_info.resolution_wh)
# Holds the active plate tracks so each plate's locked text stays identical across frames
plate_state = {"tracks": []}
# Prepare the live window (fall back to headless mode if the GUI is not available)
window_title = "Vehicle Plates Detection and Reading System"
try:
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    display_width = int(video_info.resolution_wh[0] * DISPLAY_SCALE)
    display_height = int(video_info.resolution_wh[1] * DISPLAY_SCALE)
    cv2.resizeWindow(window_title, display_width, display_height)
    gui_available = True
except Exception:
    gui_available = False
    print("GUI not available, running headless. Output will be saved to file.")
for frame in frames_generator:
    # Detect and read plates in the frame
    annotated_frame = process_frame(
        frame=frame,
        rfdetr_model=rfdetr_model,
        ocr_model=ocr_model,
        state=plate_state
    )
    # Write the drawn frame to the video file to be saved
    output_video.write(annotated_frame)
    # Show video with vehicle plate detection and reading
    if gui_available:
        cv2.imshow(window_title, annotated_frame)
        # Switch off video when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release all open windows
output_video.release()
cv2.destroyAllWindows()
print(f"Done. Output saved to: {output_file}")
```