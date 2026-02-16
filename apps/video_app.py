import streamlit as st
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO
import tempfile
import time
from scipy.spatial import distance as dist
from collections import OrderedDict

DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

CLASS_NAMES = {
    0: "Green_Tomato",
    1: "Red_Tomato"
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}

@st.cache_resource
def load_detection_model(onnx_path="model/nbest.onnx"):
    providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)

@st.cache_resource
def load_video_detection_model(onnx_path="model/nbest.onnx"):
    return YOLO(onnx_path, task='detect')

@st.cache_resource
def load_segmentation_model(onnx_path="model/best.onnx"):
    providers = ["CUDAExecutionProvider"] if DEVICE == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)

def predict_mask(session, image, input_size=(256, 256)):
    orig_h, orig_w = image.shape[:2]
    img_resized = cv2.resize(image, input_size)
    img_input = img_resized.transpose(2,0,1)[None,...].astype(np.float32) / 255.0
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: img_input})[0]
    pred = np.argmax(pred, axis=1).squeeze(0)
    return cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def create_overlay(image, mask, alpha=0.5):
    class_colours = {
        0: (0, 0, 0),
        1: (128, 0, 128),
        2: (255, 0, 0),
        3: (0, 255, 0),
    }
    overlay = np.zeros_like(image, dtype=np.uint8)
    for class_id, colour in class_colours.items():
        if class_id == 0:
            continue
        overlay[mask==class_id] = colour
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return blended

def run_detection(session, image, input_size=(640,640), conf_threshold=0.25):
    orig_h, orig_w = image.shape[:2]
    img_resized = cv2.resize(image, input_size)
    img_input = img_resized.transpose(2,0,1)[None,...].astype(np.float32) / 255.0
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})[0]

    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)

    scale_x = orig_w / input_size[0]
    scale_y = orig_h / input_size[1]

    results = []
    for det in outputs:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_threshold:
            continue
        class_id = int(round(cls))
        results.append((
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
            float(conf),
            class_id
        ))
    return results

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    st_video_frame = st.empty()
    
    tracked_tomatoes = {}   
    next_id = 0

    total_tomatoes = 0
    red_tomatoes = 0
    green_tomatoes = 0

    distance_threshold = 40     
    iou_threshold = 0.3
    min_confidence = 0.5
    min_area = 500              
    max_missing = 15            
    confirm_streak = 3          

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            if w * h < min_area:
                continue

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls_id = int(box.cls)
            detections.append({
                'center': (cx, cy),
                'class_id': cls_id,
                'bbox': (x1, y1, x2, y2)
            })

        for track_id in list(tracked_tomatoes.keys()):
            tracked_tomatoes[track_id]['missed'] += 1
            if tracked_tomatoes[track_id]['missed'] > max_missing:
                del tracked_tomatoes[track_id]

        for det in detections:
            cx, cy = det['center']
            cls_id = det['class_id']
            bbox = det['bbox']

            matched_id = None
            best_iou = 0

            for tomato_id, track_info in tracked_tomatoes.items():
                if track_info['class_id'] != cls_id:
                    continue

                iou = bbox_iou(track_info['bbox'], bbox)
                distance = np.linalg.norm(np.array(track_info['center']) - np.array((cx, cy)))

                if (iou > iou_threshold or distance < distance_threshold) and iou > best_iou:
                    best_iou = iou
                    matched_id = tomato_id

            if matched_id is not None:
                tracked_tomatoes[matched_id]['center'] = (cx, cy)
                tracked_tomatoes[matched_id]['bbox'] = bbox
                tracked_tomatoes[matched_id]['missed'] = 0
                tracked_tomatoes[matched_id]['streak'] += 1


                if not tracked_tomatoes[matched_id]['counted'] and tracked_tomatoes[matched_id]['streak'] >= confirm_streak:
                    if cls_id == 1:
                        red_tomatoes += 1
                    elif cls_id == 0:
                        green_tomatoes += 1
                    total_tomatoes += 1
                    tracked_tomatoes[matched_id]['counted'] = True

                final_id = matched_id

            else:
                tracked_tomatoes[next_id] = {
                    'center': (cx, cy),
                    'class_id': cls_id,
                    'bbox': bbox,
                    'missed': 0,
                    'streak': 1,
                    'counted': False
                }
                final_id = next_id
                next_id += 1

 
            x1, y1, x2, y2 = tracked_tomatoes[final_id]['bbox']
            label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 0, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"ID {final_id}: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        count_text = f"Total: {total_tomatoes}, Red: {red_tomatoes}, Green: {green_tomatoes}"
        cv2.putText(annotated_frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        st_video_frame.image(annotated_frame, channels="BGR")
        time.sleep(1/30)

    cap.release()


def main():
    st.title("Tomato Analyzer")
    st.subheader("Analyze Tomato Diseases and Detect Tomatoes in an Image/Video")

    uploaded_file = st.file_uploader("Upload an image or video:", type=["jpg","jpeg","png","bmp","mp4","mov","avi"])
    if not uploaded_file:
        return

    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        img = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(img, caption="Original Image", use_container_width=True)

        col1, col2 = st.columns(2)
        seg_button = col1.button("Analyze Disease")
        det_button = col2.button("Detect Tomatoes")

        if seg_button:
            st.info("Analyzing for diseases...")
            model_seg = load_segmentation_model()
            mask = predict_mask(model_seg, img)
            overlay = create_overlay(img, mask, alpha=0.5)
            st.image(overlay, caption="Segmentation Overlay", use_container_width=True)
            st.markdown("Legend")
            st.markdown("""
            Purple Region: Early_Blight: This is a fungal disease of tomato leaf where there appear brown spots with some form of concentric rings or line like region within the brown spot.

            Red Region: Late_Blight: Another fungal disease of tomato leaf where irregular-shaped marks appear from the edge of the leaf.

            Green Region: Leaf_Miner: It is a case where a white pest infests a tomato leaf.
            """)

        if det_button:
            st.info("Detecting tomatoes...")
            model_det = load_detection_model()
            detections = run_detection(model_det, img)
            img_draw = img.copy()
            for x1, y1, x2, y2, conf, class_id in detections:
                label = CLASS_NAMES.get(class_id, f"Class {class_id}")
                color = CLASS_COLORS.get(class_id, (255, 0, 0))
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            st.image(img_draw, caption="Detection Bounding Boxes", use_container_width=True)
            st.markdown(f"Number of tomatoes detected: `{len(detections)}`")

    elif file_type == 'video':
        st.video(uploaded_file)
        if st.button("Detect Tomatoes in Video"):
            st.info("Processing video for tomato detection...")
            model_video_det = load_video_detection_model()
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_video_path = tmp_file.name

            process_video(temp_video_path, model_video_det)
            
    else:
        st.warning("Please upload a supported file type (image or video).")

if __name__ == "__main__":
    main()