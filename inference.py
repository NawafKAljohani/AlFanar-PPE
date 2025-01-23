import os
from ultralytics import YOLO
import cv2
import supervision as sv
from multiprocessing import Queue, Process
from dotenv import load_dotenv
import requests
import logging
import dashboard  # Import dashboard module to access accumulate_event and push_event functions

# ============================
# Configuration and Initialization
# ============================

def initialize_environment():
    """Load environment variables and configure logging."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

logger = initialize_environment()

PEOPLE_MODEL_PATH = os.getenv("PEOPLE_MODEL_PATH")
PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH")

# ============================
# Models Initialization
# ============================

def load_model(model_path, model_name):
    """Load a YOLO model from a given path."""
    try:
        model = YOLO(model_path)
        logger.info(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        logger.critical(f"Failed to load {model_name} model: {e}")
        exit(1)

person_model = load_model(PEOPLE_MODEL_PATH, "Person detection")
ppe_model = load_model(PPE_MODEL_PATH, "PPE detection")

# ============================
# Directory and Constants
# ============================

VIOLATION_CLASSES = [1, 3, 5]  # No_Gloves (1), No_Sleeves (3), No_Helmet (5)
streams_file = "cameras.streams"
assert os.path.exists(streams_file), f"Streams file not found: {streams_file}"

# ============================
# Main Stream Processing
# ============================

logger.info(f"Processing streams from {streams_file}")
results = person_model(source=streams_file, stream=True, verbose=False)  # Multi-stream inference

event_queue = Queue(maxsize=10)  # Queue for sending events
event_session = requests.Session()  # Event API session

# Start the push_event process
event_process = Process(target=dashboard.push_event, args=(event_session, event_queue))
event_process.start()

# Initialize a dictionary to keep track of detected violations for each person ID
person_violations = {}

for result in results:
    frame = result.orig_img
    detections = sv.Detections.from_ultralytics(result)
    stream_name = result.path if hasattr(result, 'path') else f"Stream_{id(result)}"

    # Ensure location_id is always valid (1 or more)
    location_id = 1 if stream_name == "0" else stream_name

    # Initialize detections and violations tracking
    stream_violations = {}
    detections = sv.ByteTrack().update_with_detections(detections)

    # Process person detections
    person_indices = [i for i, class_id in enumerate(detections.class_id) if class_id == 0]
    person_detections = {i: detections.xyxy[i] for i in person_indices}

    for person_id, bbox in person_detections.items():
        x1, y1, x2, y2 = bbox
        tracker_id = detections.tracker_id[person_id]

        if tracker_id not in person_violations:
            person_violations[tracker_id] = {1: 0, 3: 0, 5: 0}  # 1: No_Gloves, 3: No_Sleeves, 5: No_Helmet

        # Annotate person detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Person #{tracker_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Run PPE detection
        ppe_results = ppe_model.predict(source=frame, save=False, verbose=False)[0]
        ppe_detections = sv.Detections.from_ultralytics(ppe_results)

        detection_types = []

        # Track violations for this person
        for ppe_idx, ppe_bbox in enumerate(ppe_detections.xyxy):
            ppe_class_id = ppe_detections.class_id[ppe_idx]
            if (ppe_bbox[0] >= x1 and ppe_bbox[1] >= y1 and
                ppe_bbox[2] <= x2 and ppe_bbox[3] <= y2 and
                ppe_class_id in VIOLATION_CLASSES):

                detection_type = "No_Gloves" if ppe_class_id == 1 else "No_Sleeves" if ppe_class_id == 3 else "No_Helmet"

                # Ensure only a maximum of 1 violation for helmet, 2 for gloves, and 2 for sleeves
                if detection_type == "No_Gloves" and person_violations[tracker_id][1] < 1:
                    detection_types.append(detection_type)
                    person_violations[tracker_id][1] += 1  # Increment gloves violation
                elif detection_type == "No_Sleeves" and person_violations[tracker_id][3] < 2:
                    detection_types.append(detection_type)
                    person_violations[tracker_id][3] += 1  # Increment sleeves violation
                elif detection_type == "No_Helmet" and person_violations[tracker_id][5] < 1:
                    detection_types.append(detection_type)
                    person_violations[tracker_id][5] += 1  # Increment helmet violation

                # Annotate PPE violation
                violation_label = f"Violation: {detection_type.replace('_', ' ')}"
                cv2.rectangle(frame, (int(ppe_bbox[0]), int(ppe_bbox[1])), (int(ppe_bbox[2]), int(ppe_bbox[3])), (0, 0, 255), 2)
                cv2.putText(frame, violation_label, (int(ppe_bbox[0]), int(ppe_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Send violations to the event queue if we have any violations
        if detection_types:
            dashboard.accumulate_event(location_id=location_id, frame=frame, detection_types=detection_types, results_queue=event_queue)

    # Display the annotated frameq
    cv2.imshow(f"Location {location_id}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Processing ended by user.")
        break

cv2.destroyAllWindows()
event_queue.put(None)  # Signal push_event to stop
event_process.join()  # Ensure push_event process terminates
