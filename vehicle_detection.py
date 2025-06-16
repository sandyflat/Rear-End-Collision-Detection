from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolov8n.pt")
model.to("cpu")

vehicle_classes = {
    2: ("car", (0, 255, 0)),
    3: ("motorcycle", (0, 255, 0)),
    5: ("bus", (0, 255, 0)),
    7: ("truck", (0, 255, 0))
}

def detect_vehicles(frame, lane_zones, filter_inside=True, min_overlap_ratio=0.05):
    result = model.track(frame, persist=True, verbose=False)[0]
    boxes = []
    annotated_frame = frame.copy()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_polygon = np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2]
        ], dtype=np.int32)

        bbox_area = (x2 - x1) * (y2 - y1)

        if filter_inside:
            intersection_area, _ = cv2.intersectConvexConvex(
                box_polygon.astype(np.float32),
                lane_zones['full_trapezoid'].astype(np.float32)
            )
            overlap_ratio = intersection_area / bbox_area if bbox_area > 0 else 0
            if overlap_ratio < min_overlap_ratio:
                continue

        # Determine which zone this vehicle is in
        zone_color = (0, 255, 0)
        for zone_name, color in [('red_zone', (0, 0, 255)),
                                 ('yellow_zone', (0, 255, 255)),
                                 ('green_zone', (0, 255, 0))]:
            zone_polygon = lane_zones[zone_name]
            intersection_area, _ = cv2.intersectConvexConvex(
                box_polygon.astype(np.float32),
                zone_polygon.astype(np.float32)
            )
            overlap_ratio = intersection_area / bbox_area if bbox_area > 0 else 0
            if overlap_ratio >= min_overlap_ratio:
                zone_color = color
                break

        label, _ = vehicle_classes[cls_id]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), zone_color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        boxes.append((x1, y1, x2, y2))

    return annotated_frame, boxes