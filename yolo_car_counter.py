import argparse

import cv2
from ultralytics import YOLO


def count_cars_from_webcam(camera_index: int = 0, output_path: str | None = None, model_name: str = "yolov8n.pt") -> None:
    """
    Detect and count cars from a webcam stream using YOLOv8.
    Press 'q' to stop.
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise ValueError(f"Could not open webcam with index: {camera_index}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    peak_cars = 0

    print("Starting webcam detection... Press 'q' in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from webcam.")
            break

        results = model(frame, verbose=False)[0]

        car_count = 0
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0].item())
                if class_id == 2:  # COCO class id for car
                    car_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "Car",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        peak_cars = max(peak_cars, car_count)
        frame_idx += 1

        cv2.putText(
            frame,
            f"Cars in frame: {car_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("YOLOv8 Car Counter - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Detection completed.")
    print(f"Total frames processed: {frame_idx}")
    print(f"Peak cars in a frame: {peak_cars}")
    if output_path:
        print(f"Annotated video saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and count cars from webcam using YOLOv8.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--output", default=None, help="Path to save annotated output video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights (default: yolov8n.pt)")
    args = parser.parse_args()

    count_cars_from_webcam(args.camera, args.output, args.model)


if __name__ == "__main__":
    main()
