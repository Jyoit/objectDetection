import cv2
from ultralytics import YOLO

# Load the YOLOv8 model 
model = YOLO("yolov8m.pt")  # Use the nano version yolov8n.pt for faster processing  Larger models like yolov8m.pt or yolov8l.pt are more accurate but slower.

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera, or replace with video file path

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for better processing
    frame = cv2.resize(frame, (640, 480))
    
    # Run YOLOv8 inference on the frame
    results = model(frame, conf=0.5, iou=0.45)  # Increase confidence threshold to 0.5

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()