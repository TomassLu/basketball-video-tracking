import cv2
from ultralytics import YOLO

# Detecting players in the first frame for selection for tracking
def detect_persons_in_first_frame(video_path):
    model = YOLO("yolov8l.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None

    ret, frame = cap.read()
    if not ret:
        print("Error reading video frame")
        return None

    results = model(frame)
    person_rects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            # Display rectangles around players
            if cls == 0 and conf > 0.4 and (x2 - x1) * (y2 - y1) > 1000:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                person_rects.append(((x1, y1), (x2, y2)))

    # Resizing the window
    original_height, original_width = frame.shape[:2]
    frame = cv2.resize(frame, (1280, 720))  # Resize the frame
    scale_x = 1280 / original_width
    scale_y = 720 / original_height
    selected_box = None
    # Letting user select player for tracking
    def click_event(event, x, y, flags, param):
        nonlocal selected_box
        scaled_x = int(x / scale_x)
        scaled_y = int(y / scale_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in person_rects:
                (x1, y1), (x2, y2) = box
                if x1 <= scaled_x <= x2 and y1 <= scaled_y <= y2:
                    selected_box = box
                    cv2.destroyAllWindows()
                    break


    cv2.imshow("Select a Person", frame)
    cv2.setMouseCallback("Select a Person", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    return selected_box

# Calculating Intersection over Union
def compute_iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
    boxBArea = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Tracking the selected player from the first frame
def track_selected_person(video_path, selected_box, N=10):
    model = YOLO("yolov8l.pt")
    cap = cv2.VideoCapture(video_path)
    tracker = cv2.TrackerCSRT_create() # Using tracker CSRT
    (x1, y1), (x2, y2) = selected_box
    tracker.init(cap.read()[1], (x1, y1, x2 - x1, y2 - y1))
    frame_count = 0

    tracked_frames = []  # List to store frames

    #Tracking the select player with tracker and periodically correcting the bounding box location with iou
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        success, new_box = tracker.update(frame)

        if success:
            x, y, w, h = map(int, new_box)
            updated_box = ((x, y), (x + w, y + h))

            if frame_count % N == 0:
                results = model(frame)
                best_match = updated_box
                highest_iou = 0

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())

                        if cls == 0 and conf > 0.4:
                            detected_box = ((x1, y1), (x2, y2))
                            iou = compute_iou(updated_box, detected_box)

                            if iou > highest_iou:
                                highest_iou = iou
                                best_match = detected_box

                if highest_iou < 0.5:
                    (x1, y1), (x2, y2) = best_match
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        tracked_frames.append(frame)  # Store frame in list
        frame = cv2.resize(frame, (1280, 720))  # Resize the frame
        cv2.imshow("Tracking", frame) #Showing the tracked frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_video_from_frames(frames, output_path, fps=30):
    if not frames:
        print("No frames to create video.")
        return

    # Get the frame dimensions from the first frame
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()


video_path = "basketball_video.mp4"
selected_box = detect_persons_in_first_frame(video_path)
if selected_box:
    tracked_frames = []  # List to store frames
    track_selected_person(video_path, selected_box)
    output_video_path = 'tracked_video.mp4'
    print("Generating video")
    create_video_from_frames(tracked_frames, output_video_path)
    print("Video generation finished. File name 'tracked_video.mp4'")

cv2.destroyAllWindows()