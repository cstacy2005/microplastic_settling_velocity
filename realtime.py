from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import csv
import time
import datetime
import math
import statistics

cap = cv2.VideoCapture(0)
x = input("Enter Name: ")
csv_path = f"./videos/RealTime/{x}.csv"

if (cap.isOpened() == False): 
    print("Error reading video file")

# Height set to width to adjust for frame rotation   
h, w, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter(f"./videos/RealTime/{x}.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10, [w,h])

# Initialize CSV file
header = ['frame', 'MP_count', 'x_center', 'y_center', 'velocity_1f', 'velocity_1f_cm',
          'velocity_5f', 'velocity_avg', "velocity_avg_cm", "size", "size_5f", "size_avg", 'calculated_fps']
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
 
data = []

track_history = defaultdict(lambda: [])

# Load the YOLO model
model = YOLO("runs/detect/train30/weights/best.pt")

# Horizontal line at 1/4 of the frame height for MP counting
line = [0, int(h / 4), w, int(h / 4)]  

total_MP = 0
frame_count = 0
prev_frame = None
velocity_1f = None
velocity_1f_cm = None
velocity_5f = None
velocity_avg = None
velocity_avg_cm = None
size = None
size_5f = None
size_avg = None
calculated_fps = None
distance_1f = None
MP_frame_end = None
MP_frame_start = None

# Measured distance of camera frame to use as conversion factor from pixels to centimeters
FRAME_CM = 28  

start_time = time.time()
fps_frame_count = 0

while True:
    ret, frame = cap.read()

    rotatedFrame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    dt = str(datetime.datetime.now())
    font = cv2.FONT_HERSHEY_SIMPLEX
    timeFrame = cv2.putText(rotatedFrame, dt, (10, 50), font, 0.3, (0, 0, 128), 1, cv2.LINE_AA)
    
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Method 1 of calculating FPS: Calculate FPS every second
    """ if elapsed_time >= 1.0:
        calculated_fps = fps_frame_count / elapsed_time

        start_time = current_time
        fps_frame_count = 0 """
    
    # Method 2 of calculating FPS: Calculate FPS every 10 frames
    if fps_frame_count == 10:
        calculated_fps = (fps_frame_count /elapsed_time)

        start_time = current_time
        fps_frame_count = 0
    
    fpsFrame = cv2.putText(timeFrame,str(calculated_fps),(10,30), font, 0.3, (0, 0, 128), 1, cv2.LINE_AA)
    
    video_writer.write(fpsFrame)
    
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    imageresults = model(fpsFrame)

    annotator = Annotator(fpsFrame,line_width=2)

    # Process detections
    for result in imageresults:
        boxes = result.boxes

        for box in boxes:
            conf = float(box.conf)
            if conf < 0.5:
                continue
                
            cls_id = int(box.cls)
            label = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()  #[x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Calculate center of bounding box            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Annotate image with bounding box and label
            annotator.box_label(xyxy, f'{label} {conf:.2f}', color=(255, 0, 0))
            
            if len(data) > 0 and not isinstance(data[-1][0], str):
                if (frame_count - data[-1][0]) == 0:
                    continue
            
            # Update MP count when new MP crosses the line
            if prev_frame is None or (frame_count - prev_frame >= 300):
                if center_y > line[1]:
                    total_MP += 1
                    prev_frame = frame_count
                      
            current_data = [frame_count, total_MP, center_x, center_y, velocity_1f, velocity_1f_cm, velocity_5f, velocity_avg, velocity_avg_cm, size, size_5f, size_avg, calculated_fps, distance_1f]
            
            # Calculate size (mm) for the current frame
            size = ((y2 - y1) / h) * FRAME_CM * 10
            current_data = [frame_count, total_MP, center_x, center_y, velocity_1f, velocity_1f_cm, velocity_5f, velocity_avg, velocity_avg_cm, size, size_5f, size_avg, calculated_fps, distance_1f]
            
            # Calculate avg size (mm) for the last 5 frames
            if len(data) >= 5 and (MP_frame_end is None or (frame_count - MP_frame_start >= 5)):
                sizes = []
                print(sizes)
                for row in data[-5:]:
                    sizes.append(row[9])
                print(sizes)
                size_5f = statistics.mean(sizes)
                current_data = [frame_count, total_MP, center_x, center_y, velocity_1f, velocity_1f_cm, velocity_5f, velocity_avg, velocity_avg_cm, size, size_5f, size_avg, calculated_fps, distance_1f]
            
            # Calculate distance and velocity for the last frame
            if len(data) > 0 and (MP_frame_end is None or (frame_count - MP_frame_start >=1)):
                distance_1f = math.sqrt((center_x - data[-1][2]) ** 2 + (center_y - data[-1][3]) ** 2)
                if frame_count - data[-1][0] == 0:
                    velocity_1f = distance_1f * calculated_fps
                else:
                    velocity_1f = distance_1f * calculated_fps * (1/(frame_count - data[-1][0]))
                velocity_1f_cm = velocity_1f * (FRAME_CM/h)
                current_data = [frame_count, total_MP, center_x, center_y, velocity_1f, velocity_1f_cm, velocity_5f, velocity_avg, velocity_avg_cm, size, size_5f, size_avg, calculated_fps, distance_1f]
            
            # Calculate average velocity for the last 5 frames
            if len(data) >= 6 and (MP_frame_end is None or (frame_count - MP_frame_start >= 6)):
                velocities = []
                for row in data[-5:]:
                    velocities.append(row[4])
                velocity_5f = statistics.mean(velocities)
                current_data = [frame_count, total_MP, center_x, center_y, velocity_1f, velocity_1f_cm, velocity_5f, velocity_avg, velocity_avg_cm, size, size_5f, size_avg, calculated_fps, distance_1f]

            # Calculate overall velocity and size averages for the current MP
            if len(data) > 2 and not isinstance(data[-1][0], str) and (frame_count- data[-1][0] >= 100):
                current_mp_velocities = []
                all_frames = []
                current_sizes = []
                blank_count = 0
                if MP_frame_start is None:
                    for row in data[1:]:
                        current_mp_velocities.append(row[4])
                    for row in data[0:]:
                        current_sizes.append(row[9])
                else:
                    for row in data[0:]:
                        if not isinstance(row[0], str):
                            all_frames.append(row[0])
                        else:
                            blank_count += 1
                    MP_start_index = all_frames.index((MP_frame_start)) + blank_count
                    for row in data[MP_start_index:]:
                        if isinstance(row[4], float):
                            current_mp_velocities.append(row[4])
                        if isinstance(row[9], float):
                            current_sizes.append(row[9])
                    
                velocity_avg = statistics.mean(current_mp_velocities)
                velocity_avg_cm = velocity_avg * (FRAME_CM/h)
                size_avg = statistics.mean(current_sizes)
                
                current_data = ["", "", "", "", "", "", "", velocity_avg, velocity_avg_cm, "", "", size_avg, "", ""]
                
                velocity_1f = None
                velocity_1f_cm = None
                velocity_5f = None
                velocity_avg = None
                velocity_avg_cm = None
                size = None
                size_5f = None
                size_avg = None
                distance_1f = None
                center_x = None
                center_y = None
                MP_frame_end = data[-2][0]
                MP_frame_start = frame_count + 1           
            
            data.append(current_data)
            
    annotated_frame = annotator.result()
    
    # Display current velocity and MP count on frame
    if len(data) > 1:
        annotated_frame = cv2.putText(annotated_frame, "Velocity: " + str(data[-1][5]), (10, 70), font, 0.3, (0, 0, 128), 1, cv2.LINE_AA)
        annotated_frame = cv2.putText(annotated_frame, "MP Count: " + str(data[-1][1]), (10, 90), font, 0.3, (0, 0, 128), 1, cv2.LINE_AA)
    
    cv2.imshow("Detection", annotated_frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    fps_frame_count +=1

# Calculate overall velocity and size averages for the final MP
current_mp_velocities = []
all_frames = []
current_sizes = []
blank_count = 0
if MP_frame_start is None:
    for row in data[1:]:
        current_mp_velocities.append(row[4])
    for row in data[0:]:
        current_sizes.append(row[9])
else:
    for row in data[0:]:
        if not isinstance(row[0], str):
            all_frames.append(row[0])
        else:
            blank_count += 1
    MP_start_index = all_frames.index((MP_frame_start)) + blank_count
    for row in data[MP_start_index:]:
        if isinstance(row[4], float):
            current_mp_velocities.append(row[4])
        if isinstance(row[9], float):
            current_sizes.append(row[9])
velocity_avg = statistics.mean(current_mp_velocities)
velocity_avg_cm = velocity_avg * (FRAME_CM/h) 
size_avg = statistics.mean(current_sizes)
data.append(["", "", "", "", "", "", "", velocity_avg, velocity_avg_cm, "", "", size_avg, "", ""])

# Write data to CSV file   
with open(csv_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)    

cap.release()
video_writer.release()
cv2.destroyAllWindows()