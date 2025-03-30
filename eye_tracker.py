import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
import os
from threading import Thread
from queue import Queue
import threading

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True

class GazeTracker:
    def __init__(self):
        self.calibration_points = []
        self.calibration_data = []
        self.is_calibrated = False
        self.screen_width = 1920
        self.screen_height = 1080
        self.last_valid_gaze = None
        self.smoothing_factor = 0.7
        self.min_distance = 1e-6
        self.frame_count = 0
        self.skip_frames = 2  # Process every nth frame
        self.face_scale = 0.5  # Increased scale for better detection
        self.last_face = None
        self.debug_frame = None
        
    def calibrate(self, frame):
        if len(self.calibration_points) >= 9:
            self.is_calibrated = True
            return True
            
        points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        
        current_point = points[len(self.calibration_points)]
        screen_x = int(current_point[0] * self.screen_width)
        screen_y = int(current_point[1] * self.screen_height)
        
        cv2.circle(frame, (screen_x, screen_y), 20, (0, 255, 0), -1)
        cv2.putText(frame, f"Look at the green dot ({len(self.calibration_points) + 1}/9)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        eyes = self.detect_eyes(frame)
        if eyes is not None:
            self.calibration_data.append((eyes, (screen_x, screen_y)))
            self.calibration_points.append(current_point)
            return True
        return False
    
    def detect_face(self, frame):
        """Optimized face detection"""
        if self.frame_count % self.skip_frames != 0 and self.last_face is not None:
            return self.last_face
            
        # Create debug frame
        self.debug_frame = frame.copy()
        
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect faces with more lenient parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Reduced for finer detection
            minNeighbors=2,    # More lenient
            minSize=(80, 80),  # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            cv2.putText(self.debug_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None
            
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Draw face rectangle
        cv2.rectangle(self.debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(self.debug_frame, "Face", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        self.last_face = (x, y, w, h)
        return self.last_face
    
    def detect_eyes(self, frame):
        """Improved eye detection"""
        face = self.detect_face(frame)
        if face is None:
            return None
            
        x, y, w, h = face
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale and enhance contrast
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.equalizeHist(gray_roi)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Detect eyes with more lenient parameters
        eyes = eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.01,  # Smaller scale factor for more detections
            minNeighbors=1,    # More lenient
            minSize=(int(w*0.1), int(h*0.1)),  # Minimum size relative to face
            maxSize=(int(w*0.4), int(h*0.4))   # Maximum size relative to face
        )
        
        if len(eyes) < 2:
            cv2.putText(self.debug_frame, f"Found {len(eyes)} eyes", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None
            
        # Filter and sort eyes
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Check vertical position (in upper 70% of face)
            if ey > h * 0.7:
                continue
                
            # Check size (between 10% and 30% of face width)
            if ew < w * 0.1 or ew > w * 0.3:
                continue
                
            valid_eyes.append((ex, ey, ew, eh))
            
            # Draw eye rectangles on debug frame
            abs_x = x + ex
            abs_y = y + ey
            cv2.rectangle(self.debug_frame, (abs_x, abs_y),
                         (abs_x+ew, abs_y+eh), (0, 255, 0), 2)
            cv2.putText(self.debug_frame, "Eye", (abs_x, abs_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(valid_eyes) < 2:
            cv2.putText(self.debug_frame, f"Found {len(valid_eyes)} valid eyes", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None
            
        # Sort eyes by x-coordinate (left to right)
        valid_eyes = sorted(valid_eyes, key=lambda x: x[0])[:2]
        
        # Check eye distance
        eye1, eye2 = valid_eyes
        distance = abs(eye1[0] - eye2[0])
        if distance < w * 0.15:  # Reduced minimum distance
            cv2.putText(self.debug_frame, "Eyes too close", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return None
            
        # Return eye positions relative to frame
        return [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in valid_eyes]
    
    def estimate_gaze(self, eyes):
        """Estimate gaze point using calibrated data"""
        if not self.is_calibrated or len(self.calibration_data) < 9:
            return None
            
        try:
            # Calculate eye centers
            eye_centers = [(x + w//2, y + h//2) for x, y, w, h in eyes]
            
            # Draw eye centers on debug frame
            for center in eye_centers:
                cv2.circle(self.debug_frame, center, 3, (0, 0, 255), -1)
            
            # Find closest calibration points
            distances = []
            for cal_eyes, cal_point in self.calibration_data:
                cal_centers = [(x + w//2, y + h//2) for x, y, w, h in cal_eyes]
                dist = np.mean([np.linalg.norm(np.array(ec) - np.array(cc)) 
                              for ec, cc in zip(eye_centers, cal_centers)])
                dist = max(dist, self.min_distance)
                distances.append((dist, cal_point))
            
            # Use weighted average of closest points
            distances.sort(key=lambda x: x[0])
            weights = [1/d for d, _ in distances[:3]]
            total_weight = sum(weights)
            if total_weight == 0:
                return self.last_valid_gaze
            weights = [w/total_weight for w in weights]
            
            x = sum(w * p[0] for w, (_, p) in zip(weights, distances[:3]))
            y = sum(w * p[1] for w, (_, p) in zip(weights, distances[:3]))
            
            # Apply smoothing
            gaze_point = (int(x), int(y))
            if self.last_valid_gaze is not None:
                gaze_point = (
                    int(self.smoothing_factor * self.last_valid_gaze[0] + (1 - self.smoothing_factor) * x),
                    int(self.smoothing_factor * self.last_valid_gaze[1] + (1 - self.smoothing_factor) * y)
                )
            
            # Draw estimated gaze point on debug frame
            cv2.putText(self.debug_frame, "Gaze point", gaze_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            self.last_valid_gaze = gaze_point
            return gaze_point
            
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            return self.last_valid_gaze

def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5  # Reduced for better pupil detection
    params.maxArea = 1000  # Reduced to avoid false positives
    params.filterByCircularity = True
    params.minCircularity = 0.2  # More lenient for pupil shape
    params.filterByConvexity = True
    params.minConvexity = 0.5
    return cv2.SimpleBlobDetector_create(params)

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # More lenient face detection
    coords = cascade.detectMultiScale(gray_frame, 1.1, 3, minSize=(100, 100))
    if len(coords) == 0:
        return None
    biggest = max(coords, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = biggest
    frame = img[y:y+h, x:x+w]
    return frame, (x, y, w, h)

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    
    # Get face dimensions for relative sizing
    face_height, face_width = img.shape[:2]
    
    # Improved eye detection with better parameters
    eyes = cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
    
    # Filter eyes based on size and position
    valid_eyes = []
    for (x, y, w, h) in eyes:
        # Check if eye is in upper half of face (more lenient)
        if y > face_height * 0.85:  # Increased from 0.8 to 0.85
            continue
            
        # Check if eye size is reasonable (between 10% and 35% of face width)
        if w < face_width * 0.1 or w > face_width * 0.35:
            continue
            
        valid_eyes.append((x, y, w, h))
    
    # Sort by y-coordinate and take top 2 eyes
    valid_eyes = sorted(valid_eyes, key=lambda x: x[1])[:2]
    
    # Improved eye distance check
    if len(valid_eyes) == 2:
        eye1, eye2 = valid_eyes
        distance = abs(eye1[0] - eye2[0])
        # Eyes should be at least 20% of face width apart, but not more than 45%
        if distance < face_width * 0.2 or distance > face_width * 0.45:
            valid_eyes = []
    
    # Create blob detector with parameters specific to each eye
    detector = create_blob_detector()
    
    for (x, y, w, h) in valid_eyes:
        detected_eye = img[y:y + h, x:x + w]
        detected_eye_pos = (x, y, w, h)
        detected_eye = cut_eyebrows(detected_eye)
        detected_eye_gray = cv2.cvtColor(detected_eye, cv2.COLOR_BGR2GRAY)
        detected_eye_gray = cv2.equalizeHist(detected_eye_gray)
        detected_eye_gray = cv2.GaussianBlur(detected_eye_gray, (5, 5), 0)
        
        # Method 1: Adaptive thresholding (more reliable)
        threshold = cv2.adaptiveThreshold(detected_eye_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        threshold = cv2.erode(threshold, None, iterations=1)
        threshold = cv2.dilate(threshold, None, iterations=2)
        threshold = cv2.medianBlur(threshold, 3)
        
        # Find contours in thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # Check if contour area is reasonable (between 1% and 15% of eye area)
            if area > w * h * 0.01 and area < w * h * 0.15:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                    eye_center = (w // 2, h // 2)
                    direction = np.array(center) - np.array(eye_center)
                    
                    # Draw pupil center with crosshair
                    cv2.circle(detected_eye, center, 2, (0, 0, 255), -1)
                    cv2.line(detected_eye, (center[0]-5, center[1]), (center[0]+5, center[1]), (0, 0, 255), 1)
                    cv2.line(detected_eye, (center[0], center[1]-5), (center[0], center[1]+5), (0, 0, 255), 1)
                    
                    return detected_eye, direction
        
        # Method 2: Blob detection (fallback)
        keypoints = detector.detect(detected_eye_gray)
        if keypoints:
            detected_eye, direction = draw_gaze_direction(detected_eye, detected_eye_pos, keypoints)
            if direction is not None:
                return detected_eye, direction
                
    return None, None

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)  # Adjusted to 1/4 to keep more of the eye region
    img = img[eyebrow_h:height, 0:width]
    return img

def draw_gaze_direction(img, eye_pos, keypoints):
    if keypoints:
        # Sort keypoints by size and take the largest one
        keypoints = sorted(keypoints, key=lambda x: x.size, reverse=True)
        for kp in keypoints:
            center = (int(kp.pt[0]), int(kp.pt[1]))
            eye_center = (eye_pos[2] // 2, eye_pos[3] // 2)
            direction = np.array(center) - np.array(eye_center)
            # Draw pupil center with crosshair
            cv2.circle(img, center, 2, (0, 0, 255), -1)
            cv2.line(img, (center[0]-5, center[1]), (center[0]+5, center[1]), (0, 0, 255), 1)
            cv2.line(img, (center[0], center[1]-5), (center[0], center[1]+5), (0, 0, 255), 1)
            return img, direction
    return img, None

def gaze_to_screen_coords(direction, screen_width, screen_height):
    sensitivity = 50  # Reduced sensitivity for more stable tracking
    x, y = direction
    screen_x = np.clip(screen_width / 2 + x * sensitivity, 0, screen_width - 1)
    screen_y = np.clip(screen_height // 2 + y * sensitivity, 0, screen_height - 1)
    return int(screen_x), int(screen_y)

def generate_heatmap(gaze_points, screen_width, screen_height): 
    heatmap, xedges, yedges = np.histogram2d(
        [p[0] for p in gaze_points],
        [p[1] for p in gaze_points],
        bins=[screen_width // 20, screen_height // 20],
        range=[[0, screen_width], [0, screen_height]]
    )

    heatmap = gaussian_filter(heatmap, sigma=10)

    extent = [0, screen_width, screen_height, 0]
    plt.imshow(heatmap.T, extent=extent, cmap='jet', alpha=0.7)

    plt.colorbar(label='Gaze Density')
    plt.title('Eye Gaze Heatmap')
    plt.xlabel('Screen Width (pixels)')
    plt.ylabel('Screen Height (pixels)')
    plt.gca().invert_yaxis()
    plt.show()

def main():
    tracker = GazeTracker()
    vs = WebcamVideoStream(src=0).start()
    gaze_points = []
    gaze_trail = []
    trail_length = 50
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    cv2.namedWindow('Eye Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Eye Tracker', tracker.screen_width, tracker.screen_height)
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
            
        frame = cv2.flip(frame, 1)
        tracker.frame_count += 1
        frame_count += 1
        
        # Calculate FPS
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        if not tracker.is_calibrated:
            if tracker.calibrate(frame):
                cv2.waitKey(1000)
        else:
            eyes = tracker.detect_eyes(frame)
            if eyes is not None:
                gaze_point = tracker.estimate_gaze(eyes)
                if gaze_point is not None:
                    gaze_points.append(gaze_point)
                    gaze_trail.append(gaze_point)
                    if len(gaze_trail) > trail_length:
                        gaze_trail.pop(0)
                    
                    # Draw gaze visualization
                    for i, point in enumerate(gaze_trail):
                        alpha = i / len(gaze_trail)
                        color = (0, 0, int(255 * alpha))
                        cv2.circle(frame, point, 12, (255, 255, 255), 2)
                        cv2.circle(frame, point, 10, color, -1)
                    
                    cv2.circle(frame, gaze_point, 35, (255, 255, 255), 2)
                    cv2.circle(frame, gaze_point, 30, (0, 0, 0), 2)
                    cv2.circle(frame, gaze_point, 25, (0, 0, 255), -1)
                    
                    text = f"X: {gaze_point[0]}, Y: {gaze_point[1]}"
                    cv2.putText(frame, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        
        # Display FPS and debug info
        cv2.putText(frame, f"FPS: {fps}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if tracker.debug_frame is not None:
            debug_frame = cv2.resize(tracker.debug_frame, (640, 480))
            cv2.imshow('Debug View', debug_frame)
        
        cv2.imshow('Eye Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vs.stop()
    cv2.destroyAllWindows()
    
    if gaze_points:
        generate_heatmap(gaze_points, tracker.screen_width, tracker.screen_height)

def process_static_image(image_path, output_dir='processed_images'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Get image dimensions
    height, width = frame.shape[:2]
    
    # Process the image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.05, 2, minSize=(80, 80))
    
    # Draw rectangles around all detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    result = detect_faces(frame, face_cascade)
    if result:
        face_frame, face_pos = result
        detected_eye, direction = detect_eyes(face_frame, eye_cascade)
        
        if direction is not None:
            # Calculate screen coordinates
            screen_coords = gaze_to_screen_coords(direction, width, height)
            
            # Draw gaze point
            cv2.circle(frame, screen_coords, 35, (255, 255, 255), 2)
            cv2.circle(frame, screen_coords, 30, (0, 0, 0), 2)
            cv2.circle(frame, screen_coords, 25, (0, 0, 255), -1)
            
            # Add coordinates text
            text = f"X: {screen_coords[0]}, Y: {screen_coords[1]}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(frame, (5, 40), (text_width + 15, text_height + 50), (255, 255, 255), -1)
            cv2.rectangle(frame, (5, 40), (text_width + 15, text_height + 50), (0, 0, 0), 2)
            
            # Draw text
            cv2.putText(frame, text, (10, 80), font, font_scale, (0, 0, 0), thickness)
            
            # Add direction info
            cv2.putText(frame, f"Direction: ({direction[0]:.1f}, {direction[1]:.1f})", 
                      (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Eyes not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Face not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save the processed image
    output_path = os.path.join(output_dir, f"processed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, frame)
    
    # Show the image
    cv2.imshow('Static Eye Tracking', frame)
    return frame

def test_static_images():
    # Get all images from the images directory
    image_dir = 'images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the images directory")
        return
    
    current_image = 0
    while True:
        # Process current image
        image_path = os.path.join(image_dir, image_files[current_image])
        frame = process_static_image(image_path)
        
        # Show navigation instructions
        print(f"\nImage {current_image + 1}/{len(image_files)}: {image_files[current_image]}")
        print("Press 'n' for next image")
        print("Press 'p' for previous image")
        print("Press 'q' to quit")
        
        # Handle key presses
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_image = (current_image + 1) % len(image_files)
        elif key == ord('p'):
            current_image = (current_image - 1) % len(image_files)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Ask user whether to run live tracking or static testing
    print("Choose mode:")
    print("1. Live eye tracking")
    print("2. Static image testing")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        main()
    elif choice == '2':
        test_static_images()
    else:
        print("Invalid choice. Please enter 1 or 2.")