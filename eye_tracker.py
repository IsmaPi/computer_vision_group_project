import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time
from threading import Thread
from queue import Queue
from tracker import EyeTracker
from video_stream import WebcamVideoStream
from visualization import draw_gaze_point, draw_heatmap, draw_status, draw_calibration_grid
from utils import create_output_dir, save_processed_image, get_image_files, load_image, resize_image
import os
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Improved blob detector parameters
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.minArea = 20  # Reduced minimum area for better detection
detector_params.maxArea = 600  # Increased maximum area
detector_params.filterByCircularity = True
detector_params.minCircularity = 0.3  # More lenient circularity threshold
detector_params.filterByConvexity = True
detector_params.minConvexity = 0.5  # More lenient convexity threshold
detector = cv2.SimpleBlobDetector_create(detector_params)

# Calibration points (corners of the screen)
CALIBRATION_POINTS = [
    (0, 0),           # Top-left
    (1920, 0),        # Top-right
    (1920, 1080),     # Bottom-right
    (0, 1080),        # Bottom-left
    (960, 540)        # Center point for better accuracy
]

# Check if we're in a GUI environment
def check_display():
    try:
        # Try to create a test window
        test_window = cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test')
        return True
    except:
        return False

# Global display flag
HAS_DISPLAY = check_display()

def show_frame(title, frame):
    """Safely show a frame, with fallback for non-GUI environments"""
    if HAS_DISPLAY:
        cv2.imshow(title, frame)
    else:
        print(f"Warning: Cannot display window '{title}'. No GUI environment detected.")
        # Save frame to file instead
        output_dir = create_output_dir()
        save_processed_image(frame, output_dir)

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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
            if self.frame is not None:
                self.frame = cv2.flip(self.frame, 1)  # Mirror the frame
            
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True

def create_calibration_screen(point_idx, screen_width, screen_height):
    """Create a calibration screen with a marker at the specified point."""
    screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    point = CALIBRATION_POINTS[point_idx]
    
    # Draw a pulsing circle animation
    radius = 50 + int(10 * np.sin(time.time() * 4))  # Pulsing effect
    
    # Draw outer guide circles
    cv2.circle(screen, point, radius + 20, (200, 200, 200), 2)
    cv2.circle(screen, point, radius + 10, (150, 150, 150), 2)
    
    # Draw main target circle
    cv2.circle(screen, point, radius, (0, 255, 0), -1)  # Green dot
    cv2.circle(screen, point, radius + 2, (255, 255, 255), 2)
    
    # Add instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(screen, "Look at the green dot and press SPACE when ready", 
                (screen_width//2 - 300, 50), font, 1, (0, 0, 0), 2)
    cv2.putText(screen, f"Calibration point {point_idx + 1}/{len(CALIBRATION_POINTS)}", 
                (screen_width//2 - 150, 100), font, 1, (0, 0, 0), 2)
    
    return screen

def calibrate(cap, screen_width, screen_height):
    """Perform calibration by collecting gaze samples for each corner."""
    calibration_data = []
    print("\n=== Starting Calibration ===")
    print("Look at each green dot in sequence when prompted.")
    print("Press 'Space' when you're looking at the dot.")
    print("Press 'q' to quit calibration.\n")
    
    for i, point in enumerate(CALIBRATION_POINTS):
        print(f"\nLook at the green dot in the {['top-left', 'top-right', 'bottom-right', 'bottom-left', 'center'][i]} corner.")
        print("Press 'Space' when ready, then look at the dot for 2 seconds.")
        
        # Show calibration screen
        calibration_screen = create_calibration_screen(i, screen_width, screen_height)
        cv2.imshow('Calibration', calibration_screen)
        
        # Wait for space key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        # Collect gaze samples for 2 seconds
        samples = []
        start_time = time.time()
        while time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret:
                continue
                
            result = detect_faces(frame, face_cascade)
            if result:
                face_frame, face_pos = result
                detected_eye, direction = detect_eyes(face_frame, eye_cascade)
                if direction is not None:
                    samples.append(direction)
            
            # Show live feed with calibration screen overlay
            overlay = frame.copy()
            cv2.addWeighted(calibration_screen, 0.3, overlay, 0.7, 0, overlay)
            cv2.imshow('Calibration', overlay)
            cv2.waitKey(1)
        
        if samples:
            # Calculate average gaze direction for this point
            avg_direction = np.mean(samples, axis=0)
            calibration_data.append((point, avg_direction))
            print(f"Calibration point {i+1} recorded successfully!")
        else:
            print(f"Warning: No valid samples collected for point {i+1}")
    
    cv2.destroyAllWindows()
    return calibration_data

def map_gaze_to_screen(direction, calibration_data, screen_width, screen_height):
    """Map gaze direction to screen coordinates using calibration data."""
    if not calibration_data:
        return None
    
    # Convert calibration data to numpy arrays for easier computation
    screen_points = np.array([p[0] for p in calibration_data])
    gaze_directions = np.array([p[1] for p in calibration_data])
    
    # Find the closest calibration points
    distances = np.linalg.norm(gaze_directions - direction, axis=1)
    closest_indices = np.argsort(distances)[:2]
    
    # Interpolate between the two closest points
    weights = 1 / (distances[closest_indices] + 1e-6)
    weights = weights / np.sum(weights)
    
    screen_coords = np.sum(screen_points[closest_indices] * weights[:, np.newaxis], axis=0)
    
    # Ensure coordinates are within screen bounds
    screen_coords = np.clip(screen_coords, [0, 0], [screen_width, screen_height])
    
    return tuple(map(int, screen_coords))

def detect_faces(img, cascade):
    """Enhanced face detection with better preprocessing"""
    # Create a copy for debug visualization
    debug_img = img.copy()
    
    # Convert to grayscale and enhance contrast
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Detect faces with optimized parameters
    coords = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(coords) == 0:
        cv2.putText(debug_img, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return None
    
    # Get the largest face
    biggest = max(coords, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = biggest
    
    # Draw face rectangle on debug image
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(debug_img, "Face", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    frame = img[y:y+h, x:x+w]
    return frame, (x, y, w, h)

def detect_eyes(img, cascade):
    """Improved eye detection with better filtering"""
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Detect eyes with optimized parameters
    eyes = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(25, 25),
        maxSize=(80, 80)
    )
    
    if len(eyes) < 2:
        return None, None
    
    # Sort eyes vertically and take top 2
    eyes = sorted(eyes, key=lambda x: x[1])[:2]
    
    # Sort horizontally to ensure left-to-right order
    eyes = sorted(eyes, key=lambda x: x[0])
    
    detected_eyes = []
    for (x, y, w, h) in eyes:
        eye_roi = img[y:y + h, x:x + w]
        eye_roi = cut_eyebrows(eye_roi)
        
        # Enhanced preprocessing
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        eye_gray = cv2.equalizeHist(eye_gray)
        eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)
        
        # Adaptive thresholding for better pupil detection
        eye_gray = cv2.adaptiveThreshold(
            eye_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        keypoints = detector.detect(eye_gray)
        
        if len(keypoints) > 0:
            eye_roi, direction = draw_gaze_direction(eye_roi, (x, y, w, h), keypoints)
            if direction is not None:
                return eye_roi, direction
    
    return None, None

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    return img

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

def draw_gaze_direction(img, eye_pos, keypoints):
    """Enhanced gaze direction visualization"""
    if not keypoints:
        return img, None
    
    # Get the largest keypoint (likely the pupil)
    largest_kp = max(keypoints, key=lambda kp: kp.size)
    
    center = (int(largest_kp.pt[0]), int(largest_kp.pt[1]))
    eye_center = (eye_pos[2] // 2, eye_pos[3] // 2)
    
    # Calculate normalized direction vector
    direction = np.array(center) - np.array(eye_center)
    magnitude = np.linalg.norm(direction)
    
    if magnitude < 1e-6:  # Avoid division by zero
        return img, None
    
    direction = direction / magnitude
    
    # Draw visualization
    cv2.circle(img, center, 3, (0, 255, 0), -1)  # Pupil center
    cv2.circle(img, eye_center, 3, (0, 0, 255), -1)  # Eye center
    
    # Draw direction line
    end_point = (
        int(eye_center[0] + direction[0] * 20),
        int(eye_center[1] + direction[1] * 20)
    )
    cv2.line(img, eye_center, end_point, (255, 0, 0), 2)
    
    return img, direction

def gaze_to_screen_coords(direction, screen_width, screen_height):
    sensitivity = 15
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

    extent = [0, screen_width, screen_height, 0]  # Flip Y-axis for correct orientation
    plt.imshow(heatmap.T, extent=extent, cmap='jet', alpha=0.7)

    plt.colorbar(label='Gaze Density')
    plt.title('Eye Gaze Heatmap')
    plt.xlabel('Screen Width (pixels)')
    plt.ylabel('Screen Height (pixels)')
    plt.gca().invert_yaxis()  # Optional, depending on your coordinate system
    plt.show()

def process_static_images():
    """Process static images from directory"""
    print("\nStatic image testing not implemented yet.")
    return

def live_tracking():
    """Live eye tracking with webcam"""
    # Initialize video stream
    cap = WebcamVideoStream(0).start()
    time.sleep(1.0)  # Give camera time to initialize
    
    # Get webcam dimensions
    frame = cap.read()
    if frame is None:
        print("Error: Could not initialize webcam")
        return
    screen_height, screen_width = frame.shape[:2]
    
    # Initialize tracker
    tracker = EyeTracker()
    gaze_points = []
    calibration_data = None
    debug_mode = True  # Start with debug mode on
    
    try:
        # Calibration phase
        print("\n=== Starting Calibration ===")
        print("Look at each green dot in sequence when prompted.")
        print("Press 'Space' when you're looking at the dot.")
        print("Press 'q' to quit calibration.\n")
        
        for i, point in enumerate(tracker.CALIBRATION_POINTS):
            print(f"\nLook at the green dot in the {['top-left', 'top-right', 'bottom-right', 'bottom-left', 'center'][i]} corner.")
            print("Press 'Space' when ready, then look at the dot for 2 seconds.")
            
            # Scale calibration point to webcam dimensions
            scaled_point = (
                int(point[0] * screen_width / 1920),
                int(point[1] * screen_height / 1080)
            )
            
            # Show calibration screen
            calibration_screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
            calibration_screen = draw_calibration_grid(calibration_screen, screen_width, screen_height)
            
            # Wait for space key
            while True:
                frame = cap.read()
                if frame is None:
                    continue
                
                # Debug visualization
                debug_frame = frame.copy()
                faces = tracker.detect_faces(debug_frame)
                for (x, y, w, h) in faces:
                    # Draw face rectangle
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_roi = debug_frame[y:y+h, x:x+w]
                    
                    # Detect and draw eyes
                    eyes = tracker.detect_eyes(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Show debug info
                cv2.putText(debug_frame, "Face and eye detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show live feed with calibration screen overlay
                overlay = debug_frame.copy()
                cv2.addWeighted(calibration_screen, 0.3, overlay, 0.7, 0, overlay)
                cv2.imshow('Calibration', overlay)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    cap.stop()
                    return
            
            # Collect gaze samples for 2 seconds
            samples = []
            start_time = time.time()
            while time.time() - start_time < 2:
                frame = cap.read()
                if frame is None:
                    continue
                
                # Debug visualization
                debug_frame = frame.copy()
                faces = tracker.detect_faces(debug_frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_roi = debug_frame[y:y+h, x:x+w]
                    eyes = tracker.detect_eyes(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                if tracker.calibrate(frame, scaled_point):
                    samples.append(scaled_point)
                    cv2.putText(debug_frame, "Gaze point recorded!", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show live feed with calibration screen overlay
                overlay = debug_frame.copy()
                cv2.addWeighted(calibration_screen, 0.3, overlay, 0.7, 0, overlay)
                cv2.imshow('Calibration', overlay)
                cv2.waitKey(1)
            
            if samples:
                print(f"Calibration point {i+1} recorded successfully!")
            else:
                print(f"Warning: No valid samples collected for point {i+1}")
        
        cv2.destroyAllWindows()
        
        # Live tracking phase
        print("\nCalibration complete! Starting live tracking...")
        print("\nControls:")
        print("D: Toggle debug visualization")
        print("H: Show heatmap")
        print("Q: Quit")
        
        while True:
            frame = cap.read()
            if frame is None:
                continue
            
            # Get gaze point
            gaze_point = tracker.estimate_gaze(frame)
            if gaze_point:
                gaze_points.append(gaze_point)
            
            # Create visualization frame
            if debug_mode:
                frame_to_show = frame.copy()
                faces = tracker.detect_faces(frame_to_show)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_to_show, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_roi = frame_to_show[y:y+h, x:x+w]
                    eyes = tracker.detect_eyes(face_roi)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame_to_show, "Debug Mode", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                frame_to_show = frame.copy()
            
            # Draw gaze point
            if gaze_point:
                frame_to_show = draw_gaze_point(frame_to_show, gaze_point)
            
            # Show frame
            cv2.imshow('Eye Tracking', frame_to_show)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
            elif key == ord('h') and gaze_points:
                draw_heatmap(gaze_points, screen_width, screen_height)
    
    except KeyboardInterrupt:
        print("\nTracking interrupted by user.")
    finally:
        # Cleanup
        cap.stop()
        cv2.destroyAllWindows()
        
        if gaze_points:
            print("\nGenerating final heatmap...")
            draw_heatmap(gaze_points, screen_width, screen_height)

def main():
    """Main entry point"""
    print("\nEye Tracking System")
    print("==================")
    print("1: Start Live Tracking")
    print("2: Test with Static Images")
    print("Q: Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == '1':
        live_tracking()
    elif choice == '2':
        process_static_images()
    elif choice == 'q':
        print("\nGoodbye!")
    else:
        print("\nInvalid choice. Goodbye!")

if __name__ == '__main__':
    main()