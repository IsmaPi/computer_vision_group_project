import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.minArea = 30
detector_params.maxArea = 500
detector = cv2.SimpleBlobDetector_create(detector_params)

# Calibration points (corners of the screen)
CALIBRATION_POINTS = [
    (0, 0),           # Top-left
    (1920, 0),        # Top-right
    (1920, 1080),     # Bottom-right
    (0, 1080)         # Bottom-left
]

def create_calibration_screen(point_idx, screen_width, screen_height):
    """Create a calibration screen with a marker at the specified point."""
    screen = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    point = CALIBRATION_POINTS[point_idx]
    # Draw a large red circle at the calibration point
    cv2.circle(screen, point, 50, (0, 0, 255), -1)
    # Draw a white border around the circle
    cv2.circle(screen, point, 52, (255, 255, 255), 2)
    return screen

def calibrate(cap, screen_width, screen_height):
    """Perform calibration by collecting gaze samples for each corner."""
    calibration_data = []
    print("\n=== Starting Calibration ===")
    print("Look at each red dot in sequence when prompted.")
    print("Press 'Space' when you're looking at the dot.")
    print("Press 'q' to quit calibration.\n")
    
    for i, point in enumerate(CALIBRATION_POINTS):
        print(f"\nLook at the red dot in the {['top-left', 'top-right', 'bottom-right', 'bottom-left'][i]} corner.")
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
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) == 0:
        return None
    biggest = max(coords, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = biggest
    frame = img[y:y+h, x:x+w]
    return frame, (x, y, w, h)

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    eyes = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    eyes = sorted(eyes, key=lambda x: x[1])[:2]  # select top 2 eyes
    detected_eyes = []
    for (x, y, w, h) in eyes:
        detected_eye = img[y:y + h, x:x + w]
        detected_eye_pos = (x, y, w, h)
        detected_eye = cut_eyebrows(detected_eye)
        detected_eye_gray = cv2.cvtColor(detected_eye, cv2.COLOR_BGR2GRAY)
        detected_eye_gray = cv2.equalizeHist(detected_eye_gray)
        detected_eye_gray = cv2.GaussianBlur(detected_eye_gray, (7, 7), 0)
        detected_eye_keypoints = detector.detect(detected_eye_gray)
        if len(detected_eye_gray) > 0:
            detected_eye, direction = draw_gaze_direction(detected_eye, detected_eye_pos, detected_eye_keypoints)
            if direction is not None:
                return detected_eye, direction
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
    if keypoints:
        for kp in keypoints:
            center = (int(kp.pt[0]), int(kp.pt[1]))
            eye_center = (eye_pos[2] // 2, eye_pos[3] // 2)
            direction = np.array(center) - np.array(eye_center)
            return img, direction
    return img, None

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

def main():
    cap = cv2.VideoCapture(0)
    gaze_points = []
    screen_width, screen_height = 1920, 1080
    
    # Perform calibration first
    calibration_data = calibrate(cap, screen_width, screen_height)
    if calibration_data is None:
        print("Calibration failed. Exiting...")
        return
    
    print("\nCalibration completed successfully!")
    print("Starting eye tracking...")
    print("Press 'q' to quit.")
    
    threshold = 45
    duration = 15  # seconds
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
            
        result = detect_faces(frame, face_cascade)
        if result:
            face_frame, face_pos = result
            detected_eye, direction = detect_eyes(face_frame, eye_cascade)
            if direction is not None:
                screen_coords = map_gaze_to_screen(direction, calibration_data, screen_width, screen_height)
                if screen_coords:
                    gaze_points.append(screen_coords)
                    # Draw gaze point on frame
                    cv2.circle(frame, (face_pos[0], face_pos[1]), 5, (0, 255, 0), 2)
        
        cv2.imshow('Eye Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if gaze_points:
        generate_heatmap(gaze_points, screen_width, screen_height)

if __name__ == '__main__':
    main()