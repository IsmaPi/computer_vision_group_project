import cv2
import numpy as np

class EyeTracker:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        # Blob detector parameters - more sensitive settings
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.detector_params.filterByArea = True
        self.detector_params.minArea = 10  # Reduced minimum area
        self.detector_params.maxArea = 1000  # Increased maximum area
        self.detector_params.filterByCircularity = True
        self.detector_params.minCircularity = 0.2  # More lenient circularity
        self.detector_params.filterByConvexity = True
        self.detector_params.minConvexity = 0.4  # More lenient convexity
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)
        
        # Calibration state
        self.waiting_for_space = False
        self.current_calibration_point = None
        self.calibration_data = []
        
        # Calibration points (corners of the screen)
        self.CALIBRATION_POINTS = [
            (0, 0),           # Top-left
            (1920, 0),        # Top-right
            (1920, 1080),     # Bottom-right
            (0, 1080),        # Bottom-left
            (960, 540)        # Center point for better accuracy
        ]
        
    def detect_faces(self, img):
        """Detect faces in the image with more sensitive parameters"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        # More sensitive face detection parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # More gradual scaling
            minNeighbors=3,   # Fewer neighbors required
            minSize=(60, 60), # Smaller minimum face size
            maxSize=(300, 300) # Larger maximum face size
        )
        return faces
    
    def detect_eyes(self, face_img):
        """Detect eyes in the face region with more sensitive parameters"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Get face dimensions
        height, width = face_img.shape[:2]
        
        # Define the upper region of interest (top 60% of face)
        roi_height = int(height * 0.6)
        roi_gray = gray[0:roi_height, :]
        
        # More sensitive eye detection parameters
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,  # More gradual scaling
            minNeighbors=4,    # Slightly more neighbors for better accuracy
            minSize=(int(width * 0.1), int(height * 0.1)),  # Relative to face size
            maxSize=(int(width * 0.3), int(height * 0.3))   # Relative to face size
        )
        
        if len(eyes) < 2:
            return []
        
        # Filter eyes by vertical position (should be in upper half of face)
        eyes = [eye for eye in eyes if eye[1] < height/2]
        
        if len(eyes) < 2:
            return []
        
        # Sort eyes by x-coordinate (horizontal position)
        eyes = sorted(eyes, key=lambda x: x[0])
        
        # If we have more than 2 eyes, take the leftmost and rightmost ones
        if len(eyes) > 2:
            left_eye = min(eyes, key=lambda x: x[0])
            right_eye = max(eyes, key=lambda x: x[0])
            eyes = [left_eye, right_eye]
        
        # Ensure left eye is actually on the left side and right eye on the right
        if len(eyes) == 2:
            left_x = eyes[0][0]
            right_x = eyes[1][0]
            if abs(left_x - right_x) < width * 0.2:  # Eyes should be at least 20% of face width apart
                return []
            if left_x > width/2 or right_x < width/2:  # Left eye should be in left half, right eye in right half
                return []
        
        return eyes[:2]  # Return at most 2 eyes
    
    def process_eye(self, eye_img):
        """Process eye image to detect pupil with enhanced preprocessing"""
        # Resize for better processing
        eye_img = cv2.resize(eye_img, (60, 60))
        
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        threshold = cv2.erode(threshold, kernel, iterations=1)
        threshold = cv2.dilate(threshold, kernel, iterations=2)
        threshold = cv2.medianBlur(threshold, 3)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the largest contour
        mask = np.zeros_like(threshold)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply mask to threshold
        threshold = cv2.bitwise_and(threshold, threshold, mask=mask)
        
        keypoints = self.detector.detect(threshold)
        return keypoints
    
    def get_gaze_direction(self, eye_img, eye_pos):
        """Get gaze direction from eye image with improved accuracy"""
        keypoints = self.process_eye(eye_img)
        if not keypoints:
            return None
            
        # Get the largest keypoint (likely the pupil)
        largest_kp = max(keypoints, key=lambda kp: kp.size)
        center = (int(largest_kp.pt[0]), int(largest_kp.pt[1]))
        eye_center = (eye_pos[2] // 2, eye_pos[3] // 2)
        
        # Calculate normalized direction vector
        direction = np.array(center) - np.array(eye_center)
        magnitude = np.linalg.norm(direction)
        
        if magnitude < 1e-6:
            return None
            
        # Normalize direction
        direction = direction / magnitude
        
        # Add horizontal bias to account for natural eye movement
        direction[0] *= 1.5  # Increase horizontal sensitivity
        
        return direction
    
    def calibrate(self, frame, point):
        """Record calibration point when space is pressed"""
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return False
            
        # Get the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        
        # Get eyes
        eyes = self.detect_eyes(face_img)
        if len(eyes) != 2:  # Must have exactly 2 eyes
            return False
            
        # Process both eyes
        directions = []
        for eye in eyes:
            eye_img = face_img[eye[1]:eye[1]+eye[3], eye[0]:eye[0]+eye[2]]
            direction = self.get_gaze_direction(eye_img, eye)
            if direction is not None:
                directions.append(direction)
                
        if len(directions) == 2:
            # Average the directions from both eyes
            avg_direction = np.mean(directions, axis=0)
            
            # Store calibration data with timestamp
            self.calibration_data.append((point, avg_direction))
            
            # If we have multiple samples for this point, average them
            if len(self.calibration_data) > 1:
                # Group samples by point
                point_samples = [(p, d) for p, d in self.calibration_data if p == point]
                if len(point_samples) > 1:
                    # Calculate average direction for this point
                    avg_direction = np.mean([d for _, d in point_samples], axis=0)
                    # Remove old samples for this point
                    self.calibration_data = [(p, d) for p, d in self.calibration_data if p != point]
                    # Add averaged sample
                    self.calibration_data.append((point, avg_direction))
            
            return True
            
        return False
    
    def estimate_gaze(self, frame):
        """Estimate gaze point using calibration data with improved accuracy"""
        if not self.calibration_data:  # Must have calibration data
            return None
            
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return None
            
        # Get the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        
        # Get eyes
        eyes = self.detect_eyes(face_img)
        if len(eyes) != 2:  # Must have exactly 2 eyes
            return None
            
        # Process both eyes
        directions = []
        for eye in eyes:
            eye_img = face_img[eye[1]:eye[1]+eye[3], eye[0]:eye[0]+eye[2]]
            direction = self.get_gaze_direction(eye_img, eye)
            if direction is not None:
                directions.append(direction)
                
        if len(directions) == 2:
            # Average the directions from both eyes
            avg_direction = np.mean(directions, axis=0)
            
            # Convert calibration data to numpy arrays for easier computation
            screen_points = np.array([p[0] for p in self.calibration_data])
            gaze_directions = np.array([p[1] for p in self.calibration_data])
            
            # Find the closest calibration points
            distances = np.linalg.norm(gaze_directions - avg_direction, axis=1)
            closest_indices = np.argsort(distances)[:3]  # Get 3 closest points
            
            # Calculate weights based on inverse distance
            weights = 1 / (distances[closest_indices] + 1e-6)
            weights = weights / np.sum(weights)
            
            # Interpolate between the closest points
            screen_coords = np.sum(screen_points[closest_indices] * weights[:, np.newaxis], axis=0)
            
            # Add smoothing to reduce jitter
            if hasattr(self, 'last_coords'):
                alpha = 0.7  # Smoothing factor
                screen_coords = alpha * screen_coords + (1 - alpha) * np.array(self.last_coords)
            
            # Store current coordinates for next frame
            self.last_coords = screen_coords
            
            # Ensure coordinates are within screen bounds
            screen_coords = np.clip(screen_coords, [0, 0], [1920, 1080])
            
            return tuple(map(int, screen_coords))
            
        return None 