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
                screen_coords = gaze_to_screen_coords(direction, screen_width, screen_height)
                gaze_points.append(screen_coords)
                cv2.circle(frame, (face_pos[0], face_pos[1]), 5, (0, 255, 0), 2)
        cv2.imshow('Eye Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if gaze_points:
        generate_heatmap(gaze_points, screen_width, screen_height)

if __name__ == '__main__':
    gaze_points = []
    screen_width, screen_height = 1920, 1080
    main()