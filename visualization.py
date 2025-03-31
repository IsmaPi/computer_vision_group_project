import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def draw_gaze_point(frame, point, color=(0, 0, 255), radius=8):
    """Draw gaze point on frame with improved visibility"""
    if point is None:
        return frame
    
    # Draw outer glow
    for r in range(radius + 4, radius, -1):
        alpha = (r - radius) / 4
        glow_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, point, r, glow_color, -1)
    
    # Draw main point
    cv2.circle(frame, point, radius, color, -1)
    
    # Draw crosshair
    cross_size = radius * 2
    thickness = 2
    
    # Draw crosshair lines
    cv2.line(frame, (point[0] - cross_size, point[1]), (point[0] + cross_size, point[1]), color, thickness)
    cv2.line(frame, (point[0], point[1] - cross_size), (point[0], point[1] + cross_size), color, thickness)
    
    # Draw corner markers
    marker_size = radius
    cv2.line(frame, (point[0] - marker_size, point[1] - marker_size), (point[0] - marker_size, point[1]), color, thickness)
    cv2.line(frame, (point[0] - marker_size, point[1] - marker_size), (point[0], point[1] - marker_size), color, thickness)
    cv2.line(frame, (point[0] + marker_size, point[1] - marker_size), (point[0] + marker_size, point[1]), color, thickness)
    cv2.line(frame, (point[0] + marker_size, point[1] - marker_size), (point[0], point[1] - marker_size), color, thickness)
    cv2.line(frame, (point[0] - marker_size, point[1] + marker_size), (point[0] - marker_size, point[1]), color, thickness)
    cv2.line(frame, (point[0] - marker_size, point[1] + marker_size), (point[0], point[1] + marker_size), color, thickness)
    cv2.line(frame, (point[0] + marker_size, point[1] + marker_size), (point[0] + marker_size, point[1]), color, thickness)
    cv2.line(frame, (point[0] + marker_size, point[1] + marker_size), (point[0], point[1] + marker_size), color, thickness)
    
    return frame

def draw_status(frame, status_text, color=(0, 255, 0)):
    """Draw status text on the frame"""
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def draw_calibration_grid(frame, width, height):
    """Draw calibration grid with dots"""
    # Draw grid lines
    for x in range(0, width, width//4):
        cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
    for y in range(0, height, height//4):
        cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)
    
    # Draw calibration points
    points = [
        (0, 0),           # Top-left
        (width, 0),       # Top-right
        (width, height),  # Bottom-right
        (0, height),      # Bottom-left
        (width//2, height//2)  # Center
    ]
    
    for point in points:
        cv2.circle(frame, point, 10, (0, 255, 0), -1)
        cv2.circle(frame, point, 12, (0, 255, 0), 2)
    
    return frame

def draw_heatmap(points, width, height, alpha=0.5):
    """Draw heatmap of gaze points"""
    if not points:
        return
    
    # Create heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add points to heatmap with Gaussian distribution
    for point in points:
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            # Create Gaussian kernel
            kernel_size = 31
            sigma = 10
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            
            for i in range(kernel_size):
                for j in range(kernel_size):
                    dx = i - center
                    dy = j - center
                    kernel[i, j] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
            # Add kernel to heatmap
            y1 = max(0, y - center)
            y2 = min(height, y + center + 1)
            x1 = max(0, x - center)
            x2 = min(width, x + center + 1)
            
            k_y1 = max(0, center - y)
            k_y2 = min(kernel_size, center + (height - y))
            k_x1 = max(0, center - x)
            k_x2 = min(kernel_size, center + (width - x))
            
            heatmap[y1:y2, x1:x2] += kernel[k_y1:k_y2, k_x1:k_x2]
    
    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Create window and show heatmap
    cv2.namedWindow('Gaze Heatmap', cv2.WINDOW_NORMAL)
    cv2.imshow('Gaze Heatmap', heatmap)
    cv2.waitKey(1)  # Update the window 