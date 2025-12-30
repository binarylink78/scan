import base64
from io import BytesIO
import cv2
from matplotlib import patches, pyplot as plt
import numpy as np
import os
import face_recognition
from PIL import Image
from flask import jsonify
import logging

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def get_dominant_color(hsv_image, n_clusters=3):
    """Extract dominant color using K-means clustering"""
    pixels = hsv_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
    cluster_counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(cluster_counts)]
    return dominant_color

def color_distance(color1, color2):
    """Calculate Euclidean distance between colors"""
    return np.linalg.norm(color1 - color2)

def detect_color_tampering(passport_img, face_img):
    """Detect potential color tampering by comparing passport background and face color"""
    passport_hsv = cv2.cvtColor(passport_img, cv2.COLOR_BGR2HSV)
    face_hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    
    passport_color = get_dominant_color(passport_hsv, n_clusters=3)
    face_color = get_dominant_color(face_hsv, n_clusters=3)
    
    color_diff = color_distance(passport_color, face_color)
    passport_variance = np.var(passport_hsv, axis=(0,1)).mean()
    face_variance = np.var(face_hsv, axis=(0,1)).mean()
    variance_diff = abs(passport_variance - face_variance)
    
    # Adjusted thresholds
    color_threshold = 75  # Previously 50
    variance_threshold = 30  # Previously 20
    
    tampering_score = 0
    if color_diff > color_threshold:
        tampering_score += 1
    if variance_diff > variance_threshold:
        tampering_score += 1
    
    return {
        'is_tampered': tampering_score > 1,  # Require stronger evidence
        'color_difference': float(color_diff),
        'color_variance_difference': float(variance_diff),
        'tampering_score': tampering_score
    }

def crop_face_with_color_analysis(temp_file_path, save_folder):
    """
    Crop face and perform color-based tampering detection
    """
    # Read the original passport image
    passport_img = cv2.imread(temp_file_path)
    
    # Convert to RGB for face recognition
    rgb_image = cv2.cvtColor(passport_img, cv2.COLOR_BGR2RGB)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_image)
    
    if face_locations:
        top, right, bottom, left = face_locations[0]
        
        # Expand face region with margin
        top = max(0, top - 30)
        left = max(0, left - 30)
        bottom = min(passport_img.shape[0], bottom + 30)
        right = min(passport_img.shape[1], right + 30)
        
        # Crop face region
        face_img = passport_img[top:bottom, left:right]
        
        # Perform color tampering detection
        color_analysis = detect_color_tampering(passport_img, face_img)
        
        # Save cropped face
        cropped_image_path = os.path.join(save_folder, "cropped_face.jpg")
        cv2.imwrite(cropped_image_path, face_img)
        
        return cropped_image_path, color_analysis
    
    raise Exception("No face detected in the image")

def detect_pixelation(image_path):
    """
    Detect pixelation levels.
    Higher pixel score indicates screen capture.
    Returns pixel score to aid in classification.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    pixel_score = np.mean(edges)
    print(f"Pixel Score: {pixel_score}")  # Debugging line to check pixelation score
    return pixel_score

def detect_moire(image_path):
    """Detect Moiré patterns that indicate printed paper."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    mean_magnitude = np.mean(magnitude_spectrum)
    print(f"Moiré Magnitude: {mean_magnitude}")  # Debugging line to check Moiré detection
    return mean_magnitude

def detect_black_white(image_path):
    """Detect if image is black and white printed."""
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    print(f"Average Saturation: {avg_saturation}")  # Debugging line to check saturation level
    return avg_saturation

def classify_image(image_path):
    """
    Classify if image is from screen or printed paper.
    Returns (is_screen, is_paper) as integers (0 or 1).
    """
    # Detect pixel score to evaluate pixelation (screen indicator)
    pixel_score = detect_pixelation(image_path)
    is_screen = int(pixel_score > 120)  # Adjusted pixelation threshold
    print(f"Is Screen: {is_screen}")  # Debugging line to check screen classification

    # Check for printed paper indicators (Moire patterns or black/white image)
    moire_pattern = detect_moire(image_path)
    is_moire = int(moire_pattern > 150)  # Adjust threshold to reduce Moiré false positives
    print(f"Is Moiré: {is_moire}")  # Debugging line to check Moiré detection
    bw_saturation = detect_black_white(image_path)
    is_bw = int(bw_saturation < 40)  # Adjust saturation threshold for better color detection
    print(f"Is B/W: {is_bw}")  # Debugging line to check black and white detection
    
    # Determine if the image is paper (based on Moiré or black-and-white detection)
    is_paper = int((is_moire or is_bw) and not is_screen)
    print(f"Is Paper: {is_paper}")  # Debugging line to check paper classification

    # If pixelation is low but Moiré or saturation suggests a screen capture, classify as screen
    if not is_screen and (is_moire or is_bw):
        is_screen = 1  # Force screen classification if Moiré or BW detected
        is_paper = 0  # This is not a printed paper

    return is_screen, is_paper


def base64_to_file(base64_data, filename):
    img_data = base64.b64decode(base64_data)
    img = BytesIO(img_data)
    img.name = filename
    return img

            
            
def correct_image_orientation(image_path):
    """Fix the image orientation based on EXIF data."""
    img = Image.open(image_path)

    try:
        # Check for EXIF data
        exif = img._getexif()

        if exif is not None:
            # Loop through EXIF tags to find the orientation tag
            for tag, value in exif.items():
                if tag == 274:  # 274 is the orientation tag in EXIF
                    if value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 6:
                        img = img.rotate(270, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)

        # Return the image with corrected orientation
        return img

    except (AttributeError, KeyError, IndexError):
        # If no EXIF data is found, just return the image as is
        return img

            

def resize_image(image, target_width=800):
    img = Image.open(image)
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    img = img.resize((target_width, target_height))
    return img

def show_image_with_faces(image, face_locations):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for (top, right, bottom, left) in face_locations:
        rect = patches.Rectangle((left, top), right - left, bottom - top,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def crop_face(image_path, save_folder, size=(200, 200), margin=30):
   
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Unable to load image: {image_path}")
    
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_image)
    
    logging.info(f"Number of faces detected: {len(face_locations)}")
    
    if face_locations:
        top, right, bottom, left = face_locations[0]
        
        top = max(10, top - margin) 
        left = max(10, left - margin) 
        bottom = min(img.shape[0], bottom + margin)
        right = min(img.shape[1], right + margin)
        
        face_image = img[top:bottom, left:right]
        
        face_image_resized = cv2.resize(face_image, size)
        
        cropped_image_name = "cropped_face.jpg"
        cropped_image_path = os.path.join(save_folder, cropped_image_name)
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        success = cv2.imwrite(cropped_image_path, face_image_resized)
        
        if not success:
            raise Exception(f"Failed to save the cropped face image at {cropped_image_path}")
        
        logging.info(f"Cropped face image saved at {cropped_image_path}")
        return cropped_image_path
    else:
        raise Exception("No face detected in the image.")
