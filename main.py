from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os
from routes.passport_ocr import passport_ocr_bp
from routes.face_match import face_match_bp

load_dotenv()

app = Flask(__name__)
CORS(app)

#os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX', r'C:/Program Files/Tesseract-OCR/tessdata')
#pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'C:/Program Files/Tesseract-OCR/tesseract.exe')


# for linux
os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX', r'=/usr/share/tesseract-ocr/5/tessdata')
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'/usr/bin/tesseract')


fast_mrz = FastMRZ()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

CSV_FILE = 'mrz_data.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['document_number', 'document_type', 'given_name', 'surname', 'optional_data',
                             'country_code', 'date_of_birth', 'date_of_expiry', 'nationality', 'mrz_type',
                             'sex', 'status', 'extracted_at', 'image_path'])

def add_mrz_to_csv(mrz_data):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([mrz_data['document_number'],
                         mrz_data['document_code'],
                         mrz_data['given_name'],
                         mrz_data['surname'],
                         mrz_data['optional_data'],
                         mrz_data['issuer_code'],
                         mrz_data['birth_date'],
                         mrz_data['expiry_date'],
                         mrz_data['nationality_code'],
                         mrz_data['mrz_type'],
                         mrz_data['sex'],
                         mrz_data['status'],
                         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            None])

create_csv()

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

@app.route("/passport_ocr/extract-mrz/", methods=["POST"])
def extract_mrz():
    temp_file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({"detail": "No file part in the request."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"detail": "No file selected for uploading."}), 400

        if not allowed_file(file.filename):
            return jsonify({"detail": "Unsupported file type. Please upload an image."}), 415

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
            logger.info(f"File saved temporarily at {temp_file_path}")

        passport_mrz = fast_mrz.get_details(temp_file_path, include_checkdigit=False)
        if not passport_mrz:
            return jsonify({"detail": "Failed to extract MRZ from the uploaded file."}), 422

        add_mrz_to_csv(passport_mrz)

        logger.info("MRZ data extracted successfully.")

        surname = passport_mrz.get("surname", "")
        given_name = passport_mrz.get("given_name", "")

        folder_name = f"{given_name}_{surname}".lower().replace(" ", "_")
        save_folder = os.path.join("uploads", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        
        uploaded_image_path = os.path.join(save_folder, "uploaded_image.jpg")
        file.seek(0)
        with open(uploaded_image_path, "wb") as f:
            f.write(file.read())
        logger.info(f"Uploaded image saved at {uploaded_image_path}")

        try:
            # Use new face cropping method with color analysis
            cropped_image_path, color_analysis = crop_face_with_color_analysis(temp_file_path, save_folder)
            logger.info(f"Cropped face image saved at {cropped_image_path}")

            with open(cropped_image_path, "rb") as cropped_file:
                cropped_image_base64 = base64.b64encode(cropped_file.read()).decode('utf-8')
        
            # Check for color tampering
            if color_analysis['is_tampered']:
                logger.warning("Potential color tampering detected")
                return jsonify({
                    "status": "COLOR_TAMPERING_DETECTED",
                    "color_analysis": color_analysis,
                    "message": "Passport image shows signs of passport manipulation"
                }), 403

        except Exception as face_crop_error:
            logger.error(f"Failed to crop face: {str(face_crop_error)}", exc_info=True)
            cropped_image_base64 = None

        response_data = {
            "status": "SUCCESS",
            "result": {
                "mrz": passport_mrz,
                "passport_photo": cropped_image_base64,
                "color_analysis": color_analysis
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        # return jsonify({"status": "FAILURE", "message": str(e)}), 500
        return jsonify({"status": "FAILURE", "message": "Error processing file. Please try again."}), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
            

@app.route("/passport_ocr/face_match/", methods=["POST"])
def face_match():
    image1_path = None
    image2_path = None

    try:
       # Check if source_url is base64 or a file upload
        source_url = request.files.get("source_url") or request.form.get("source_url")
        target_url = request.files.get("target_url") or request.form.get("target_url")

        # Ensure both source_url and target_url are provided
        if not source_url:
            return jsonify({"status": "FAILURE", "message": "Source URL is NULL"}), 400

        if not target_url:
            return jsonify({"status": "FAILURE", "message": "Target URL is NULL"}), 400

        # Helper function to handle base64 string to file
        def base64_to_file(base64_data, filename):
            img_data = base64.b64decode(base64_data)
            img = BytesIO(img_data)
            img.name = filename
            return img

        # Check if source_url is base64 or file
        if isinstance(source_url, str) and source_url.startswith('data:image'):
            # It's base64
            source_file = base64_to_file(source_url.split(",")[1], "passport_photo.jpg")
        else:
            # It's a file
            source_file = source_url

        # Check if target_url is base64 or file
        if isinstance(target_url, str) and target_url.startswith('data:image'):
            # It's base64
            target_file = base64_to_file(target_url.split(",")[1], "selfie.jpg")
        else:
            # It's a file
            target_file = target_url

        # Save images temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp2:
            temp1.write(source_file.read())
            temp2.write(target_file.read())
            image1_path = temp1.name
            image2_path = temp2.name

        # Correct image orientation
        image1_corrected = correct_image_orientation(image1_path)
        image2_corrected = correct_image_orientation(image2_path)

        # Save corrected images back
        image1_corrected.save(image1_path)
        image2_corrected.save(image2_path)

        # Resize images before processing
        image1_resized = resize_image(image1_path)
        image2_resized = resize_image(image2_path)

        # Convert to numpy array for face_recognition
        image1 = np.array(image1_resized)
        image2 = np.array(image2_resized)

        # Detect faces
        face1_locations = face_recognition.face_locations(image1, model="hog", number_of_times_to_upsample=2)
        face2_locations = face_recognition.face_locations(image2, model="hog", number_of_times_to_upsample=2)

        if len(face1_locations) == 0 or len(face2_locations) == 0:
            return jsonify({
                "status": "FAILURE",
                "message": "No face detected in one or both images.",
                "datetime": datetime.now().isoformat()
            }), 422

        # Get face encodings
        face1_encodings = face_recognition.face_encodings(image1, face1_locations)
        face2_encodings = face_recognition.face_encodings(image2, face2_locations)

        if len(face1_encodings) == 0 or len(face2_encodings) == 0:
            return jsonify({"status": "FAILURE", "message": "No face encodings found."}), 422

        # Compare faces
        face1_encoding = face1_encodings[0]
        face2_encoding = face2_encodings[0]
        distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
        similarity_score = max(0, 1 - distance)

        # Generate response
        threshold = 0.5  
        if similarity_score >= threshold:
            return jsonify({
                "status": "SUCCESS",
                "message": "Faces matched successfully.",
                "similarity_score": similarity_score,
            }), 200
        else:
            return jsonify({
                "status": "FAILURE",
                "message": "Faces do not match. Please try again.",
                "similarity_score": similarity_score,
            }), 400

    except Exception as e:
        return jsonify({"status": "FAILURE", "message": str(e)}), 500

    finally:
        # Cleanup temporary files
        if image1_path and os.path.exists(image1_path):
            os.remove(image1_path)
        if image2_path and os.path.exists(image2_path):
            os.remove(image2_path)

# Helper function to convert base64 to a file
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
    



=======
# Register blueprints
app.register_blueprint(passport_ocr_bp, url_prefix='/passport_ocr')
app.register_blueprint(face_match_bp, url_prefix='/passport_ocr')
>>>>>>> 74e8f321266d1a053c57cc02e703cee99a2c5e03

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7000, debug=True)
