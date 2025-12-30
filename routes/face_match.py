import base64
import datetime
from io import BytesIO
import os
from flask import Blueprint, request, jsonify
import tempfile
import numpy as np
import face_recognition
from utils.file_utils import base64_to_file
from utils.image_utils import correct_image_orientation, resize_image

face_match_bp = Blueprint("face_match", __name__)

@face_match_bp.route("/face_match/", methods=["POST"])
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
                # "message": "No face detected in one or both images.",
                "message": "No face detected in image.",
                # "datetime": datetime.now().isoformat()
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
        threshold = 0.5  # Default threshold
        # if similarity_score >= threshold:
        #     return jsonify({
        #         "status": "SUCCESS",
        #         "message": "Faces matched successfully.",
        #         "similarity_score": similarity_score,
        #     }), 200
        # else:
        #     return jsonify({
        #         "status": "FAILURE",
        #         "message": "Faces do not match. Please try again.",
        #         "similarity_score": similarity_score,
        #     }), 400
        
        return jsonify({
            "status": similarity_score >= threshold and "SUCCESS" or "FAILURE",
            "message": "Faces matched successfully.",
            "similarity_score": similarity_score,
        }), 200
          


    except Exception as e:
        return jsonify({"status": "FAILURE", "message": str(e)}), 500

    finally:
        # Cleanup temporary files
        if image1_path and os.path.exists(image1_path):
            os.remove(image1_path)
        if image2_path and os.path.exists(image2_path):
            os.remove(image2_path)
