from typing import OrderedDict
from venv import logger
from flask import Blueprint, app, request, jsonify
import os
import tempfile
import base64

import pytesseract
from utils.file_utils import allowed_file, add_mrz_to_csv
from utils.image_utils import classify_image, crop_face_with_color_analysis
from fastmrz import FastMRZ

passport_ocr_bp = Blueprint("passport_ocr", __name__)

os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX', r'C:/Program Files/Tesseract-OCR/tessdata')
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'C:/Program Files/Tesseract-OCR/tesseract.exe')


# for linux
# os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX', r'=/usr/share/tesseract-ocr/5/tessdata')
# pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'/usr/bin/tesseract')

fast_mrz = FastMRZ()


@passport_ocr_bp.route("/extract-mrz/", methods=["POST"])
def extract_mrz():
    temp_file_path = None
    is_screen_image = None  
    is_paper_image = None
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

        passport_mrz = fast_mrz.get_mrz(temp_file_path)
        if not passport_mrz:
            return jsonify({"detail": "Failed to extract MRZ from the uploaded file."}), 422

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
            # Use new face cropping method with crop_face_with_color_analysis, detect_moire, detect_screen_reflection, detect_blur, detect_pixelation
            cropped_image_path = os.path.join(save_folder, "cropped_face.jpg")
            cropped_image_path, color_analysis = crop_face_with_color_analysis(uploaded_image_path, save_folder)

            is_screen, is_paper = classify_image(cropped_image_path)

            logger.info(f"Cropped face image saved at {cropped_image_path}")

            with open(cropped_image_path, "rb") as cropped_file:
                cropped_image_base64 = base64.b64encode(cropped_file.read()).decode('utf-8')

        except Exception as face_crop_error:
            logger.error(f"Failed to crop face: {str(face_crop_error)}", exc_info=True)
            cropped_image_base64 = None

        response_data = {
            "status": "SUCCESS",
            "result": {
                "mrz": passport_mrz,
                "passport_photo": cropped_image_base64,
                "is_screen": is_screen,
                "is_paper": is_paper,
                "tampered": int(color_analysis['is_tampered'])
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        # return jsonify({"status": "FAILURE", "message": "Error processing file. Please try again."}), 500
        return jsonify({"status": "FAILURE", "message": str(e)}), 500

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

