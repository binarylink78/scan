import base64
from datetime import datetime
from io import BytesIO
import csv
import os

CSV_FILE = "mrz_data.csv"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

def add_mrz_to_csv(mrz_data):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            mrz_data['document_number'],
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


def base64_to_file(base64_data):
    return BytesIO(base64.b64decode(base64_data.split(",")[1]))
