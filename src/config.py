MODEL_PATH = "D:\\license_plate_data\\models\\best_license_plate_recognition.pt"

OCR_LANGUAGES = ['en']

CONF_THRESHOLD = 0.30

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

HISTORY_SIZE = 10

import re
PLATE_PATTERN = re.compile(r"^[A-Z0-9]+$")

PHOTO_TYPES = ["jpg", "jpeg", "png", "mp4", "mov", "avi", "jfif"]
