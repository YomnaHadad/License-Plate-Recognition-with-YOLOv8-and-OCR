import easyocr, cv2
from config import ALLOWLIST, OCR_LANGUAGES, PLATE_PATTERN

def correct_plate(ocr_text):
  ocr_text = ocr_text.upper().replace(" ", "")
  return ocr_text


def easyocr_reader():
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return reader

def recognize_plate(plate_crop):
    '''1. Preprocessing the license plate region before OCR
       2. ocr reading the detected plate'''
    if plate_crop.size == 0:
        return ""
    
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    plate_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    reader = easyocr_reader()
    ocr_result = reader.readtext(
    plate_resized,
    detail=0,
    allowlist=ALLOWLIST
    )
    if len(ocr_result) > 0:
        candidate = correct_plate(ocr_result[0])
        if candidate and PLATE_PATTERN.match(candidate):
          return candidate
    return ""