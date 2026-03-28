import cv2
import numpy as np

def clean_prescription(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # remove paper shadows
    bg = cv2.medianBlur(gray, 21)
    diff = 255 - cv2.absdiff(gray, bg)

    # normalize ink contrast
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # adaptive threshold for handwriting
    thresh = cv2.adaptiveThreshold(
        norm,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,31,7
    )

    kernel = np.ones((2,2),np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return clean