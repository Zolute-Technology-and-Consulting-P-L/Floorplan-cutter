import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io

# Load the PDF file
pdf_path = "/mnt/data/myfile.pdf"
doc = fitz.open(pdf_path)

# Extract the first page as an image
page = doc[0]
pix = page.get_pixmap()
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Convert image to OpenCV format
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Convert to grayscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to enhance edges
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)

# Perform Canny edge detection
edges = cv2.Canny(thresh, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest bounding box containing the floor plan
x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 200 and h > 200:  # Ignore small elements (sidebars, text)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + w), max(y_max, y + h)

# Crop the image to include the main floor plan
cropped_img = img_cv[y_min:y_max, x_min:x_max]

# Ensure we do not crop too much from the right side
height, width, _ = cropped_img.shape
sidebar_width_threshold = width * 0.15  # Dynamically determine sidebar size

# Detect sidebar and remove only if it's beyond a reasonable width
if x_max - x_min > sidebar_width_threshold:
    cropped_img = cropped_img[:, :x_max - int(sidebar_width_threshold)]

# Save the extracted floor plan
output_path = "/mnt/data/myoutpt.png"
cv2.imwrite(output_path, cropped_img)
print(f"Saved extracted floor plan to {output_path}")
