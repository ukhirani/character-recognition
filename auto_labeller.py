import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd

# ---- CONFIG ----
IMAGE_PATH = '20250514_111255.jpg'  # The image you provided
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Tesseract configuration
# You might need to install Tesseract and set the path:
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS: brew install tesseract
# For Linux: sudo apt install tesseract-ocr

def detect_grid(image_path):
    """Detect grid in the image and return cell coordinates"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Hough Line Transform to detect grid lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    h_lines = []
    v_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 20:  # Horizontal line
                h_lines.append((min(y1, y2) + abs(y2 - y1) // 2, 0, width))  # y-position, start-x, end-x
            elif abs(x2 - x1) < 20:  # Vertical line
                v_lines.append((min(x1, x2) + abs(x2 - x1) // 2, 0, height))  # x-position, start-y, end-y
    
    # Sort lines by position
    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])
    
    # Filter out duplicate lines (lines that are very close to each other)
    def filter_lines(lines, threshold=20):
        if not lines:
            return []
        filtered = [lines[0]]
        for line in lines[1:]:
            if abs(line[0] - filtered[-1][0]) > threshold:
                filtered.append(line)
        return filtered
    
    h_lines = filter_lines(h_lines)
    v_lines = filter_lines(v_lines)
    
    # If we don't detect enough lines, use fixed grid detection
    if len(h_lines) < 8 or len(v_lines) < 6:  # 7 rows and 5 columns in your grid
        print("Using fixed grid detection")
        rows, cols = 7, 5  # Based on your image
        cell_h = height // rows
        cell_w = width // cols
        
        # Create fixed grid
        h_lines = [(r * cell_h, 0, width) for r in range(rows + 1)]
        v_lines = [(c * cell_w, 0, height) for c in range(cols + 1)]
    
    # Draw lines for visualization
    vis_img = img.copy()
    for y, x1, x2 in h_lines:
        cv2.line(vis_img, (x1, y), (x2, y), (0, 0, 255), 2)
    for x, y1, y2 in v_lines:
        cv2.line(vis_img, (x, y1), (x, y2), (0, 0, 255), 2)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "detected_grid.png"), vis_img)
    
    # Extract cell coordinates from grid intersections
    cells = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1 = h_lines[i][0]
            y2 = h_lines[i + 1][0]
            x1 = v_lines[j][0]
            x2 = v_lines[j + 1][0]
            cells.append((x1, y1, x2 - x1, y2 - y1))
    
    return img, cells

def preprocess_cell(cell_img):
    """Preprocess a cell image for OCR recognition"""
    # Convert to grayscale if not already
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img
    
    # Apply thresholding to get binary image (dark text on white background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours to detect the character
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours found, extract the bounding box
    if contours:
        # Find the largest contour (likely the character)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add some margin
        margin_x = int(0.2 * w)
        margin_y = int(0.2 * h)
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(binary.shape[1] - x, w + 2 * margin_x)
        h = min(binary.shape[0] - y, h + 2 * margin_y)
        
        # Extract the character region
        char_img = binary[y:y+h, x:x+w]
        
        # Resize to a standard size for better OCR
        char_img = cv2.resize(char_img, (100, 100), interpolation=cv2.INTER_AREA)
        
        # Create a white canvas with padding
        result = np.ones((120, 120), dtype=np.uint8) * 255
        result[10:110, 10:110] = char_img
    else:
        # If no contours, just use the binary image
        result = binary
    
    # Convert back to BGR for Tesseract (since we'll display it too)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result_bgr

def recognize_cell(cell_img, cell_index):
    """Recognize character in a cell using OCR"""
    # Preprocess the cell image
    processed = preprocess_cell(cell_img)
    
    # Save processed image for debugging
    proc_path = os.path.join(OUTPUT_DIR, f"processed_{cell_index}.png")
    cv2.imwrite(proc_path, processed)
    
    # Convert to PIL Image for Tesseract
    pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    
    # Configure Tesseract to recognize a single character
    custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    
    try:
        # Recognize the character
        char = pytesseract.image_to_string(pil_img, config=custom_config).strip()
        
        # Get confidence score
        data = pytesseract.image_to_data(pil_img, config=custom_config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        confidence = float(confidences[0])/100 if confidences else 0.0
        
        # If empty or multiple characters, take first or none
        if len(char) > 1:
            char = char[0]
        elif len(char) == 0:
            char = '?'
            confidence = 0.0
            
        return char, confidence
    except Exception as e:
        print(f"Error recognizing cell {cell_index}: {e}")
        return '?', 0.0

def main():
    print(f"Detecting grid in {IMAGE_PATH}...")
    try:
        img, cells = detect_grid(IMAGE_PATH)
    except Exception as e:
        print(f"Error detecting grid: {e}")
        return
    
    # Create a copy for visualization
    output_img = img.copy()
    
    print(f"Found {len(cells)} grid cells")
    results = []
    
    # Process each cell
    for i, (x, y, w, h) in enumerate(cells):
        # Extract cell from image
        cell_img = img[y:y+h, x:x+w]
        
        # Save the cell image for debugging
        cell_path = os.path.join(OUTPUT_DIR, f"cell_{i}.png")
        cv2.imwrite(cell_path, cell_img)
        
        # Recognize character in the cell
        label, confidence = recognize_cell(cell_img, i)
        
        # Store result
        results.append({
            'cell_idx': i,
            'x': x, 'y': y, 'w': w, 'h': h,
            'label': label,
            'confidence': confidence
        })
        
        # Draw rectangle and label on output image
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_img, label, (x+10, y+h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        print(f"Cell {i}: Predicted '{label}' with confidence {confidence:.2f}")
    
    # Save output image
    output_path = os.path.join(OUTPUT_DIR, "labeled_grid.png")
    cv2.imwrite(output_path, output_img)
    
    # Create a grid visualization
    rows, cols = 7, 5  # Based on your image
    grid = [[''] * cols for _ in range(rows)]
    
    # Extract row/column from cell index
    for res in results:
        idx = res['cell_idx']
        row = idx // cols
        col = idx % cols
        if row < rows and col < cols:
            grid[row][col] = res['label']
    
    print("\nDetected Grid:")
    for row in grid:
        print(' '.join(row))
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Labeled image saved as {output_path}")

if __name__ == '__main__':
    main()