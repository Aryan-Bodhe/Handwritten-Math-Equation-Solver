import cv2
import numpy as np
import os

def connected_component_analysis(image_path, save_dir=os.path.join(os.getcwd(), "cache/final_letters"), output_size=(32, 32)):
    """Applies Connected Component Analysis (CCA) to segment characters in an image from left to right.
       Ensures all output images have the same dimensions. Detects superscripts after merging characters with similar x-coordinates but differing y-coordinates.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Apply binary threshold with inversion
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    os.makedirs(save_dir, exist_ok=True)

    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    sorted_indices = sorted(range(1, num_labels), key=lambda i: stats[i][0])
    
    merged_stats = []
    skip_indices = set()
    
    # Merge characters with nearly the same x-coordinate
    for i in range(len(sorted_indices)):
        if i in skip_indices:
            continue
        
        idx1 = sorted_indices[i]
        x1, y1, w1, h1, area1 = stats[idx1]
        
        x_new, y_new, w_new, h_new = x1, y1, w1, h1
        
        # Check for merging condition
        for j in range(i + 1, len(sorted_indices)):
            if j in skip_indices:
                continue
            
            idx2 = sorted_indices[j]
            x2, y2, w2, h2, area2 = stats[idx2]
            
            if abs(x1 - x2) < 5:
                # Merge bounding boxes
                x_new = min(x1, x2)
                y_new = min(y1, y2)
                w_new = max(x1 + w1, x2 + w2) - x_new
                h_new = max(y1 + h1, y2 + h2) - y_new
                skip_indices.add(j)
            else:
                break
        
        merged_stats.append((x_new, y_new, w_new, h_new))
    
    # Sort merged bounding boxes from left to right
    merged_stats.sort(key=lambda s: s[0])
    
    avg_bottom_y = np.mean([y + h for x, y, w, h in merged_stats])
    avg_height = np.mean([h for x, y, w, h in merged_stats])
    superscript_flags = np.zeros(len(merged_stats), dtype=int)
    
    for i, (x, y, w, h) in enumerate(merged_stats):
        bottom_y = y + h
        is_superscript = (bottom_y < avg_bottom_y - 0.3 * avg_height)
        if is_superscript:
            superscript_flags[i] = 1
        
        # Draw bounding box
        color = (0, 0, 255) if is_superscript else (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Extract and save character
        char_img = binary[y:y + h, x:x + w]
        char_img = resize_with_padding(char_img, output_size)
        char_filename = f"{save_dir}/char_{i+1}.png"
        cv2.imwrite(char_filename, char_img)
    
    cv2.imwrite("./cache/cca_output.png", output)
    print(f"Identified and processed {len(merged_stats)} characters (sorted left to right).")
    print(f"Final images have been saved to {save_dir}")
    return superscript_flags

def resize_with_padding(image, target_size):
    h, w = image.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 0
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded
