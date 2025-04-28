# %% Imports
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import warnings
import os
import math
from skimage import measure # Needed for object detection helper

# %% Constants and Configuration
MODEL_NAME = "Intel/dpt-hybrid-midas"  # Or "Intel/dpt-large"
# --- IMPORTANT: SET THESE ---
IMAGE_PATHS = [r"C:\Users\harsh\Downloads\WhatsApp Image 2025-04-28 at 09.59.20_04fe27d4.jpg", r"C:\Users\harsh\Downloads\WhatsApp Image 2025-04-28 at 09.52.40_e8159334.jpg"] # <-- CHANGE THIS - Needs an image from Pixel 6a with the reference object
# <-- CHANGE THIS - Needs an image from Pixel 6a with the reference object
KNOWN_OBJECT_WIDTH_METERS = 0.0856  # Standard ID-1 card width (e.g., credit card)
# --- Camera Parameters (Approximate for Pixel 6a - VERIFY FOR YOUR DEVICE/SETTINGS) ---
# Option 1: Focal Length (Requires knowing the focal length for the *specific resolution* used)
KNOWN_FOCAL_LENGTH_PIXELS = 3031 # Estimated for 4032x3024 resolution
# Option 2: Field of View (Generally more robust if focal length per resolution is unknown)
# PIXEL_6A_HFOV_DEGREES = 78.0 # Approximate Horizontal FoV
# PIXEL_6A_VFOV_DEGREES = 65.0 # Approximate Vertical FoV
USE_FOCAL_LENGTH_METHOD = True # Set to True to use focal length, False to use FoV for dimension calculation

# --- Visualization ---
DISPLAY_SCALE_FACTOR = 0.5 # Scale factor for the final output window (e.g., 0.5 for half size)

# %% Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Helper Functions

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    if area1 <= 0: return 0.0

    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    if area2 <= 0: return 0.0

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = area1 + area2 - intersection_area
    if union_area <= 0: return 0.0

    iou = intersection_area / union_area
    return iou

# --- Object Detection Functions (kept as dependencies for multistage_object_detection) ---
def detect_objects_from_depth_gradients_improved(depth_map, gradient_threshold=0.05,
                                               blur_kernel_size=5, edge_closing_size=2,
                                               post_dilation_size=3, min_object_size=500,
                                               padding_ratio=0.05, max_objects=5):
    """Detects objects from depth gradients (internal helper)."""
    def preprocess_depth_map(depth_map):
        valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
        if not np.any(valid_mask): return None, valid_mask
        depth_norm = depth_map.copy()
        min_depth, max_depth = np.min(depth_norm[valid_mask]), np.max(depth_norm[valid_mask])
        if max_depth > min_depth: depth_norm = (depth_norm - min_depth) / (max_depth - min_depth)
        else: depth_norm = np.zeros_like(depth_norm)
        depth_norm[~valid_mask] = 0
        return depth_norm, valid_mask

    depth_norm, valid_mask = preprocess_depth_map(depth_map)
    if depth_norm is None: return []
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_blur = cv2.GaussianBlur(depth_uint8, (blur_kernel_size, blur_kernel_size), 0)
    grad_x = cv2.Sobel(depth_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if np.max(gradient_magnitude) > 0: gradient_magnitude /= np.max(gradient_magnitude)
    edges = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255
    if edge_closing_size > 0:
        kernel_close = np.ones((edge_closing_size, edge_closing_size), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    else: edges_closed = edges
    kernel_small = np.ones((3, 3), np.uint8)
    edges_processed = cv2.dilate(edges_closed, kernel_small, iterations=1)
    edges_canny = cv2.Canny(depth_blur, 30, 90)
    combined_edges = cv2.bitwise_or(edges_processed, edges_canny)
    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_drawing = np.zeros_like(combined_edges)
    cv2.drawContours(edge_drawing, contours, -1, 255, 1)
    kernel_dilate = np.ones((3, 3), np.uint8)
    edge_dilated = cv2.dilate(edge_drawing, kernel_dilate, iterations=2)
    object_regions = 255 - edge_dilated
    labeled_regions = measure.label(object_regions, connectivity=2)
    region_properties = measure.regionprops(labeled_regions)
    valid_regions = []
    for region in region_properties:
        if region.area > min_object_size:
            region_mask = (labeled_regions == region.label)
            if post_dilation_size > 0:
                dilate_kernel = np.ones((post_dilation_size, post_dilation_size), np.uint8)
                mask_dilated = cv2.dilate(region_mask.astype(np.uint8) * 255, dilate_kernel)
                region_mask = mask_dilated > 0
            depth_pixels = depth_map[region_mask & valid_mask]
            if len(depth_pixels) > 0: mean_depth = np.mean(depth_pixels)
            else: continue
            min_row, min_col, max_row, max_col = region.bbox
            valid_regions.append({'bbox': (min_row, min_col, max_row, max_col), 'mean_depth': mean_depth})
    if not valid_regions: # Fallback to simple thresholding if gradient fails
        if np.any(valid_mask):
            min_depth = np.min(depth_map[valid_mask])
            foreground_mask = (depth_map <= min_depth * 1.05) & valid_mask
            labeled_foreground = measure.label(foreground_mask, connectivity=2)
            foreground_properties = measure.regionprops(labeled_foreground)
            for region in foreground_properties:
                if region.area > min_object_size:
                    min_row, min_col, max_row, max_col = region.bbox
                    mean_depth = np.mean(depth_map[labeled_foreground == region.label])
                    valid_regions.append({'bbox': (min_row, min_col, max_row, max_col), 'mean_depth': mean_depth})

    valid_regions.sort(key=lambda x: x['mean_depth'])
    valid_regions = valid_regions[:max_objects]
    height, width = depth_map.shape
    padded_boxes = []
    for region in valid_regions:
        min_row, min_col, max_row, max_col = region['bbox']
        h_pad, w_pad = int((max_row - min_row) * padding_ratio), int((max_col - min_col) * padding_ratio)
        min_row, min_col = max(0, min_row - h_pad), max(0, min_col - w_pad)
        max_row, max_col = min(height, max_row + h_pad), min(width, max_col + w_pad)
        padded_boxes.append(((min_col, min_row), (max_col, max_row))) # (x1,y1),(x2,y2)
    return padded_boxes

def detect_objects_by_depth_thresholding(depth_map, threshold_factor=1.05, min_object_size=500,
                                       padding_ratio=0.05, max_objects=5):
    """Detects objects by depth thresholding (internal helper)."""
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if not np.any(valid_mask): return []
    min_depth = np.min(depth_map[valid_mask])
    depth_threshold = min_depth * threshold_factor
    foreground_mask = (depth_map <= depth_threshold) & valid_mask
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask_processed = cv2.morphologyEx(foreground_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    labeled_mask = measure.label(foreground_mask_processed, connectivity=2)
    region_properties = measure.regionprops(labeled_mask)
    height, width = depth_map.shape
    bounding_boxes = []
    for region in region_properties:
        if region.area >= min_object_size:
            min_row, min_col, max_row, max_col = region.bbox
            h_pad, w_pad = int((max_row - min_row) * padding_ratio), int((max_col - min_col) * padding_ratio)
            min_row, min_col = max(0, min_row - h_pad), max(0, min_col - w_pad)
            max_row, max_col = min(height, max_row + h_pad), min(width, max_col + w_pad)
            region_mask = labeled_mask == region.label
            mean_depth = np.mean(depth_map[region_mask & valid_mask])
            bounding_boxes.append({'bbox': ((min_col, min_row), (max_col, max_row)), 'mean_depth': mean_depth}) # (x1,y1),(x2,y2)
    bounding_boxes.sort(key=lambda x: x['mean_depth'])
    return [box['bbox'] for box in bounding_boxes[:max_objects]]

def multistage_object_detection(depth_map, max_objects=5):
    """Combines gradient and threshold detection for robustness."""
    gradient_boxes = detect_objects_from_depth_gradients_improved(
        depth_map, gradient_threshold=0.05, blur_kernel_size=5, edge_closing_size=2,
        post_dilation_size=5, min_object_size=300, padding_ratio=0.1, max_objects=max_objects
    )
    if gradient_boxes:
        return gradient_boxes

    threshold_boxes = detect_objects_by_depth_thresholding(
        depth_map, threshold_factor=1.1, min_object_size=300, padding_ratio=0.1, max_objects=max_objects
    )
    if threshold_boxes:
        return threshold_boxes

    # Fallback: box around the closest point if other methods fail
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if np.any(valid_mask):
        min_depth = np.min(depth_map[valid_mask])
        y_coords, x_coords = np.where((depth_map == min_depth) & valid_mask)
        if len(y_coords) > 0 and len(x_coords) > 0:
            y, x = y_coords[0], x_coords[0]
            height, width = depth_map.shape
            box_size = min(width, height) // 4
            x1, y1 = max(0, x - box_size // 2), max(0, y - box_size // 2)
            x2, y2 = min(width, x + box_size // 2), min(height, y + box_size // 2)
            if x2 > x1 and y2 > y1:
                 return [((x1, y1), (x2, y2))]

    return [] # Return empty list if no objects found

# --- Dimension Calculation ---
def calculate_real_world_dimensions(
    depth_meters, bbox, image_width, image_height,
    focal_length_pixels=None, hfov_degrees=None, vfov_degrees=None
):
    """Calculates real-world dimensions from depth, bbox, and camera parameters."""
    (x1, y1), (x2, y2) = bbox
    width_pixels = x2 - x1
    height_pixels = y2 - y1
    if width_pixels <= 0 or height_pixels <= 0 or depth_meters <= 0:
        return 0.0, 0.0

    width_meters, height_meters = 0.0, 0.0
    if focal_length_pixels is not None and focal_length_pixels > 0:
        width_meters = (width_pixels * depth_meters) / focal_length_pixels
        height_meters = (height_pixels * depth_meters) / focal_length_pixels
    elif hfov_degrees is not None and vfov_degrees is not None:
        hfov_radians = math.radians(hfov_degrees)
        vfov_radians = math.radians(vfov_degrees)
        if hfov_radians > 0 and vfov_radians > 0:
            total_width_at_depth = 2 * depth_meters * math.tan(hfov_radians / 2)
            total_height_at_depth = 2 * depth_meters * math.tan(vfov_radians / 2)
            width_meters = (width_pixels / image_width) * total_width_at_depth
            height_meters = (height_pixels / image_height) * total_height_at_depth
        else:
             warnings.warn("FoV values must be positive.", UserWarning)
    else:
        raise ValueError("Either focal_length_pixels or both hfov_degrees and vfov_degrees must be provided")

    return width_meters, height_meters

# --- Visualization ---
def visualize_objects_with_dimensions(image, objects_analysis, roi_box=None, scale_factor=1.0):
    """Visualizes filtered objects with their dimensions and depths on the image."""
    if image is None:
        print("Error: Cannot visualize on a null image.")
        return

    vis_img = image.copy()
    orig_h, orig_w = image.shape[:2]

    # Resize image if needed for display
    if scale_factor != 1.0:
        new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
        vis_img = cv2.resize(vis_img, (new_w, new_h))
    else:
        new_w, new_h = orig_w, orig_h

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)] # Green, Blue, Red, ...

    # Draw filtered object boxes
    for i, obj in enumerate(objects_analysis):
        bbox = obj['bbox']
        # Scale box coordinates
        x1 = int(bbox[0][0] * (new_w / orig_w))
        y1 = int(bbox[0][1] * (new_h / orig_h))
        x2 = int(bbox[1][0] * (new_w / orig_w))
        y2 = int(bbox[1][1] * (new_h / orig_h))

        color = colors[i % len(colors)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # Prepare text
        dims_text = f"{obj['width_meters']*100:.1f}x{obj['height_meters']*100:.1f}cm"
        depth_text = f"D: {obj['depth_meters']:.2f}m"
        label = f"Obj {i+1}"

        # Put text above the box
        cv2.putText(vis_img, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(vis_img, dims_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(vis_img, depth_text, (x1, y1 - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # Draw ROI box (optional, in a distinct color like Yellow)
    if roi_box:
        rx1 = int(roi_box[0][0] * (new_w / orig_w))
        ry1 = int(roi_box[0][1] * (new_h / orig_h))
        rx2 = int(roi_box[1][0] * (new_w / orig_w))
        ry2 = int(roi_box[1][1] * (new_h / orig_h))
        roi_color = (0, 255, 255) # Yellow
        cv2.rectangle(vis_img, (rx1, ry1), (rx2, ry2), roi_color, 2)
        cv2.putText(vis_img, "Ref", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

    # Display image
    cv2.imshow("Filtered Objects with Dimensions", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save image
    save_path = "objects_with_dimensions_filtered.jpg"
    cv2.imwrite(save_path, vis_img)
    print(f"Final visualization saved to {save_path}")


    # --- 1. Load Model ---
try:
    processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
    model = DPTForDepthEstimation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model/processor: {e}")
    exit()
for IMAGE_PATH in IMAGE_PATHS:
    # --- 2. Load Image ---
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        exit()
    try:
        image_pil = Image.open(IMAGE_PATH).convert("RGB")
        original_width, original_height = image_pil.size
        image_display_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR) # For OpenCV display/ROI
        print(f"Loaded image: {IMAGE_PATH} ({original_width}x{original_height})")

        # Resolution warning for focal length
        if USE_FOCAL_LENGTH_METHOD and (original_width != 4032 or original_height != 3024):
                warnings.warn(
                f"WARNING: Image resolution is {original_width}x{original_height}, "
                f"but focal length {KNOWN_FOCAL_LENGTH_PIXELS}px was potentially estimated for 4032x3024. "
                f"Calculated dimensions might be inaccurate. Consider using FoV method or adjusting focal length.", UserWarning
            )

    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # --- 3. Predict Depth ---
    try:
        inputs = processor(images=image_pil, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            outputs = model(pixel_values)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original image size
        prediction_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        relative_depth_map = prediction_resized.squeeze().cpu().numpy()
        print("Depth map predicted.")
    except Exception as e:
        print(f"Error during depth prediction: {e}")
        exit()

    # --- 4. Select Reference Object (ROI) ---
    cv2.namedWindow("Select Reference Object (e.g., ID card)", cv2.WINDOW_NORMAL)
    preview_width = min(1280, original_width)
    preview_height = int(preview_width * (original_height / original_width))
    cv2.resizeWindow("Select Reference Object (e.g., ID card)", preview_width, preview_height)
    roi = cv2.selectROI("Select Reference Object (e.g., ID card)", image_display_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Reference Object (e.g., ID card)")

    if roi == (0, 0, 0, 0) or roi[2] == 0 or roi[3] == 0:
        print("ROI selection cancelled or invalid. Exiting.")
        exit()
    x, y, w, h = roi
    roi_box_coords = ((x, y), (x + w, y + h)) # Save for filtering/visualization
    print(f"Reference ROI selected: x={x}, y={y}, width={w}, height={h}")

    # --- 5. Calculate Scale Factor ---
    object_width_pixels_ref = w
    roi_depth_values = relative_depth_map[y:y+h, x:x+w]

    if object_width_pixels_ref <= 0 or roi_depth_values.size == 0:
        print("Error: Invalid reference object dimensions or depth data in ROI.")
        exit()

    d_relative_ref = np.median(roi_depth_values)
    if d_relative_ref <= 1e-6:
        print("Error: Relative depth of the reference object is too small or zero.")
        exit()

    # This scale factor relates relative depth to absolute depth
    # AbsoluteDepth = ScaleFactor / RelativeDepth
    # Derived from: AbsoluteDepth = (KnownWidth * FocalLength) / PixelWidth
    #             ScaleFactor / RelativeDepth = (KnownWidth * FocalLength) / PixelWidth
    #             ScaleFactor = (KnownWidth * FocalLength * RelativeDepth) / PixelWidth
    scale_factor = (KNOWN_OBJECT_WIDTH_METERS * KNOWN_FOCAL_LENGTH_PIXELS * d_relative_ref) / object_width_pixels_ref
    print(f"Calculated Scale Factor: {scale_factor:.4f}")

    # Optional: Estimate distance to reference object
    estimated_distance_ref = scale_factor / d_relative_ref
    print(f"Estimated distance to reference object: {estimated_distance_ref:.2f} meters")


    # --- 6. Detect Other Objects ---
    print("Detecting other objects based on depth map...")
    detected_boxes = multistage_object_detection(relative_depth_map, max_objects=5) # Find up to 5 objects
    print(f"Detected {len(detected_boxes)} potential object boxes.")

    # --- 7. Filter Detected Objects ---
    filtered_boxes = []
    iou_scores = []
    if detected_boxes:
        for i, det_box in enumerate(detected_boxes):
            iou = calculate_iou(det_box, roi_box_coords)
            iou_scores.append({'iou': iou, 'box': det_box, 'index': i})

        # Sort by IoU, highest first
        iou_scores.sort(key=lambda item: item['iou'], reverse=True)

        # Identify the box most likely to be the reference object (highest IoU)
        ref_object_index = iou_scores[0]['index'] if iou_scores and iou_scores[0]['iou'] > 0.1 else -1 # Threshold IoU

        # Keep boxes that are *not* the reference object and *not* the full image frame
        for score_info in iou_scores:
            if score_info['index'] != ref_object_index:
                    # Check if the box covers nearly the entire image (potential artifact)
                    b_x1, b_y1 = score_info['box'][0]
                    b_x2, b_y2 = score_info['box'][1]
                    box_area = (b_x2 - b_x1) * (b_y2 - b_y1)
                    image_area = original_width * original_height
                    # Filter out boxes covering >95% of the image or tiny boxes
                    if box_area / image_area < 0.95 and box_area > 100:
                        filtered_boxes.append(score_info['box'])

    print(f"Filtered down to {len(filtered_boxes)} objects (removed reference object overlap and potential artifacts).")

    # --- 8. Calculate Depth & Dimensions for Filtered Objects ---
    object_analysis_results = []
    if filtered_boxes:
        print("\n--- Analyzing Filtered Objects ---")
        for i, bbox in enumerate(filtered_boxes):
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]

            # Ensure coordinates are within bounds
            y1, y2 = max(0, y1), min(original_height, y2)
            x1, x2 = max(0, x1), min(original_width, x2)

            if y1 >= y2 or x1 >= x2: continue # Skip invalid boxes

            # Calculate absolute depth using the scale factor
            detected_obj_depth_values = relative_depth_map[y1:y2, x1:x2]
            if detected_obj_depth_values.size == 0: continue

            d_relative_detected = np.median(detected_obj_depth_values)
            if d_relative_detected <= 1e-6: continue

            absolute_depth_meters = scale_factor / d_relative_detected

            # Calculate real-world dimensions
            try:
                width_m, height_m = calculate_real_world_dimensions(
                    absolute_depth_meters,
                    bbox,
                    original_width,
                    original_height,
                    focal_length_pixels=KNOWN_FOCAL_LENGTH_PIXELS if USE_FOCAL_LENGTH_METHOD else None,
                    # Uncomment and provide if using FoV method
                    # hfov_degrees=PIXEL_6A_HFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None,
                    # vfov_degrees=PIXEL_6A_VFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None
                )
            except ValueError as e:
                    print(f"Dimension calculation error: {e}")
                    width_m, height_m = 0.0, 0.0 # Assign default values on error


            if width_m > 0 and height_m > 0:
                object_info = {
                    'index': i,
                    'bbox': bbox,
                    'depth_meters': absolute_depth_meters,
                    'width_meters': width_m,
                    'height_meters': height_m,
                }
                object_analysis_results.append(object_info)
                print(f"  Object {i+1}: Depth={absolute_depth_meters:.2f}m, Dimensions={width_m*100:.1f}cm x {height_m*100:.1f}cm")

    else:
        print("No objects remained after filtering.")

    # --- 9. Visualize Final Results ---
    if object_analysis_results:
            print("\nDisplaying final visualization...")
            visualize_objects_with_dimensions(
                image_display_bgr, # Use the OpenCV BGR image
                object_analysis_results,
                roi_box=roi_box_coords, # Show the reference ROI as well
                scale_factor=DISPLAY_SCALE_FACTOR
            )
    else:
            print("Nothing to visualize.")

    print("\nProcessing complete.")