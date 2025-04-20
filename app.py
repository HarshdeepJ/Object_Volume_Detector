import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image # Ensure Image is imported
from transformers import DPTImageProcessor, DPTForDepthEstimation
import warnings
import math
from skimage import measure
import io
import os
import tempfile
from streamlit_drawable_canvas import st_canvas


torch.classes.__path__ = []
# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.functional')

st.set_page_config(
    page_title="3D Object Measurement Tool",
    page_icon="📏",
    layout="wide"
)

# Title and description
st.title("📏 3D Object Measurement Tool")
st.markdown("""
This application estimates the real-world dimensions of objects from a single image.
1.  **Upload an image** containing the objects you want to measure.
2.  **Estimate Depth:** Click the button to generate a depth map.
3.  **Select Reference:** **Draw a rectangle** around the reference object (known size) on the image below. The image may be scaled down for easier drawing if it's large.
4.  **Measure:** Click the button to get the estimated dimensions and distances of other objects.
""") # Updated instruction

# --- Constants and Configuration ---
MODEL_NAME = "Intel/dpt-hybrid-midas"  # Or "Intel/dpt-large"
MAX_CANVAS_DISPLAY_WIDTH = 800 # <--- NEW: Max width for the drawing canvas

st.sidebar.subheader("Configuration")
KNOWN_OBJECT_WIDTH_METERS = st.sidebar.number_input(
    "Reference Object Width (meters)",
    min_value=0.01,
    max_value=1.0,
    value=0.0856, # Standard ID-1 card width
    step=0.001,
    format="%.4f",
    help="Enter the actual width of the reference object you will select in the image. Default is standard credit card width (85.6mm)."
)

# --- Camera Parameters ---
st.sidebar.subheader("Camera Parameters")
st.sidebar.info("Provide either focal length OR field of view. Accuracy depends heavily on these values.")
camera_method = st.sidebar.radio(
    "Calculation Method",
    ["Focal Length", "Field of View (FoV)"],
    key="camera_method_radio",
    help="Choose how to provide camera intrinsics."
)

USE_FOCAL_LENGTH_METHOD = (camera_method == "Focal Length")
if USE_FOCAL_LENGTH_METHOD:
    KNOWN_FOCAL_LENGTH_PIXELS = st.sidebar.number_input(
        "Camera Focal Length (pixels)",
        min_value=100,
        max_value=10000,
        value=1500, # A reasonable default, adjust based on your camera/image
        step=10,
        help="Focal length in pixels. This depends on the camera sensor and image resolution. Search online for your phone model + 'focal length pixels' or use EXIF data."
    )
    HFOV_DEGREES = None
    VFOV_DEGREES = None
else:
    KNOWN_FOCAL_LENGTH_PIXELS = None
    HFOV_DEGREES = st.sidebar.number_input(
        "Horizontal Field of View (degrees)",
        min_value=20.0,
        max_value=150.0,
        value=78.0, # Example FoV
        step=0.5,
        format="%.1f",
        help="Horizontal angle the camera can see. Check your camera/phone specifications."
    )
    VFOV_DEGREES = st.sidebar.number_input(
        "Vertical Field of View (degrees)",
        min_value=20.0,
        max_value=150.0,
        value=65.0, # Example FoV
        step=0.5,
        format="%.1f",
        help="Vertical angle the camera can see. Check your camera/phone specifications."
    )

# --- Advanced settings ---
with st.sidebar.expander("Advanced Settings"):
    gradient_threshold = st.slider("Depth Gradient Threshold", 0.01, 0.20, 0.05, 0.01, help="Sensitivity for detecting edges in the depth map. Lower values find more edges.")
    min_object_size = st.slider("Min Object Area (pixels)", 100, 5000, 500, 50, help="Minimum number of pixels for an area to be considered an object.")
    max_objects = st.slider("Max Objects to Detect", 1, 20, 5, 1, help="Maximum number of objects to find (excluding the reference).")
    object_padding_ratio = st.slider("Object BBox Padding", 0.0, 0.2, 0.05, 0.01, help="Expand detected object bounding boxes slightly.")

# --- Helper Functions ---
@st.cache_resource # Cache the model loading
def load_model():
    """Load the depth estimation model."""
    st.info("Loading depth estimation model... This might take a minute on first run.")
    try:
        processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
        model = DPTForDepthEstimation.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        st.success("Model loaded successfully.")
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure you have an internet connection and the necessary libraries installed.")
        return None, None, None

def predict_depth(image_pil, processor, model, device):
    """Predict depth map from a PIL image."""
    if processor is None or model is None or device is None:
        st.error("Model not loaded. Cannot predict depth.")
        return None
    try:
        with torch.no_grad():
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate prediction to original image size
        prediction_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], # PIL size is (width, height), interpolate needs (height, width)
            mode="bilinear",
            align_corners=False,
        )
        relative_depth_map = prediction_resized.squeeze().cpu().numpy()
        return relative_depth_map
    except Exception as e:
        st.error(f"Error during depth prediction: {e}")
        return None

def normalize_depth_for_display(depth_map):
    """Normalize depth map for visualization."""
    depth_vis = depth_map.copy()
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if np.any(valid_mask):
        min_val = np.min(depth_vis[valid_mask])
        max_val = np.max(depth_vis[valid_mask])
        if max_val > min_val:
            depth_vis[valid_mask] = (depth_vis[valid_mask] - min_val) / (max_val - min_val)
        else: # Handle case where all valid depths are the same
            depth_vis[valid_mask] = 0.5 # Or 0 or 1
    depth_vis[~valid_mask] = 0 # Set invalid areas to 0
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO) # Or COLORMAP_MAGMA, INFERNO
    return depth_colored

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are format: ((x1, y1), (x2, y2))"""
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    if area1 <= 0: return 0.0

    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    if area2 <= 0: return 0.0

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = area1 + area2 - intersection_area
    if union_area <= 0: return 0.0

    iou = intersection_area / union_area
    return iou

def detect_objects_from_depth_gradients_improved(depth_map, gradient_threshold=0.05,
                                               blur_kernel_size=5, edge_closing_size=7,
                                               post_dilation_size=5, min_object_size=500,
                                               padding_ratio=0.05, max_objects=5):
    """Detects objects from depth gradients (more robust version)."""
    height, width = depth_map.shape

    # 1. Preprocessing: Normalize and handle NaNs/zeros
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if not np.any(valid_mask): return []

    depth_norm = depth_map.copy()
    min_depth = np.min(depth_norm[valid_mask])
    max_depth = np.max(depth_norm[valid_mask])

    if max_depth > min_depth:
        depth_norm[valid_mask] = (depth_norm[valid_mask] - min_depth) / (max_depth - min_depth)
    else: # Handle flat depth map
        depth_norm[valid_mask] = 0.5
    depth_norm[~valid_mask] = 0 # Fill invalid areas

    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # 2. Blur to reduce noise
    depth_blur = cv2.GaussianBlur(depth_uint8, (blur_kernel_size, blur_kernel_size), 0)

    # 3. Calculate Gradient Magnitude
    grad_x = cv2.Sobel(depth_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_grad = np.max(gradient_magnitude)
    if max_grad > 0:
        gradient_magnitude /= max_grad # Normalize gradient

    # 4. Threshold Gradient Edges
    edges = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255

    # 5. Morphological Operations to close gaps
    if edge_closing_size > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_closing_size, edge_closing_size))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    else:
        edges_closed = edges

    # Optional: Combine with Canny edges for potentially sharper boundaries
    # edges_canny = cv2.Canny(depth_blur, 30, 90)
    # combined_edges = cv2.bitwise_or(edges_closed, edges_canny)

    # 6. Find Contours on the closed edges
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_regions = []
    if contours:
        # Create a mask from contours to find connected components inside
        contour_mask = np.zeros_like(edges_closed)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Label connected components *within* the contours (potential objects)
        labeled_regions, num_labels = measure.label(contour_mask, connectivity=2, return_num=True)
        region_properties = measure.regionprops(labeled_regions, intensity_image=depth_map) # Use original depth for stats

        for region in region_properties:
            # Filter by size and ensure it used valid depth pixels for mean calculation
            valid_region_mask = (labeled_regions == region.label) & valid_mask
            if region.area > min_object_size and np.any(valid_region_mask):
                # Calculate mean depth only from valid pixels within the region
                mean_depth = np.mean(depth_map[valid_region_mask])
                min_row, min_col, max_row, max_col = region.bbox
                valid_regions.append({'bbox': (min_row, min_col, max_row, max_col), 'mean_depth': mean_depth, 'area': region.area})

    # Sort by depth (closer objects first) or area (larger objects first)
    valid_regions.sort(key=lambda x: x['mean_depth']) # Closer first
    # valid_regions.sort(key=lambda x: x['area'], reverse=True) # Larger first

    # Apply padding and limit number of objects
    padded_boxes = []
    for region in valid_regions[:max_objects]:
        min_row, min_col, max_row, max_col = region['bbox']
        h_pad = int((max_row - min_row) * padding_ratio)
        w_pad = int((max_col - min_col) * padding_ratio)
        # Apply padding and clamp to image boundaries
        final_min_row = max(0, min_row - h_pad)
        final_min_col = max(0, min_col - w_pad)
        final_max_row = min(height, max_row + h_pad)
        final_max_col = min(width, max_col + w_pad)

        # Ensure box has valid dimensions after padding/clamping
        if final_max_row > final_min_row and final_max_col > final_min_col:
            padded_boxes.append(((final_min_col, final_min_row), (final_max_col, final_max_row))) # Format: ((x1,y1),(x2,y2))

    return padded_boxes


def detect_objects_by_depth_thresholding(depth_map, threshold_factor=1.1, min_object_size=500,
                                       padding_ratio=0.05, max_objects=5):
    """Detects objects by thresholding based on the minimum depth."""
    height, width = depth_map.shape
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if not np.any(valid_mask): return []

    min_depth = np.min(depth_map[valid_mask])
    depth_threshold = min_depth * threshold_factor

    # Select pixels close to the minimum depth
    foreground_mask = (depth_map <= depth_threshold) & valid_mask

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask_processed = cv2.morphologyEx(foreground_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    foreground_mask_processed = cv2.morphologyEx(foreground_mask_processed, cv2.MORPH_OPEN, kernel) # Remove small noise

    # Label connected components
    labeled_mask, num_labels = measure.label(foreground_mask_processed, connectivity=2, return_num=True)
    region_properties = measure.regionprops(labeled_mask, intensity_image=depth_map) # Use original depth

    bounding_boxes_info = []
    for region in region_properties:
         # Filter by size and ensure it used valid depth pixels for mean calculation
        valid_region_mask = (labeled_mask == region.label) & valid_mask
        if region.area >= min_object_size and np.any(valid_region_mask):
            min_row, min_col, max_row, max_col = region.bbox
            mean_depth = np.mean(depth_map[valid_region_mask])
            bounding_boxes_info.append({'bbox_coords': (min_row, min_col, max_row, max_col), 'mean_depth': mean_depth, 'area': region.area})

    # Sort by depth (closer objects first)
    bounding_boxes_info.sort(key=lambda x: x['mean_depth'])

    # Apply padding and limit number of objects
    padded_boxes = []
    for region in bounding_boxes_info[:max_objects]:
        min_row, min_col, max_row, max_col = region['bbox_coords']
        h_pad = int((max_row - min_row) * padding_ratio)
        w_pad = int((max_col - min_col) * padding_ratio)

        final_min_row = max(0, min_row - h_pad)
        final_min_col = max(0, min_col - w_pad)
        final_max_row = min(height, max_row + h_pad)
        final_max_col = min(width, max_col + w_pad)

        if final_max_row > final_min_row and final_max_col > final_min_col:
             # Format: ((x1,y1),(x2,y2))
            padded_boxes.append(((final_min_col, final_min_row), (final_max_col, final_max_row)))

    return padded_boxes

def multistage_object_detection(depth_map, gradient_thresh, min_size, max_num, padding_ratio):
    """Combines gradient and threshold detection for robustness."""
    # Prioritize gradient detection as it's often better for distinct objects
    gradient_boxes = detect_objects_from_depth_gradients_improved(
        depth_map,
        gradient_threshold=gradient_thresh,
        min_object_size=min_size,
        max_objects=max_num + 5, # Detect slightly more initially
        padding_ratio=padding_ratio
    )
    if len(gradient_boxes) > 0:
        # st.sidebar.write(f"Gradient Detection found {len(gradient_boxes)} candidates.")
        return gradient_boxes

    # Fallback to thresholding if gradient fails
    # st.sidebar.write("Gradient detection failed, falling back to thresholding.")
    threshold_boxes = detect_objects_by_depth_thresholding(
        depth_map,
        threshold_factor=1.15, # Slightly larger factor for thresholding fallback
        min_object_size=min_size,
        max_objects=max_num + 5,
        padding_ratio=padding_ratio
    )
    if len(threshold_boxes) > 0:
         # st.sidebar.write(f"Threshold Detection found {len(threshold_boxes)} candidates.")
        return threshold_boxes

    # Final fallback: Box around the absolute closest point(s)
    st.sidebar.warning("Detection methods found no distinct objects. Falling back to closest point.")
    valid_mask = ~np.isnan(depth_map) & (depth_map > 0)
    if np.any(valid_mask):
        min_depth_val = np.min(depth_map[valid_mask])
        y_coords, x_coords = np.where((depth_map == min_depth_val) & valid_mask)
        if len(y_coords) > 0 and len(x_coords) > 0:
            # Take the center of the closest points if multiple exist
            cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
            height, width = depth_map.shape
            # Define a small box around the closest point
            box_half_width = max(30, min(width, height) // 20)
            box_half_height = max(30, min(width, height) // 20)
            x1 = max(0, cx - box_half_width)
            y1 = max(0, cy - box_half_height)
            x2 = min(width, cx + box_half_width)
            y2 = min(height, cy + box_half_height)
            if x2 > x1 and y2 > y1:
                 return [((x1, y1), (x2, y2))] # Return as list with one box

    st.sidebar.error("No objects could be detected.")
    return [] # Return empty list if nothing is found

def calculate_real_world_dimensions(
    depth_meters, bbox, image_width, image_height,
    focal_length_pixels=None, hfov_degrees=None, vfov_degrees=None
):
    """Calculates real-world dimensions from depth, bbox, and camera parameters."""
    (x1, y1), (x2, y2) = bbox
    # Ensure coordinates are integers for pixel calculations
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    width_pixels = x2 - x1
    height_pixels = y2 - y1

    if width_pixels <= 0 or height_pixels <= 0 or depth_meters <= 0:
        st.warning(f"Invalid input for dimension calculation: Pixels=({width_pixels}x{height_pixels}), Depth={depth_meters:.2f}")
        return 0.0, 0.0

    width_meters, height_meters = 0.0, 0.0

    # Method 1: Using Focal Length
    if focal_length_pixels is not None and focal_length_pixels > 0:
        # Basic pinhole camera model: real_width / depth = pixel_width / focal_length
        width_meters = (width_pixels * depth_meters) / focal_length_pixels
        # Assume same focal length vertically, or calculate if fy is known separately
        height_meters = (height_pixels * depth_meters) / focal_length_pixels
        # print(f"Focal Method: Pixels=({width_pixels}x{height_pixels}), Depth={depth_meters:.3f}, F={focal_length_pixels} -> Real=({width_meters:.3f}x{height_meters:.3f})")


    # Method 2: Using Field of View (FoV)
    elif hfov_degrees is not None and vfov_degrees is not None:
        hfov_radians = math.radians(hfov_degrees)
        vfov_radians = math.radians(vfov_degrees)
        if hfov_radians > 0 and vfov_radians > 0 and image_width > 0 and image_height > 0:
            # Calculate the total real-world width/height visible at the object's depth
            total_width_at_depth = 2 * depth_meters * math.tan(hfov_radians / 2)
            total_height_at_depth = 2 * depth_meters * math.tan(vfov_radians / 2)

            # Calculate the object's real size as a proportion of the total visible size
            width_meters = (width_pixels / image_width) * total_width_at_depth
            height_meters = (height_pixels / image_height) * total_height_at_depth
            # print(f"FoV Method: Pixels=({width_pixels}x{height_pixels}), Depth={depth_meters:.3f}, FoV=({hfov_degrees:.1f}x{vfov_degrees:.1f}), Img=({image_width}x{image_height}) -> Real=({width_meters:.3f}x{height_meters:.3f})")

        else:
            st.warning("FoV values and image dimensions must be positive for FoV calculation.")
            return 0.0, 0.0
    else:
        # This should not happen if UI logic is correct, but good to have a fallback
        raise ValueError("Camera parameters (Focal Length or FoV) are missing or invalid.")

    return width_meters, height_meters

def visualize_objects_with_dimensions(image_bgr, objects_analysis, roi_box=None):
    """Creates visualization of objects with their dimensions on a BGR image."""
    vis_img = image_bgr.copy()
    height, width, _ = vis_img.shape

    # Define distinct colors
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),         # Green, Blue, Red
        (0, 255, 255), (255, 255, 0), (255, 0, 255),   # Yellow, Cyan, Magenta
        (0, 128, 255), (255, 128, 0), (128, 0, 255)    # Orange, Sky Blue, Purple
    ]

    # Draw ROI box if provided
    if roi_box:
        rx1, ry1 = int(round(roi_box[0][0])), int(round(roi_box[0][1]))
        rx2, ry2 = int(round(roi_box[1][0])), int(round(roi_box[1][1]))
        roi_color = (0, 255, 255) # Yellow for ROI
        cv2.rectangle(vis_img, (rx1, ry1), (rx2, ry2), roi_color, 3) # Thicker line for ROI
        cv2.putText(vis_img, "REF", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

    # Draw detected object boxes and labels
    for i, obj in enumerate(objects_analysis):
        bbox = obj['bbox']
        # Ensure coords are integers for drawing
        x1, y1 = int(round(bbox[0][0])), int(round(bbox[0][1]))
        x2, y2 = int(round(bbox[1][0])), int(round(bbox[1][1]))

        color = colors[i % len(colors)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # Prepare text labels
        label = f"Obj {obj['index']}" # Use stored index
        dims_text = f"Size: {obj['width_meters']*100:.1f} x {obj['height_meters']*100:.1f} cm"
        depth_text = f"Dist: {obj['depth_meters']:.2f} m"

        # Calculate text position (above the box)
        text_y = y1 - 10 # Starting position slightly above the box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        dims_size, _ = cv2.getTextSize(dims_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        depth_size, _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Add background rectangle for better readability
        text_bg_y1 = text_y - label_size[1] - dims_size[1] - depth_size[1] - 15 # Adjust based on number of lines
        text_bg_y2 = y1 - 5
        max_text_width = max(label_size[0], dims_size[0], depth_size[0])
        text_bg_x1 = x1
        text_bg_x2 = x1 + max_text_width + 10

        # Ensure background box doesn't go off-screen
        text_bg_y1 = max(0, text_bg_y1)
        text_bg_x2 = min(width, text_bg_x2)

        # Check if box is too close to the top, put text inside instead
        if text_bg_y1 < 10 or text_y < label_size[1] + 5: # Adjusted condition
             # Draw inside the box near the top
             text_y = y1 + label_size[1] + 5
             text_bg_y1 = y1 + 5
             text_bg_y2 = text_y + dims_size[1] + depth_size[1] + 10
             # Adjust background size more carefully
             text_bg_x1 = x1 + 5
             text_bg_x2 = x1 + 5 + max_text_width + 10
             text_bg_x2 = min(x2 - 5, text_bg_x2) # Ensure bg fits horizontally inside
             text_bg_y2 = min(y2 - 5, text_bg_y2) # Ensure bg fits vertically inside

             if text_bg_y2 > text_bg_y1 + 10 and text_bg_x2 > text_bg_x1 + 10: # Only draw background if there's space
                 sub_img = vis_img[text_bg_y1:text_bg_y2, text_bg_x1:text_bg_x2]
                 white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
                 res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
                 vis_img[text_bg_y1:text_bg_y2, text_bg_x1:text_bg_x2] = res

                 cv2.putText(vis_img, label, (text_bg_x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                 cv2.putText(vis_img, dims_text, (text_bg_x1 + 5, text_y + dims_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                 cv2.putText(vis_img, depth_text, (text_bg_x1 + 5, text_y + dims_size[1] + depth_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        else:
             # Draw above the box
             # Adjust background position to ensure it fits
             text_bg_x1 = x1
             text_bg_x2 = x1 + max_text_width + 10
             text_bg_x2 = min(width, text_bg_x2) # Clamp right edge
             text_bg_x1 = max(0, text_bg_x2 - (max_text_width + 10)) # Adjust left edge if needed

             sub_img = vis_img[text_bg_y1:text_bg_y2, text_bg_x1:text_bg_x2]
             white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
             res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0) # Semi-transparent background
             vis_img[text_bg_y1:text_bg_y2, text_bg_x1:text_bg_x2] = res

             cv2.putText(vis_img, label, (text_bg_x1 + 5, text_y - dims_size[1] - depth_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
             cv2.putText(vis_img, dims_text, (text_bg_x1 + 5, text_y - depth_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
             cv2.putText(vis_img, depth_text, (text_bg_x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    return vis_img

# --- Main App Logic ---
def main():
    # --- Initialization and State Management ---
    if 'depth_estimated' not in st.session_state:
        st.session_state.depth_estimated = False
    if 'roi_defined_by_canvas' not in st.session_state:
        st.session_state.roi_defined_by_canvas = False # Use a new flag for canvas ROI
    if 'results_calculated' not in st.session_state:
        st.session_state.results_calculated = False
    if 'uploaded_file_id' not in st.session_state:
        st.session_state.uploaded_file_id = None

    # --- Model Loading ---
    processor, model, device = load_model()
    if not all([processor, model, device]):
        st.stop() # Stop execution if model loading failed

    # --- Image Upload ---
    uploaded_file = st.file_uploader("1. Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        new_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

        if st.session_state.uploaded_file_id != new_file_id:
            st.session_state.depth_estimated = False
            st.session_state.roi_defined_by_canvas = False # Reset new flag
            st.session_state.results_calculated = False
            st.session_state.uploaded_file_id = new_file_id
            # Clear previous results/data if needed
            # Kept roi_box_coords as it's still used internally
            # Added display_scale_factor to clear list
            for key in ['depth_map', 'depth_colored', 'image_bgr', 'image_rgb', 'image_pil', 'original_width', 'original_height', 'object_analysis_results', 'roi_box_coords', 'visualization', 'scale_factor', 'estimated_distance_ref', 'display_scale_factor']:
                if key in st.session_state:
                    del st.session_state[key]
            st.info("New image uploaded. Please proceed with depth estimation.")

        # Load and display the uploaded image
        try:
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.error("Could not decode image. Please upload a valid JPG, JPEG, or PNG file.")
                st.session_state.uploaded_file_id = None
                return
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb) # Keep PIL version for canvas
            original_width, original_height = image_pil.size

            # Store image data in session state only once per file load
            if 'image_bgr' not in st.session_state:
                st.session_state.image_bgr = image_bgr
                st.session_state.image_rgb = image_rgb
                st.session_state.image_pil = image_pil # Store PIL image
                st.session_state.original_width = original_width
                st.session_state.original_height = original_height

            # Show original image first
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.session_state.uploaded_file_id = None
            return

        # --- Step 1: Depth Estimation ---
        st.subheader("2. Estimate Depth Map")
        if 'image_pil' in st.session_state: # Need PIL image for depth prediction
            if st.button("Estimate Depth", key="estimate_depth_btn", disabled=st.session_state.depth_estimated):
                with st.spinner("🧠 Estimating depth map... This may take a moment."):
                    relative_depth_map = predict_depth(st.session_state.image_pil, processor, model, device)

                    if relative_depth_map is not None:
                        st.session_state.depth_map = relative_depth_map
                        st.session_state.depth_colored = normalize_depth_for_display(relative_depth_map)
                        st.session_state.depth_estimated = True
                        # Reset subsequent steps if depth is re-estimated
                        st.session_state.roi_defined_by_canvas = False # Reset new flag
                        st.session_state.results_calculated = False
                        # Clear ROI coords and scale factor too
                        for key in ['object_analysis_results', 'visualization', 'scale_factor', 'estimated_distance_ref', 'roi_box_coords', 'display_scale_factor']:
                            if key in st.session_state: del st.session_state[key]
                        st.experimental_rerun()
                    else:
                        st.error("Depth estimation failed.")

            if st.session_state.depth_estimated and 'depth_colored' in st.session_state:
                st.image(st.session_state.depth_colored, caption="Estimated Depth Map (Relative)", use_column_width=True)
        else:
            st.warning("Waiting for image data...")


        # --- Step 2: Reference Object Selection (with Resizing) ---
        if st.session_state.depth_estimated:
            st.subheader("3. Select Reference Object")
            st.markdown(f"Draw a rectangle on the image below around your reference object (known width: **{KNOWN_OBJECT_WIDTH_METERS * 100:.1f} cm**). The image may be scaled down for easier drawing if it's large.")

            # Ensure PIL image is available for the canvas background
            if 'image_pil' in st.session_state:
                original_pil_image = st.session_state.image_pil
                original_width = st.session_state.original_width
                original_height = st.session_state.original_height

                # --- START: RESIZING LOGIC ---
                # Calculate display dimensions, respecting MAX_CANVAS_DISPLAY_WIDTH
                if original_width > MAX_CANVAS_DISPLAY_WIDTH:
                    display_scale_factor = MAX_CANVAS_DISPLAY_WIDTH / original_width
                    display_width = MAX_CANVAS_DISPLAY_WIDTH
                    display_height = int(original_height * display_scale_factor)
                    # Use LANCZOS or ANTIALIAS for better quality resizing
                    try:
                        # Pillow >= 9.1.0 uses Resampling enum
                        display_image = original_pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                    except AttributeError:
                        # Older Pillow versions use integer constants
                        display_image = original_pil_image.resize((display_width, display_height), Image.LANCZOS) # Use ANTIALIAS as fallback if needed
                    st.session_state.display_scale_factor = display_scale_factor # Store for coordinate conversion
                    # st.info(f"Image resized for drawing canvas to {display_width}x{display_height}px (Scale: {display_scale_factor:.2f})") # Optional info
                else:
                    # No resizing needed if image is already small enough
                    display_image = original_pil_image
                    display_width = original_width
                    display_height = original_height
                    st.session_state.display_scale_factor = 1.0 # Scale factor is 1
                # --- END: RESIZING LOGIC ---

                # Configure canvas properties
                stroke_width = 3
                stroke_color = "#FFFF00"  # Yellow in hex
                bg_image = display_image # Use the (potentially resized) display image
                drawing_mode = "rect"
                canvas_key = "ref_canvas"

                # Create the canvas component
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.1)",  # Transparent orange fill (optional)
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_image=bg_image, # Use display image
                    update_streamlit=True, # Important: Update Streamlit on drawing events
                    height=display_height, # Use display dimensions
                    width=display_width,   # Use display dimensions
                    drawing_mode=drawing_mode,
                    key=canvas_key,
                )

                # Process the drawing result
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    # Use the latest drawn rectangle
                    if objects and objects[-1]["type"] == "rect":
                        rect_data = objects[-1]
                        # These are coordinates on the *display* canvas
                        display_left = rect_data["left"]
                        display_top = rect_data["top"]
                        display_width_drawn = rect_data["width"]
                        display_height_drawn = rect_data["height"]

                        # Basic validation on drawn dimensions
                        if display_width_drawn > 5 and display_height_drawn > 5: # Require a minimum size on canvas
                             # --- START: SCALE COORDINATES BACK ---
                             # Retrieve the scale factor used for display
                            scale_factor = st.session_state.get('display_scale_factor', 1.0) # Default to 1 if not found

                            # Apply inverse scaling to get coordinates relative to the *original* image
                            original_x1 = display_left / scale_factor
                            original_y1 = display_top / scale_factor
                            original_x2 = (display_left + display_width_drawn) / scale_factor
                            original_y2 = (display_top + display_height_drawn) / scale_factor
                            # --- END: SCALE COORDINATES BACK ---

                            # Store the *original* scaled coordinates
                            roi_box_coords = ((original_x1, original_y1), (original_x2, original_y2))
                            st.session_state.roi_box_coords = roi_box_coords
                            st.session_state.roi_defined_by_canvas = True
                            # Optionally display coords for confirmation (use original values)
                            # st.write(f"Reference Box Selected (Original Coords): ({original_x1:.0f}, {original_y1:.0f}) to ({original_x2:.0f}, {original_y2:.0f})")
                        else:
                             # If box is too small or user finished drawing but result is invalid
                             if st.session_state.roi_defined_by_canvas: # Only reset if it was previously valid
                                 st.session_state.roi_defined_by_canvas = False
                                 if 'roi_box_coords' in st.session_state: del st.session_state.roi_box_coords
                             st.warning("Please draw a valid rectangle for the reference object (minimum size required).")

                    elif not objects and st.session_state.roi_defined_by_canvas:
                        # Handle case where user deletes the rectangle
                        st.session_state.roi_defined_by_canvas = False
                        if 'roi_box_coords' in st.session_state: del st.session_state.roi_box_coords
                        st.info("Reference rectangle cleared.")

                # If no valid rectangle drawn yet
                if not st.session_state.roi_defined_by_canvas:
                    st.info("Draw the reference rectangle on the image above.")

            else:
                st.warning("Waiting for image data to initialize drawing canvas.")


        # --- Step 3: Measure Objects ---
        if st.session_state.roi_defined_by_canvas:
            st.subheader("4. Measure Objects")
             # Make sure essential data for measurement is present
            if 'depth_map' in st.session_state and \
               'roi_box_coords' in st.session_state and \
               'original_width' in st.session_state and \
               'original_height' in st.session_state:

                # Disable button if results are already calculated OR if ROI is no longer defined
                measure_disabled = st.session_state.results_calculated or not st.session_state.roi_defined_by_canvas
                if st.button("Measure", key="measure_btn", disabled=measure_disabled):
                    with st.spinner("📏 Measuring objects..."):
                        # --- Scale Factor Calculation ---
                        # Uses st.session_state.roi_box_coords which now hold ORIGINAL scaled coordinates
                        roi_box_coords = st.session_state.roi_box_coords
                        # Ensure ROI coordinates are rounded for pixel access but retain original scale relationship
                        roi_x1, roi_y1 = roi_box_coords[0]
                        roi_x2, roi_y2 = roi_box_coords[1]

                        # Get the width *in pixels on the original image scale*
                        object_width_pixels_ref = round(roi_x2) - round(roi_x1)

                        # Ensure ROI coordinates are integers and within bounds *for depth map slicing*
                        # We use floor/ceil or round appropriately to capture the intended area on the original depth map
                        roi_y1_idx, roi_y2_idx = max(0, int(round(roi_y1))), min(st.session_state.original_height, int(round(roi_y2)))
                        roi_x1_idx, roi_x2_idx = max(0, int(round(roi_x1))), min(st.session_state.original_width, int(round(roi_x2)))

                        # Check pixel width again after rounding/clamping for index usage
                        if object_width_pixels_ref <= 0 or roi_y2_idx <= roi_y1_idx or roi_x2_idx <= roi_x1_idx:
                            st.error("Invalid Reference Object dimensions selected (width or height is zero or negative after scaling/rounding). Please redraw the yellow box.")
                            st.stop() # Stop processing this step

                        # Extract depth values within the ROI using the calculated indices
                        roi_depth_values = st.session_state.depth_map[roi_y1_idx:roi_y2_idx, roi_x1_idx:roi_x2_idx]
                        valid_roi_depths = roi_depth_values[~np.isnan(roi_depth_values) & (roi_depth_values > 1e-9)] # Use epsilon

                        if valid_roi_depths.size == 0:
                            st.error("Could not get valid depth data for the Reference Object ROI. Try redrawing the box slightly or check depth map quality.")
                            st.stop() # Stop processing this step

                        # Use median depth for robustness against outliers
                        d_relative_ref = np.median(valid_roi_depths)
                        if d_relative_ref <= 1e-9: # Use a small epsilon
                            st.error(f"Relative depth of the reference object ({d_relative_ref:.2e}) is too small or zero. Check ROI placement.")
                            st.stop() # Stop processing this step

                        # Calculate Scale Factor (Metric Depth = scale_factor / relative_depth)
                        # This part remains the same, using object_width_pixels_ref which is scaled correctly
                        scale_factor = -1 # Initialize
                        estimated_distance_ref = -1 # Initialize

                        try:
                            if USE_FOCAL_LENGTH_METHOD:
                                if KNOWN_FOCAL_LENGTH_PIXELS is None or KNOWN_FOCAL_LENGTH_PIXELS <= 0:
                                    st.error("Focal length is not set or invalid. Please check sidebar settings.")
                                    st.stop()
                                scale_factor = (KNOWN_OBJECT_WIDTH_METERS * KNOWN_FOCAL_LENGTH_PIXELS * d_relative_ref) / object_width_pixels_ref
                                estimated_distance_ref = scale_factor / d_relative_ref # D = scale / d

                            else: # Use FoV method
                                if HFOV_DEGREES is None or HFOV_DEGREES <= 0 or VFOV_DEGREES is None or VFOV_DEGREES <=0 or st.session_state.original_width <= 0:
                                    st.error("Horizontal/Vertical FoV or image width is not set or invalid. Please check sidebar settings.")
                                    st.stop()
                                hfov_radians = math.radians(HFOV_DEGREES)
                                tan_half_hfov = math.tan(hfov_radians / 2)
                                if abs(tan_half_hfov) <= 1e-9:
                                    st.error("Invalid Horizontal FoV resulting in tan(HFOV/2) near zero.")
                                    st.stop()
                                scale_factor = (KNOWN_OBJECT_WIDTH_METERS * st.session_state.original_width * d_relative_ref) / \
                                            (object_width_pixels_ref * 2 * tan_half_hfov)
                                estimated_distance_ref = scale_factor / d_relative_ref # D = scale / d

                            if scale_factor <= 0 or estimated_distance_ref <= 0 or not np.isfinite(scale_factor) or not np.isfinite(estimated_distance_ref):
                                st.error(f"Failed to calculate a valid scale factor ({scale_factor=}) or reference distance ({estimated_distance_ref=}). Check parameters and ROI.")
                                st.stop()

                            st.session_state.scale_factor = scale_factor
                            st.session_state.estimated_distance_ref = estimated_distance_ref

                        except ZeroDivisionError:
                            st.error("Calculation error: Division by zero. Check reference object width in pixels (is it > 0?) or tan(FoV/2).")
                            st.stop()
                        except Exception as e:
                            st.error(f"Error calculating scale factor: {e}")
                            st.stop()

                        # --- Detect Other Objects ---
                        # (Detection logic remains the same, operating on the full-res depth map)
                        detected_boxes = multistage_object_detection(
                            st.session_state.depth_map,
                            gradient_thresh=gradient_threshold,
                            min_size=min_object_size,
                            max_num=max_objects,
                            padding_ratio=object_padding_ratio
                        )

                        # --- Filter Detected Objects (Remove Reference, large overlaps, etc.) ---
                        # (Filtering logic remains the same, comparing against original scale roi_box_coords)
                        filtered_boxes_info = []
                        if detected_boxes:
                            iou_threshold_ref = 0.3 # Don't keep boxes overlapping significantly with Ref ROI
                            iou_threshold_overlap = 0.5 # Don't keep boxes overlapping significantly with each other (NMS like)
                            min_box_area = 50 # Minimum pixel area for a detected box
                            max_box_area_ratio = 0.95 # Maximum image area ratio

                            candidate_boxes_with_depth = []

                            # Calculate median depth for each detected box first
                            for i, det_box in enumerate(detected_boxes):
                                # Ensure coords are rounded for index usage
                                x1, y1 = round(det_box[0][0]), round(det_box[0][1])
                                x2, y2 = round(det_box[1][0]), round(det_box[1][1])
                                y1_c, y2_c = max(0, int(y1)), min(st.session_state.original_height, int(y2))
                                x1_c, x2_c = max(0, int(x1)), min(st.session_state.original_width, int(x2))

                                if x2_c <= x1_c or y2_c <= y1_c: continue # Skip invalid boxes

                                box_area = (x2_c - x1_c) * (y2_c - y1_c)
                                image_area = st.session_state.original_width * st.session_state.original_height
                                if box_area < min_box_area or (image_area > 0 and (box_area / image_area) > max_box_area_ratio):
                                    continue # Skip too small or too large boxes

                                det_depth_values = st.session_state.depth_map[y1_c:y2_c, x1_c:x2_c]
                                valid_det_depths = det_depth_values[~np.isnan(det_depth_values) & (det_depth_values > 1e-9)]
                                if valid_det_depths.size == 0: continue # Skip if no valid depth

                                median_depth = np.median(valid_det_depths)
                                if median_depth <= 1e-9: continue # Skip invalid relative depth

                                # Store original float box coords for accuracy, but use clamped ints for depth calc
                                candidate_boxes_with_depth.append({'index': i, 'box': det_box, 'median_depth': median_depth})

                            # Sort candidates by depth (closer first) - helps in NMS-like filtering
                            candidate_boxes_with_depth.sort(key=lambda item: item['median_depth'])

                            # Filter based on IoU
                            final_filtered_boxes = []
                            processed_indices = set()
                            ref_box_tuple = st.session_state.roi_box_coords # Get the reference box (original coords)

                            for i in range(len(candidate_boxes_with_depth)):
                                if i in processed_indices:
                                    continue

                                current_info = candidate_boxes_with_depth[i]
                                current_box = current_info['box']

                                # Check IoU with reference box (using original scale coords)
                                try:
                                    iou_ref = calculate_iou(current_box, ref_box_tuple)
                                except Exception as e_iou_ref:
                                    st.warning(f"Could not calculate IoU with reference for box {i}: {e_iou_ref}")
                                    iou_ref = 0 # Assume no overlap if calc fails

                                if iou_ref > iou_threshold_ref:
                                    processed_indices.add(i)
                                    continue # Skip if it overlaps too much with reference

                                # NMS step: Check IoU with other *already selected* boxes
                                suppress = False
                                for selected_box_info in final_filtered_boxes:
                                    try:
                                        iou_nms = calculate_iou(current_box, selected_box_info['bbox'])
                                    except Exception as e_iou_nms:
                                        st.warning(f"Could not calculate IoU between boxes {i} and selected: {e_iou_nms}")
                                        iou_nms = 0 # Assume no overlap if calc fails

                                    if iou_nms > iou_threshold_overlap:
                                        suppress = True
                                        break # Suppress this box if it significantly overlaps with a *closer* selected one

                                if not suppress:
                                    # If not suppressed, add it to the final list
                                    final_filtered_boxes.append({
                                        'bbox': current_box,
                                        'depth_relative': current_info['median_depth']
                                    })
                                    processed_indices.add(i) # Mark as processed/selected

                            filtered_boxes_info = final_filtered_boxes


                        # --- Calculate Dimensions for Filtered Objects ---
                        # (Dimension calculation logic remains the same, using original scale bbox)
                        object_analysis_results = []
                        if filtered_boxes_info:
                            scale_factor = st.session_state.scale_factor # Retrieve calculated scale factor
                            for i, box_info in enumerate(filtered_boxes_info):
                                bbox = box_info['bbox'] # These are original scale coords
                                d_relative_detected = box_info['depth_relative']

                                if d_relative_detected <= 1e-9: continue # Should have been filtered already, but double check

                                # Calculate absolute depth
                                absolute_depth_meters = scale_factor / d_relative_detected

                                # Calculate real-world dimensions using original image width/height
                                try:
                                    width_m, height_m = calculate_real_world_dimensions(
                                        absolute_depth_meters,
                                        bbox, # Use original scale bbox
                                        st.session_state.original_width,
                                        st.session_state.original_height,
                                        focal_length_pixels=KNOWN_FOCAL_LENGTH_PIXELS if USE_FOCAL_LENGTH_METHOD else None,
                                        hfov_degrees=HFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None,
                                        vfov_degrees=VFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None
                                    )

                                    # Basic sanity check on dimensions
                                    # Ensure dimensions and depth are positive and finite, and within reasonable bounds
                                    if np.isfinite(width_m) and np.isfinite(height_m) and np.isfinite(absolute_depth_meters) and \
                                        0 < width_m < 100 and 0 < height_m < 100 and 0 < absolute_depth_meters < 200:
                                        object_info = {
                                            'index': i + 1, # User-friendly 1-based index
                                            'bbox': bbox, # Store original scale bbox
                                            'depth_meters': absolute_depth_meters,
                                            'width_meters': width_m,
                                            'height_meters': height_m,
                                        }
                                        object_analysis_results.append(object_info)
                                    else:
                                        st.warning(f"Object {i+1} dimensions ({width_m*100:.1f}x{height_m*100:.1f}cm @ {absolute_depth_meters:.2f}m) seem unrealistic or invalid and were filtered out.")

                                except ValueError as e:
                                    st.error(f"Dimension calculation error for Object {i+1}: {e}")
                                except Exception as e_calc:
                                    st.error(f"Unexpected error during dimension calculation for Object {i+1}: {e_calc}")

                        st.session_state.object_analysis_results = object_analysis_results
                        st.session_state.results_calculated = True
                        st.experimental_rerun() # experimental_rerun to display results

                else:
                    # This else corresponds to the `if 'depth_map' in st.session_state...` check
                    st.info("Waiting for depth map and ROI selection to be ready for measurement.")


            # --- Display Measurement Results ---
            if st.session_state.results_calculated:
                st.subheader("5. Measurement Results")

                # Reference object info
                if 'estimated_distance_ref' in st.session_state:
                     st.success(f"Reference Object Estimated Distance: **{st.session_state.estimated_distance_ref:.2f} meters**")
                else:
                    st.warning("Reference object distance could not be estimated.")

                # Detected objects info
                if 'object_analysis_results' in st.session_state and st.session_state.object_analysis_results:
                    results = st.session_state.object_analysis_results
                    st.write(f"Found **{len(results)}** object(s) (excluding reference):")

                    # Display results in a table for better readability
                    import pandas as pd
                    display_data = []
                    for obj in results:
                        display_data.append({
                            "Object #": obj['index'],
                            "Distance (m)": f"{obj['depth_meters']:.2f}",
                            "Width (cm)": f"{obj['width_meters']*100:.1f}",
                            "Height (cm)": f"{obj['height_meters']*100:.1f}"
                        })
                    df = pd.DataFrame(display_data)
                    st.dataframe(df.set_index("Object #"), use_container_width=True)

                    # --- Visualization ---
                    st.subheader("Visualization")
                    st.markdown("Measurements displayed on the original image.")
                    with st.spinner("🎨 Generating visualization..."):
                        try:
                             # Make sure image_bgr and roi_box_coords (original scale) are available
                             if 'image_bgr' in st.session_state and 'roi_box_coords' in st.session_state:
                                visualization = visualize_objects_with_dimensions(
                                    st.session_state.image_bgr, # Use original BGR image
                                    results, # Results contain original scale bboxes
                                    st.session_state.roi_box_coords # Use original scale ROI coords
                                )
                                st.session_state.visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB) # Store as RGB for display
                             else:
                                st.warning("Could not generate visualization due to missing image or ROI data.")
                                if 'image_rgb' in st.session_state:
                                    st.session_state.visualization = st.session_state.image_rgb # Fallback to original
                                else:
                                    st.session_state.visualization = None # No image to show

                        except Exception as e_vis:
                            st.error(f"Error generating visualization: {e_vis}")
                            if 'image_rgb' in st.session_state:
                                st.session_state.visualization = st.session_state.image_rgb # Fallback to original
                            else:
                                st.session_state.visualization = None # No image to show


                    if 'visualization' in st.session_state and st.session_state.visualization is not None:
                        st.image(st.session_state.visualization, caption="Image with Measured Objects", use_column_width=True)
                    elif 'visualization' not in st.session_state :
                         st.warning("Visualization could not be generated.")


                else:
                    st.warning("No other objects were detected or met the filtering criteria.")

                # Add a button to clear results and allow remeasuring
                if st.button("Reset Measurements", key="reset_measurements"):
                    st.session_state.results_calculated = False
                    # Keep depth map and ROI drawing, but clear measurements and visualization
                    keys_to_clear_on_reset = ['object_analysis_results', 'visualization', 'scale_factor', 'estimated_distance_ref']
                    for key in keys_to_clear_on_reset:
                         if key in st.session_state: del st.session_state[key]
                    # We DON'T clear roi_box_coords or roi_defined_by_canvas here,
                    # allowing the user to re-measure with the same ROI if desired.
                    # display_scale_factor also remains as it depends only on the image.
                    st.experimental_rerun()


    else:
        # Show placeholder or instructions if no image is uploaded
        st.info("Upload an image using the button above to begin.")
        # Clear all state if no file is present
        keys_to_clear_on_no_file = ['depth_estimated', 'roi_defined_by_canvas', 'results_calculated', 'uploaded_file_id',
                         'depth_map', 'depth_colored', 'image_bgr', 'image_rgb', 'image_pil', 'original_width', 'original_height',
                         'object_analysis_results', 'roi_box_coords', 'visualization', 'scale_factor', 'estimated_distance_ref',
                         'display_scale_factor'] # Clear scale factor too
        state_cleared = False
        for key in keys_to_clear_on_no_file:
            if key in st.session_state:
                del st.session_state[key]
                state_cleared = True
        # Optionally experimental_rerun if state was cleared to ensure UI consistency
        # if state_cleared:
        #    st.experimental_rerun()


# --- Run the App ---
if __name__ == "__main__":
    main()