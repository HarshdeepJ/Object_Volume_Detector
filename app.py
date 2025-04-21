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

# --- Keep ALL previous code *before* the main() function ---
# (Imports, warnings, page_config, helper functions, sidebar config)
# --- Model Loading (Keep as is) ---
@st.cache_resource
def load_model():
    """Load the depth estimation model."""
    # ... (load_model function content remains the same) ...
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

# --- Other Helper Functions (Keep as is) ---
# predict_depth, normalize_depth_for_display, calculate_iou,
# detect_objects..., calculate_real_world_dimensions, visualize_objects_with_dimensions

# --- Sidebar Configuration (Keep as is) ---
# ... (Sidebar code remains the same) ...

# ... (rest of sidebar camera/advanced settings)


# --- Modified Main App Logic ---
def main():
    # --- Initialization and State Management (Expanded for Two Views) ---
    views = ['front', 'side']
    for view in views:
        # Initialize state keys for each view if they don't exist
        if f'depth_estimated_{view}' not in st.session_state: st.session_state[f'depth_estimated_{view}'] = False
        if f'roi_defined_{view}' not in st.session_state: st.session_state[f'roi_defined_{view}'] = False
        if f'results_calculated_{view}' not in st.session_state: st.session_state[f'results_calculated_{view}'] = False
        if f'uploaded_file_id_{view}' not in st.session_state: st.session_state[f'uploaded_file_id_{view}'] = None

    # --- Model Loading --- (Done once)
    processor, model, device = load_model()
    if not all([processor, model, device]):
        st.stop() # Stop execution if model loading failed

    # --- Update Title and Description for Two Views ---
    st.title("📏 3D Object Measurement Tool (Two Views)")
    st.markdown("""
    Estimate Width, Height, and Depth using **two images**: one front view and one side view.
    1.  **Upload Images:** Upload both Front and Side views using the buttons below.
    2.  **Configuration:** Set the reference object's known dimension and camera parameters in the sidebar. The reference object (or one with the same known dimension) must be clearly visible and selectable in **both** images.
    3.  **Process Each View:** For *both* the Front and Side views displayed below:
        *   Click "Estimate Depth".
        *   Draw a rectangle on the image to select the Reference Object.
        *   Click "Measure [View]" to calculate dimensions for that specific view.
    4.  **Combined Results:** Once *both* views have been successfully measured, the final 3D dimensions will be displayed at the bottom.
    """)

    # --- Image Upload (Two Uploaders) ---
    st.subheader("1. Upload Images")
    col1_upload, col2_upload = st.columns(2)
    with col1_upload:
        uploaded_file_front = st.file_uploader("Upload Front View Image", type=["jpg", "jpeg", "png"], key="upload_front")
    with col2_upload:
        uploaded_file_side = st.file_uploader("Upload Side View Image", type=["jpg", "jpeg", "png"], key="upload_side")

    # --- Processing Sections (using columns) ---
    st.subheader("2. Process Images (Estimate Depth, Select Reference, Measure)")
    col_proc1, col_proc2 = st.columns(2)

    # ===========================================================
    # ======= FRONT VIEW PROCESSING (Revised Logic) =============
    # ===========================================================
    with col_proc1:
        st.markdown("#### Front View")
        view = 'front' # Define current view context
        uploaded_file = uploaded_file_front # Use specific variable

        # --- Upload Handling & Image Loading (Front) ---
        if uploaded_file is not None:
            new_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            # Reset state ONLY if file changed for this view
            if st.session_state.get(f'uploaded_file_id_{view}') != new_file_id:
                st.session_state[f'uploaded_file_id_{view}'] = new_file_id
                # **Explicitly list keys to clear for THIS view**
                keys_to_clear = [
                    f'depth_estimated_{view}', f'roi_defined_{view}', f'results_calculated_{view}',
                    f'depth_map_{view}', f'depth_colored_{view}', f'image_bgr_{view}', f'image_rgb_{view}',
                    f'image_pil_{view}', f'original_width_{view}', f'original_height_{view}',
                    f'object_analysis_results_{view}', f'roi_box_coords_{view}', f'visualization_{view}',
                    f'scale_factor_{view}', f'estimated_distance_ref_{view}', f'display_scale_factor_{view}'
                ]
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
                # Ensure base flags are False after clearing
                st.session_state[f'depth_estimated_{view}'] = False
                st.session_state[f'roi_defined_{view}'] = False
                st.session_state[f'results_calculated_{view}'] = False
                st.info(f"New {view.capitalize()} view image uploaded. State reset for this view.")
                # Don't rerun here, let the rest of the script execute with the new file

            # Load image data if it doesn't exist in state for this view
            if f'image_pil_{view}' not in st.session_state:
                try:
                    uploaded_file.seek(0) # Use the specific upload variable
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if image_bgr is None: raise ValueError("Could not decode image")
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    st.session_state[f'image_bgr_{view}'] = image_bgr
                    st.session_state[f'image_rgb_{view}'] = image_rgb
                    st.session_state[f'image_pil_{view}'] = image_pil
                    st.session_state[f'original_width_{view}'] = image_pil.width
                    st.session_state[f'original_height_{view}'] = image_pil.height
                except Exception as e:
                    st.error(f"Error loading {view.capitalize()} image: {e}")
                    st.session_state[f'uploaded_file_id_{view}'] = None # Invalidate file ID on error
                    # Clear potentially partially loaded image data
                    for key in [f'image_bgr_{view}', f'image_rgb_{view}', f'image_pil_{view}']:
                         if key in st.session_state: del st.session_state[key]

            # Display uploaded image (if loaded successfully)
            if f'image_rgb_{view}' in st.session_state:
                 st.image(st.session_state[f'image_rgb_{view}'], caption=f"{view.capitalize()} View Uploaded", use_column_width=True)


            # --- Step 1: Depth Estimation (Front) ---
            # Enable button only if image is loaded AND depth not yet estimated
            depth_button_disabled = not (f'image_pil_{view}' in st.session_state) or st.session_state.get(f'depth_estimated_{view}', False)
            if st.button(f"Estimate Depth ({view.capitalize()})", key=f"estimate_depth_{view}", disabled=depth_button_disabled):
                if f'image_pil_{view}' in st.session_state: # Double check image exists before predicting
                    with st.spinner(f"Estimating {view.capitalize()} depth..."):
                        relative_depth_map = predict_depth(st.session_state[f'image_pil_{view}'], processor, model, device)
                        if relative_depth_map is not None:
                            st.session_state[f'depth_map_{view}'] = relative_depth_map
                            st.session_state[f'depth_colored_{view}'] = normalize_depth_for_display(relative_depth_map)
                            st.session_state[f'depth_estimated_{view}'] = True
                            # Reset subsequent steps for this view upon successful estimation
                            st.session_state[f'roi_defined_{view}'] = False
                            st.session_state[f'results_calculated_{view}'] = False
                            # Clear potentially stale ROI/results data
                            for key in [f'roi_box_coords_{view}', f'object_analysis_results_{view}', f'visualization_{view}', f'scale_factor_{view}', f'estimated_distance_ref_{view}', f'display_scale_factor_{view}']:
                                if key in st.session_state: del st.session_state[key]
                            st.experimental_rerun() # Rerun to update UI state (e.g., enable ROI)
                        else:
                            st.error(f"{view.capitalize()} Depth estimation failed.")
                else:
                    st.warning(f"Cannot estimate depth, {view.capitalize()} image data missing.")

            # Display depth map if estimated
            if st.session_state.get(f'depth_estimated_{view}', False) and f'depth_colored_{view}' in st.session_state:
                st.image(st.session_state[f'depth_colored_{view}'], caption=f"{view.capitalize()} Depth Map", use_column_width=True)
            elif not (f'image_pil_{view}' in st.session_state): # If image is not even loaded
                 pass # No need to show a message here, handled by button state


            # --- Step 2: Reference Object Selection (Front) ---
            if st.session_state.get(f'depth_estimated_{view}', False): # Only show if depth is done
                st.markdown(f"**Select Reference ({view.capitalize()})** - Draw rectangle (Known Dim: {KNOWN_OBJECT_WIDTH_METERS*100:.1f} cm)")
                if f'image_pil_{view}' in st.session_state: # Check necessary data exists
                    # ... (Keep the Resizing Logic and Canvas setup exactly as before) ...
                    MAX_CANVAS_DISPLAY_WIDTH_VIEW = 600
                    original_pil_image = st.session_state[f'image_pil_{view}']
                    original_width = st.session_state[f'original_width_{view}']
                    original_height = st.session_state[f'original_height_{view}']
                    if original_width > MAX_CANVAS_DISPLAY_WIDTH_VIEW:
                        display_scale_factor = MAX_CANVAS_DISPLAY_WIDTH_VIEW / original_width
                        display_width = MAX_CANVAS_DISPLAY_WIDTH_VIEW
                        display_height = int(original_height * display_scale_factor)
                        try: display_image = original_pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                        except AttributeError: display_image = original_pil_image.resize((display_width, display_height), Image.LANCZOS)
                        st.session_state[f'display_scale_factor_{view}'] = display_scale_factor
                    else:
                        display_image = original_pil_image; display_width = original_width; display_height = original_height
                        st.session_state[f'display_scale_factor_{view}'] = 1.0

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.1)", stroke_width=3, stroke_color="#FFFF00",
                        background_image=display_image, update_streamlit=True,
                        height=display_height, width=display_width, drawing_mode="rect",
                        key=f"canvas_{view}",
                    )
                    # --- Process Canvas Result (Front) ---
                    # ... (Keep the canvas result processing logic exactly as before, using `view` key) ...
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects and objects[-1]["type"] == "rect":
                            # ... (rest of rect processing, setting roi_box_coords_{view}, roi_defined_{view}) ...
                            rect_data = objects[-1]
                            display_left = rect_data["left"]; display_top = rect_data["top"]
                            display_width_drawn = rect_data["width"]; display_height_drawn = rect_data["height"]
                            if display_width_drawn > 5 and display_height_drawn > 5:
                                scale_f = st.session_state.get(f'display_scale_factor_{view}', 1.0)
                                ox1 = display_left / scale_f; oy1 = display_top / scale_f
                                ox2 = (display_left + display_width_drawn) / scale_f
                                oy2 = (display_top + display_height_drawn) / scale_f
                                roi_coords = ((ox1, oy1), (ox2, oy2))
                                old_roi = st.session_state.get(f'roi_box_coords_{view}')
                                roi_changed = old_roi != roi_coords
                                st.session_state[f'roi_box_coords_{view}'] = roi_coords
                                if not st.session_state.get(f'roi_defined_{view}', False) or roi_changed:
                                    st.session_state[f'roi_defined_{view}'] = True
                                    st.session_state[f'results_calculated_{view}'] = False # Reset results if ROI changed
                                    st.success(f"{view.capitalize()} Reference Selected/Updated.")
                            else: # Box too small
                                if st.session_state.get(f'roi_defined_{view}', False): # Only reset if it was defined
                                    st.session_state[f'roi_defined_{view}'] = False
                                    if f'roi_box_coords_{view}' in st.session_state: del st.session_state[f'roi_box_coords_{view}']
                                    st.warning(f"{view.capitalize()} Ref rectangle too small.")
                        elif not objects and st.session_state.get(f'roi_defined_{view}', False): # Box deleted
                            st.session_state[f'roi_defined_{view}'] = False
                            if f'roi_box_coords_{view}' in st.session_state: del st.session_state[f'roi_box_coords_{view}']
                            st.info(f"{view.capitalize()} Reference rectangle cleared.")

                    # Show guidance if ROI not defined
                    if not st.session_state.get(f'roi_defined_{view}', False):
                        st.info(f"Draw the reference rectangle on the {view.capitalize()} view image above.")
                else:
                    st.warning(f"Waiting for {view.capitalize()} image data for canvas.")


            # --- Step 3: Measure Objects (Front) ---
            if st.session_state.get(f'roi_defined_{view}', False): # Only show if ROI is drawn
                # Check if all necessary data is present before enabling button
                measure_ready = all(k in st.session_state for k in [f'depth_map_{view}', f'roi_box_coords_{view}', f'original_width_{view}', f'original_height_{view}'])
                measure_button_disabled = not measure_ready or st.session_state.get(f'results_calculated_{view}', False)

                if st.button(f"Measure ({view.capitalize()})", key=f"measure_{view}", disabled=measure_button_disabled):
                    if measure_ready: # Double check before processing
                        with st.spinner(f"Measuring {view.capitalize()} view objects..."):
                            try: # Wrap calculation in try-except
                                # --- Calculation Logic (Scale Factor, Detect, Filter, Dimensions) ---
                                # ... (Keep the entire calculation block exactly as before, using `view` key) ...
                                roi_box_coords = st.session_state[f'roi_box_coords_{view}']
                                roi_x1, roi_y1 = roi_box_coords[0]; roi_x2, roi_y2 = roi_box_coords[1]
                                object_width_pixels_ref = round(roi_x2) - round(roi_x1)
                                roi_y1_idx = max(0, int(round(roi_y1))); roi_y2_idx = min(st.session_state[f'original_height_{view}'], int(round(roi_y2)))
                                roi_x1_idx = max(0, int(round(roi_x1))); roi_x2_idx = min(st.session_state[f'original_width_{view}'], int(round(roi_x2)))
                                if object_width_pixels_ref <= 0 or roi_y2_idx <= roi_y1_idx or roi_x2_idx <= roi_x1_idx: raise ValueError("Invalid Reference Object dimensions.")
                                roi_depth_values = st.session_state[f'depth_map_{view}'][roi_y1_idx:roi_y2_idx, roi_x1_idx:roi_x2_idx]
                                valid_roi_depths = roi_depth_values[~np.isnan(roi_depth_values) & (roi_depth_values > 1e-9)]
                                if valid_roi_depths.size == 0: raise ValueError("No valid depth for Reference ROI.")
                                d_relative_ref = np.median(valid_roi_depths)
                                if d_relative_ref <= 1e-9: raise ValueError("Reference depth is too small.")
                                scale_factor = -1; estimated_distance_ref = -1
                                img_width = st.session_state[f'original_width_{view}']; img_height = st.session_state[f'original_height_{view}']
                                if USE_FOCAL_LENGTH_METHOD:
                                    if KNOWN_FOCAL_LENGTH_PIXELS <= 0: raise ValueError("Focal length invalid.")
                                    scale_factor = (KNOWN_OBJECT_WIDTH_METERS * KNOWN_FOCAL_LENGTH_PIXELS * d_relative_ref) / object_width_pixels_ref
                                    estimated_distance_ref = scale_factor / d_relative_ref
                                else: # FoV method
                                    if HFOV_DEGREES <= 0 or VFOV_DEGREES <=0 or img_width <= 0: raise ValueError("FoV/Image params invalid.")
                                    hfov_radians = math.radians(HFOV_DEGREES); tan_half_hfov = math.tan(hfov_radians / 2)
                                    if abs(tan_half_hfov) <= 1e-9: raise ValueError("tan(HFOV/2) near zero.")
                                    scale_factor = (KNOWN_OBJECT_WIDTH_METERS * img_width * d_relative_ref) / (object_width_pixels_ref * 2 * tan_half_hfov)
                                    estimated_distance_ref = scale_factor / d_relative_ref
                                if scale_factor <= 0 or estimated_distance_ref <= 0 or not np.isfinite(scale_factor) or not np.isfinite(estimated_distance_ref): raise ValueError(f"Invalid scale factor/ref distance")
                                st.session_state[f'scale_factor_{view}'] = scale_factor
                                st.session_state[f'estimated_distance_ref_{view}'] = estimated_distance_ref

                                detected_boxes = multistage_object_detection(st.session_state[f'depth_map_{view}'], gradient_threshold, min_object_size, max_objects, object_padding_ratio)
                                object_analysis_results = []
                                if detected_boxes:
                                    iou_threshold_ref = 0.3
                                    ref_box_tuple = st.session_state[f'roi_box_coords_{view}']
                                    for i, det_box in enumerate(detected_boxes):
                                        iou_ref = calculate_iou(det_box, ref_box_tuple)
                                        if iou_ref <= iou_threshold_ref:
                                            x1_det, y1_det = round(det_box[0][0]), round(det_box[0][1]); x2_det, y2_det = round(det_box[1][0]), round(det_box[1][1])
                                            y1_c, y2_c = max(0, int(y1_det)), min(img_height, int(y2_det)); x1_c, x2_c = max(0, int(x1_det)), min(img_width, int(x2_det))
                                            if x2_c <= x1_c or y2_c <= y1_c: continue
                                            det_depth_values = st.session_state[f'depth_map_{view}'][y1_c:y2_c, x1_c:x2_c]
                                            valid_det_depths = det_depth_values[~np.isnan(det_depth_values) & (det_depth_values > 1e-9)]
                                            if valid_det_depths.size > 0:
                                                median_depth_rel = np.median(valid_det_depths)
                                                if median_depth_rel > 1e-9:
                                                    abs_depth = scale_factor / median_depth_rel
                                                    width_m, height_m = calculate_real_world_dimensions(abs_depth, det_box, img_width, img_height, KNOWN_FOCAL_LENGTH_PIXELS if USE_FOCAL_LENGTH_METHOD else None, HFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None, VFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None)
                                                    if np.isfinite(width_m) and np.isfinite(height_m) and width_m > 0 and height_m > 0 and abs_depth > 0:
                                                        object_analysis_results.append({'index': 1, 'bbox': det_box, 'depth_meters': abs_depth, 'width_meters': width_m, 'height_meters': height_m})
                                                        break # Stop after first valid target

                                st.session_state[f'object_analysis_results_{view}'] = object_analysis_results
                                st.session_state[f'results_calculated_{view}'] = True

                                if object_analysis_results:
                                    vis_img = visualize_objects_with_dimensions(st.session_state[f'image_bgr_{view}'], object_analysis_results, st.session_state[f'roi_box_coords_{view}'])
                                    st.session_state[f'visualization_{view}'] = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

                                st.experimental_rerun() # Rerun to display results

                            except ValueError as e: st.error(f"{view.capitalize()} Measurement Error: {e}")
                            except ZeroDivisionError: st.error(f"{view.capitalize()} Measurement Error: Division by zero. Check inputs.")
                            except Exception as e: st.error(f"Unexpected error during {view.capitalize()} measurement: {e}")
                    else:
                         st.warning(f"Cannot measure, required data for {view.capitalize()} view is missing.")


            # --- Display Measurement Results (Front) ---
            if st.session_state.get(f'results_calculated_{view}', False):
                st.markdown(f"**{view.capitalize()} View Results:**")
                 # ... (Keep the results display and reset button logic exactly as before, using `view` key) ...
                if f'estimated_distance_ref_{view}' in st.session_state:
                     st.write(f"Ref. Distance: {st.session_state[f'estimated_distance_ref_{view}']:.2f} m")
                results = st.session_state.get(f'object_analysis_results_{view}', [])
                if results:
                    obj = results[0]
                    st.write(f"Target Object ({view.capitalize()}):")
                    st.write(f"- Dist: {obj['depth_meters']:.2f} m")
                    st.write(f"- W: {obj['width_meters']*100:.1f} cm")
                    st.write(f"- H: {obj['height_meters']*100:.1f} cm")
                    if f'visualization_{view}' in st.session_state:
                        st.image(st.session_state[f'visualization_{view}'], caption=f"{view.capitalize()} View Measurements", use_column_width=True)
                else: st.warning(f"No valid target object found in {view.capitalize()} view.")
                if st.button(f"Reset {view.capitalize()} Measurements", key=f"reset_{view}"):
                    st.session_state[f'results_calculated_{view}'] = False
                    keys_to_clear = [f'object_analysis_results_{view}', f'visualization_{view}', f'scale_factor_{view}', f'estimated_distance_ref_{view}']
                    for key in keys_to_clear:
                        if key in st.session_state: del st.session_state[key]
                    st.experimental_rerun()


        else: # No front image uploaded
            st.info("Upload Front View image using the button above.")


    # ===========================================================
    # ======== SIDE VIEW PROCESSING (Revised Logic) =============
    # ===========================================================
    with col_proc2:
        st.markdown("#### Side View")
        view = 'side' # Define current view context
        uploaded_file = uploaded_file_side # Use specific variable

        # --- Upload Handling & Image Loading (Side) ---
        # (Exact same logic as Front view, just using 'side' keys and `uploaded_file_side`)
        if uploaded_file is not None:
            new_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            if st.session_state.get(f'uploaded_file_id_{view}') != new_file_id:
                st.session_state[f'uploaded_file_id_{view}'] = new_file_id
                keys_to_clear = [
                    f'depth_estimated_{view}', f'roi_defined_{view}', f'results_calculated_{view}',
                    f'depth_map_{view}', f'depth_colored_{view}', f'image_bgr_{view}', f'image_rgb_{view}',
                    f'image_pil_{view}', f'original_width_{view}', f'original_height_{view}',
                    f'object_analysis_results_{view}', f'roi_box_coords_{view}', f'visualization_{view}',
                    f'scale_factor_{view}', f'estimated_distance_ref_{view}', f'display_scale_factor_{view}'
                ]
                for key in keys_to_clear:
                    if key in st.session_state: del st.session_state[key]
                st.session_state[f'depth_estimated_{view}'] = False
                st.session_state[f'roi_defined_{view}'] = False
                st.session_state[f'results_calculated_{view}'] = False
                st.info(f"New {view.capitalize()} view image uploaded. State reset for this view.")

            if f'image_pil_{view}' not in st.session_state:
                try:
                    uploaded_file.seek(0)
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if image_bgr is None: raise ValueError("Could not decode image")
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    st.session_state[f'image_bgr_{view}'] = image_bgr
                    st.session_state[f'image_rgb_{view}'] = image_rgb
                    st.session_state[f'image_pil_{view}'] = image_pil
                    st.session_state[f'original_width_{view}'] = image_pil.width
                    st.session_state[f'original_height_{view}'] = image_pil.height
                except Exception as e:
                    st.error(f"Error loading {view.capitalize()} image: {e}")
                    st.session_state[f'uploaded_file_id_{view}'] = None
                    for key in [f'image_bgr_{view}', f'image_rgb_{view}', f'image_pil_{view}']:
                         if key in st.session_state: del st.session_state[key]

            if f'image_rgb_{view}' in st.session_state:
                 st.image(st.session_state[f'image_rgb_{view}'], caption=f"{view.capitalize()} View Uploaded", use_column_width=True)

            # --- Step 1: Depth Estimation (Side) ---
            depth_button_disabled = not (f'image_pil_{view}' in st.session_state) or st.session_state.get(f'depth_estimated_{view}', False)
            if st.button(f"Estimate Depth ({view.capitalize()})", key=f"estimate_depth_{view}", disabled=depth_button_disabled):
                 if f'image_pil_{view}' in st.session_state:
                    with st.spinner(f"Estimating {view.capitalize()} depth..."):
                        # ... (predict_depth call and state setting, same as front) ...
                        relative_depth_map = predict_depth(st.session_state[f'image_pil_{view}'], processor, model, device)
                        if relative_depth_map is not None:
                            st.session_state[f'depth_map_{view}'] = relative_depth_map
                            st.session_state[f'depth_colored_{view}'] = normalize_depth_for_display(relative_depth_map)
                            st.session_state[f'depth_estimated_{view}'] = True
                            st.session_state[f'roi_defined_{view}'] = False
                            st.session_state[f'results_calculated_{view}'] = False
                            for key in [f'roi_box_coords_{view}', f'object_analysis_results_{view}', f'visualization_{view}', f'scale_factor_{view}', f'estimated_distance_ref_{view}', f'display_scale_factor_{view}']:
                                if key in st.session_state: del st.session_state[key]
                            st.experimental_rerun()
                        else: st.error(f"{view.capitalize()} Depth estimation failed.")
                 else: st.warning(f"Cannot estimate depth, {view.capitalize()} image data missing.")

            if st.session_state.get(f'depth_estimated_{view}', False) and f'depth_colored_{view}' in st.session_state:
                st.image(st.session_state[f'depth_colored_{view}'], caption=f"{view.capitalize()} Depth Map", use_column_width=True)

            # --- Step 2: Reference Object Selection (Side) ---
            if st.session_state.get(f'depth_estimated_{view}', False):
                st.markdown(f"**Select Reference ({view.capitalize()})** - Draw rectangle (Known Dim: {KNOWN_OBJECT_WIDTH_METERS*100:.1f} cm)")
                if f'image_pil_{view}' in st.session_state:
                    # ... (Keep the Resizing Logic and Canvas setup exactly as before, using `view` key) ...
                    MAX_CANVAS_DISPLAY_WIDTH_VIEW = 600
                    original_pil_image = st.session_state[f'image_pil_{view}']
                    original_width = st.session_state[f'original_width_{view}']
                    original_height = st.session_state[f'original_height_{view}']
                    if original_width > MAX_CANVAS_DISPLAY_WIDTH_VIEW:
                        display_scale_factor = MAX_CANVAS_DISPLAY_WIDTH_VIEW / original_width; display_width = MAX_CANVAS_DISPLAY_WIDTH_VIEW; display_height = int(original_height * display_scale_factor)
                        try: display_image = original_pil_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                        except AttributeError: display_image = original_pil_image.resize((display_width, display_height), Image.LANCZOS)
                        st.session_state[f'display_scale_factor_{view}'] = display_scale_factor
                    else:
                        display_image = original_pil_image; display_width = original_width; display_height = original_height
                        st.session_state[f'display_scale_factor_{view}'] = 1.0

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.1)", stroke_width=3, stroke_color="#FFFF00",
                        background_image=display_image, update_streamlit=True,
                        height=display_height, width=display_width, drawing_mode="rect", key=f"canvas_{view}",
                    )
                    # --- Process Canvas Result (Side) ---
                    # ... (Keep the canvas result processing logic exactly as before, using `view` key) ...
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects and objects[-1]["type"] == "rect":
                            rect_data = objects[-1]
                            display_left = rect_data["left"]; display_top = rect_data["top"]; display_width_drawn = rect_data["width"]; display_height_drawn = rect_data["height"]
                            if display_width_drawn > 5 and display_height_drawn > 5:
                                scale_f = st.session_state.get(f'display_scale_factor_{view}', 1.0)
                                ox1 = display_left / scale_f; oy1 = display_top / scale_f; ox2 = (display_left + display_width_drawn) / scale_f; oy2 = (display_top + display_height_drawn) / scale_f
                                roi_coords = ((ox1, oy1), (ox2, oy2))
                                old_roi = st.session_state.get(f'roi_box_coords_{view}')
                                roi_changed = old_roi != roi_coords
                                st.session_state[f'roi_box_coords_{view}'] = roi_coords
                                if not st.session_state.get(f'roi_defined_{view}', False) or roi_changed:
                                    st.session_state[f'roi_defined_{view}'] = True
                                    st.session_state[f'results_calculated_{view}'] = False
                                    st.success(f"{view.capitalize()} Reference Selected/Updated.")
                            else:
                                if st.session_state.get(f'roi_defined_{view}', False):
                                    st.session_state[f'roi_defined_{view}'] = False; del st.session_state[f'roi_box_coords_{view}']
                                    st.warning(f"{view.capitalize()} Ref rectangle too small.")
                        elif not objects and st.session_state.get(f'roi_defined_{view}', False):
                            st.session_state[f'roi_defined_{view}'] = False; del st.session_state[f'roi_box_coords_{view}']
                            st.info(f"{view.capitalize()} Reference rectangle cleared.")

                    if not st.session_state.get(f'roi_defined_{view}', False):
                        st.info(f"Draw the reference rectangle on the {view.capitalize()} view image above.")
                else: st.warning(f"Waiting for {view.capitalize()} image data for canvas.")


            # --- Step 3: Measure Objects (Side) ---
            if st.session_state.get(f'roi_defined_{view}', False):
                measure_ready = all(k in st.session_state for k in [f'depth_map_{view}', f'roi_box_coords_{view}', f'original_width_{view}', f'original_height_{view}'])
                measure_button_disabled = not measure_ready or st.session_state.get(f'results_calculated_{view}', False)

                if st.button(f"Measure ({view.capitalize()})", key=f"measure_{view}", disabled=measure_button_disabled):
                    if measure_ready:
                        with st.spinner(f"Measuring {view.capitalize()} view objects..."):
                            try:
                                # --- Calculation Logic (Scale Factor, Detect, Filter, Dimensions) ---
                                # ... (Keep the entire calculation block exactly as before, using `view` key) ...
                                roi_box_coords = st.session_state[f'roi_box_coords_{view}']; roi_x1, roi_y1 = roi_box_coords[0]; roi_x2, roi_y2 = roi_box_coords[1]
                                object_width_pixels_ref = round(roi_x2) - round(roi_x1)
                                roi_y1_idx = max(0, int(round(roi_y1))); roi_y2_idx = min(st.session_state[f'original_height_{view}'], int(round(roi_y2)))
                                roi_x1_idx = max(0, int(round(roi_x1))); roi_x2_idx = min(st.session_state[f'original_width_{view}'], int(round(roi_x2)))
                                if object_width_pixels_ref <= 0 or roi_y2_idx <= roi_y1_idx or roi_x2_idx <= roi_x1_idx: raise ValueError("Invalid Reference Object dimensions.")
                                roi_depth_values = st.session_state[f'depth_map_{view}'][roi_y1_idx:roi_y2_idx, roi_x1_idx:roi_x2_idx]
                                valid_roi_depths = roi_depth_values[~np.isnan(roi_depth_values) & (roi_depth_values > 1e-9)]
                                if valid_roi_depths.size == 0: raise ValueError("No valid depth for Reference ROI.")
                                d_relative_ref = np.median(valid_roi_depths)
                                if d_relative_ref <= 1e-9: raise ValueError("Reference depth is too small.")
                                scale_factor = -1; estimated_distance_ref = -1
                                img_width = st.session_state[f'original_width_{view}']; img_height = st.session_state[f'original_height_{view}']
                                if USE_FOCAL_LENGTH_METHOD:
                                    if KNOWN_FOCAL_LENGTH_PIXELS <= 0: raise ValueError("Focal length invalid.")
                                    scale_factor = (KNOWN_OBJECT_WIDTH_METERS * KNOWN_FOCAL_LENGTH_PIXELS * d_relative_ref) / object_width_pixels_ref
                                    estimated_distance_ref = scale_factor / d_relative_ref
                                else: # FoV method
                                    if HFOV_DEGREES <= 0 or VFOV_DEGREES <=0 or img_width <= 0: raise ValueError("FoV/Image params invalid.")
                                    hfov_radians = math.radians(HFOV_DEGREES); tan_half_hfov = math.tan(hfov_radians / 2)
                                    if abs(tan_half_hfov) <= 1e-9: raise ValueError("tan(HFOV/2) near zero.")
                                    scale_factor = (KNOWN_OBJECT_WIDTH_METERS * img_width * d_relative_ref) / (object_width_pixels_ref * 2 * tan_half_hfov)
                                    estimated_distance_ref = scale_factor / d_relative_ref
                                if scale_factor <= 0 or estimated_distance_ref <= 0 or not np.isfinite(scale_factor) or not np.isfinite(estimated_distance_ref): raise ValueError(f"Invalid scale factor/ref distance")
                                st.session_state[f'scale_factor_{view}'] = scale_factor
                                st.session_state[f'estimated_distance_ref_{view}'] = estimated_distance_ref

                                detected_boxes = multistage_object_detection(st.session_state[f'depth_map_{view}'], gradient_threshold, min_object_size, max_objects, object_padding_ratio)
                                object_analysis_results = []
                                if detected_boxes:
                                    iou_threshold_ref = 0.3
                                    ref_box_tuple = st.session_state[f'roi_box_coords_{view}']
                                    for i, det_box in enumerate(detected_boxes):
                                        iou_ref = calculate_iou(det_box, ref_box_tuple)
                                        if iou_ref <= iou_threshold_ref:
                                            x1_det, y1_det = round(det_box[0][0]), round(det_box[0][1]); x2_det, y2_det = round(det_box[1][0]), round(det_box[1][1])
                                            y1_c, y2_c = max(0, int(y1_det)), min(img_height, int(y2_det)); x1_c, x2_c = max(0, int(x1_det)), min(img_width, int(x2_det))
                                            if x2_c <= x1_c or y2_c <= y1_c: continue
                                            det_depth_values = st.session_state[f'depth_map_{view}'][y1_c:y2_c, x1_c:x2_c]
                                            valid_det_depths = det_depth_values[~np.isnan(det_depth_values) & (det_depth_values > 1e-9)]
                                            if valid_det_depths.size > 0:
                                                median_depth_rel = np.median(valid_det_depths)
                                                if median_depth_rel > 1e-9:
                                                    abs_depth = scale_factor / median_depth_rel
                                                    width_m, height_m = calculate_real_world_dimensions(abs_depth, det_box, img_width, img_height, KNOWN_FOCAL_LENGTH_PIXELS if USE_FOCAL_LENGTH_METHOD else None, HFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None, VFOV_DEGREES if not USE_FOCAL_LENGTH_METHOD else None)
                                                    if np.isfinite(width_m) and np.isfinite(height_m) and width_m > 0 and height_m > 0 and abs_depth > 0:
                                                        object_analysis_results.append({'index': 1, 'bbox': det_box, 'depth_meters': abs_depth, 'width_meters': width_m, 'height_meters': height_m})
                                                        break

                                st.session_state[f'object_analysis_results_{view}'] = object_analysis_results
                                st.session_state[f'results_calculated_{view}'] = True

                                if object_analysis_results:
                                    vis_img = visualize_objects_with_dimensions(st.session_state[f'image_bgr_{view}'], object_analysis_results, st.session_state[f'roi_box_coords_{view}'])
                                    st.session_state[f'visualization_{view}'] = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

                                st.experimental_rerun()

                            except ValueError as e: st.error(f"{view.capitalize()} Measurement Error: {e}")
                            except ZeroDivisionError: st.error(f"{view.capitalize()} Measurement Error: Division by zero. Check inputs.")
                            except Exception as e: st.error(f"Unexpected error during {view.capitalize()} measurement: {e}")
                    else:
                        st.warning(f"Cannot measure, required data for {view.capitalize()} view is missing.")


            # --- Display Measurement Results (Side) ---
            if st.session_state.get(f'results_calculated_{view}', False):
                st.markdown(f"**{view.capitalize()} View Results:**")
                # ... (Keep the results display and reset button logic exactly as before, using `view` key and labeling width as depth) ...
                if f'estimated_distance_ref_{view}' in st.session_state:
                     st.write(f"Ref. Distance: {st.session_state[f'estimated_distance_ref_{view}']:.2f} m")
                results = st.session_state.get(f'object_analysis_results_{view}', [])
                if results:
                    obj = results[0]
                    st.write(f"Target Object ({view.capitalize()}):")
                    st.write(f"- Dist: {obj['depth_meters']:.2f} m")
                    st.write(f"- W (Depth): {obj['width_meters']*100:.1f} cm") # Label width as depth
                    st.write(f"- H: {obj['height_meters']*100:.1f} cm")
                    if f'visualization_{view}' in st.session_state:
                        st.image(st.session_state[f'visualization_{view}'], caption=f"{view.capitalize()} View Measurements", use_column_width=True)
                else: st.warning(f"No valid target object found in {view.capitalize()} view.")
                if st.button(f"Reset {view.capitalize()} Measurements", key=f"reset_{view}"):
                    st.session_state[f'results_calculated_{view}'] = False
                    keys_to_clear = [f'object_analysis_results_{view}', f'visualization_{view}', f'scale_factor_{view}', f'estimated_distance_ref_{view}']
                    for key in keys_to_clear:
                        if key in st.session_state: del st.session_state[key]
                    st.experimental_rerun()

        else: # No side image uploaded
            st.info("Upload Side View image using the button above.")


    # --- Combined 3D Results Section ---
    st.divider() # Add a visual separator
    st.subheader("3. Final Combined 3D Dimensions")

    # Check if results from BOTH views are available and valid
    front_results_valid = (st.session_state.get('results_calculated_front', False) and
                           isinstance(st.session_state.get('object_analysis_results_front'), list) and
                           len(st.session_state.get('object_analysis_results_front', [])) > 0) # Check length safely
    side_results_valid = (st.session_state.get('results_calculated_side', False) and
                          isinstance(st.session_state.get('object_analysis_results_side'), list) and
                          len(st.session_state.get('object_analysis_results_side', [])) > 0) # Check length safely

    if front_results_valid and side_results_valid:
        try:
            # Extract dimensions from the *first* object found in each view's results list
            front_obj = st.session_state['object_analysis_results_front'][0]
            side_obj = st.session_state['object_analysis_results_side'][0]

            # Assign dimensions based on views
            final_width = front_obj['width_meters']
            final_height_front = front_obj['height_meters']
            final_depth = side_obj['width_meters']  # *** KEY ASSUMPTION ***
            final_height_side = side_obj['height_meters']

            # Average the height measurements
            final_height = (final_height_front + final_height_side) / 2.0

            # Use distance from the front view
            final_distance = front_obj['depth_meters']

            st.success("Combined 3D estimates:")
            # Display using columns for better layout
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric(label="Width (from Front)", value=f"{final_width*100:.1f} cm")
            with res_col2:
                 st.metric(label="Depth (from Side)", value=f"{final_depth*100:.1f} cm") # Clarify source
            with res_col3:
                 st.metric(label="Height (Avg)", value=f"{final_height*100:.1f} cm")
            with res_col4:
                 st.metric(label="Distance (Front)", value=f"{final_distance:.2f} m") # Clarify source

        except (KeyError, IndexError, TypeError) as e:
            st.error(f"Error processing final results: {e}. Ensure both views measured a target object.")
        except Exception as e:
            st.error(f"An unexpected error occurred during final result combination: {e}")

    elif not uploaded_file_front or not uploaded_file_side:
         st.info("Upload both Front and Side images first.")
    else:
        st.warning("Process and Measure both Front and Side views completely to see combined 3D results.")
        # Optionally show which view is missing results
        if not front_results_valid: st.warning("- Front view results missing or invalid.")
        if not side_results_valid: st.warning("- Side view results missing or invalid.")

# --- Run the App ---
if __name__ == "__main__":
    main()