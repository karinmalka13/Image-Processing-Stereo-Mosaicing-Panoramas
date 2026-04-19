import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import time

def create_dog_space(img, num_octaves=3, scales_per_octave=2, sigma=1.6):
    dog_pyramid = []
    k = 2 ** (1 / scales_per_octave)
    current_img = img.astype(np.float32)
    
    for o in range(num_octaves):
        octave_gaussians = []
        s_sigma = sigma
        for s in range(scales_per_octave + 3):
            octave_gaussians.append(gaussian_filter(current_img, s_sigma))
            s_sigma *= k
        dog_octave = [octave_gaussians[i+1] - octave_gaussians[i] 
                      for i in range(len(octave_gaussians)-1)]
        dog_pyramid.append(dog_octave)
        current_img = octave_gaussians[-3][::2, ::2]
        
    return dog_pyramid

def find_keypoints(dog_pyramid, threshold=0.04):
    keypoints = [] 
    for oct_idx, dog_octave in enumerate(dog_pyramid):
        for i in range(1, len(dog_octave) - 1):
            layer = dog_octave[i]
            prev = dog_octave[i-1]
            nxt = dog_octave[i+1]
            
            is_max = np.ones(layer.shape, dtype=bool)
            # 26 neighbors check
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1), (0,0)]:
                if dx == 0 and dy == 0:
                    is_max &= (layer >= prev) & (layer >= nxt)
                    continue
                shifted = np.roll(np.roll(layer, dx, axis=1), dy, axis=0)
                is_max &= (layer >= shifted)
                is_max &= (layer >= np.roll(np.roll(prev, dx, axis=1), dy, axis=0))
                is_max &= (layer >= np.roll(np.roll(nxt, dx, axis=1), dy, axis=0))

            is_max &= np.abs(layer) > threshold
            coords = np.argwhere(is_max)
            for y, x in coords:
                keypoints.append((y, x, oct_idx, i))
    return keypoints

def extract_sift_descriptors(dog_pyramid, keypoints, window_size=16):
    descriptors = []
    valid_keypoints = []

    for y, x, oct_idx, s_idx in keypoints:
        img = dog_pyramid[oct_idx][s_idx]
        h, w = img.shape

        # Filter points too close to edge
        y_start, y_end = y - window_size//2, y + window_size//2
        x_start, x_end = x - window_size//2, x + window_size//2
        if y_start < 0 or y_end >= h or x_start < 0 or x_end >= w:
            continue
            
        sub_img = img[y_start:y_end, x_start:x_end]
        dy = np.gradient(sub_img, axis=0)
        dx = np.gradient(sub_img, axis=1)
        magnitude = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi % 360
        
        desc = []
        for i in range(0, window_size, 4):
            for j in range(0, window_size, 4):
                sub_mag = magnitude[i:i+4, j:j+4]
                sub_ang = angle[i:i+4, j:j+4]
                hist, _ = np.histogram(sub_ang, bins=8, range=(0, 360), weights=sub_mag)
                desc.extend(hist)
        
        desc = np.array(desc)
        norm = np.linalg.norm(desc)
        if norm > 0: desc /= norm
        desc[desc > 0.2] = 0.2
        norm = np.linalg.norm(desc)
        if norm > 0: desc /= norm
            
        descriptors.append(desc)
        valid_keypoints.append([x * (2**oct_idx), y * (2**oct_idx)])
        
    return np.array(valid_keypoints), np.array(descriptors)

def smart_descriptor_matcher(des1, kps1, des2, kps2, img_w, img_h, threshold=0.75):
    """
    Matches descriptors only if the physical distance between the points is small
    (less than 10% of the image size), in accordance with the exercise assumptions.
    """
    matches = []
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return matches

    max_dx = img_w * 0.1
    max_dy = img_h * 0.1

    for i, (d1, pt1) in enumerate(zip(des1, kps1)):
        x1, y1 = pt1
     
        potential_indices = []

        dx_arr = np.abs(kps2[:, 0] - x1)
        dy_arr = np.abs(kps2[:, 1] - y1)
        
        spatial_mask = (dx_arr < max_dx) & (dy_arr < max_dy)
        
        if not np.any(spatial_mask):
            continue
            
        subset_des2 = des2[spatial_mask]
        subset_indices = np.where(spatial_mask)[0]
        
        distances = np.linalg.norm(subset_des2 - d1, axis=1)
        
        if len(distances) < 2:
            continue
            
        sorted_local_indices = np.argsort(distances)
        best_local_idx = sorted_local_indices[0]
        second_local_idx = sorted_local_indices[1]
        
        best_dist = distances[best_local_idx]
        second_dist = distances[second_local_idx]
        
        # Lowe's Ratio Test
        if best_dist < threshold * second_dist:
            original_idx = subset_indices[best_local_idx]
            matches.append((i, original_idx))
            
    return matches

# def my_sift_detect_and_compute(frame):
#     if len(frame.shape) == 3:
#         gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
#     else:
#         gray = frame
        
#     dog = create_dog_space(gray)
#     kps_raw = find_keypoints(dog)
#     pts, descriptors = extract_sift_descriptors(dog, kps_raw)
#     return pts, descriptors

def my_sift_detect_and_compute(frame):
    if len(frame.shape) == 3:
        gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    else:
        gray = frame
        h, w = gray.shape
    small_gray = cv2.resize(gray.astype(np.float32), (w // 2, h // 2))
    
    dog = create_dog_space(small_gray)
    kps_raw = find_keypoints(dog)
    pts, descriptors = extract_sift_descriptors(dog, kps_raw)

    
    pts = pts * 2
    
    return pts, descriptors


def apply_ransac(pts_prev, pts_curr, iterations=100, threshold=5.0): 
    max_inliers = -1
    best_shift = (0, 0)
    num_matches = len(pts_prev)
    if num_matches < 1: return (0, 0)

    for _ in range(iterations):
        idx = np.random.randint(0, num_matches)
        sample_dx = pts_curr[idx, 0] - pts_prev[idx, 0]
        sample_dy = pts_curr[idx, 1] - pts_prev[idx, 1]
 
        expected_curr = pts_prev + np.array([sample_dx, sample_dy])
        distances = np.linalg.norm(pts_curr - expected_curr, axis=1)
        
        inliers_mask = distances < threshold
        num_inliers = np.sum(inliers_mask)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_shift = (np.mean(pts_curr[inliers_mask, 0] - pts_prev[inliers_mask, 0]),
                          np.mean(pts_curr[inliers_mask, 1] - pts_prev[inliers_mask, 1]))
    return best_shift

def calculate_all_shifts(frames):
    shifts = [] 
    h, w, c = frames[0].shape
    
    print("   Computing SIFT for first frame...")
    pts_prev, des_prev = my_sift_detect_and_compute(frames[0])
    
    for i in range(1, len(frames)):
        print(f"   Processing match {i}/{len(frames)-1}...")
        pts_curr, des_curr = my_sift_detect_and_compute(frames[i])
        
        dx, dy = 0, 0 
        matches = smart_descriptor_matcher(des_prev, pts_prev, des_curr, pts_curr, w, h)
        
        if len(matches) > 4:
            matched_pts_prev = np.array([pts_prev[m[0]] for m in matches])
            matched_pts_curr = np.array([pts_curr[m[1]] for m in matches])
            dx, dy = apply_ransac(matched_pts_prev, matched_pts_curr)
        
        shifts.append((dx, dy))
        pts_prev, des_prev = pts_curr, des_curr
        
    return shifts


def build_panorama_at_angle(frames, shifts, x_ratio):
    h, w, c = frames[0].shape
    
    y_positions = [0]
    cum_dy = 0
    for _, dy in shifts:
        cum_dy += dy
        y_positions.append(cum_dy)
    
    min_dy, max_dy = min(y_positions), max(y_positions)
    total_dx = sum([abs(s[0]) for s in shifts])
    
    canvas_w = int(total_dx) + w + 100
    canvas_h = int(h + max_dy - min_dy) + 100
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    
    current_x = 0.0
    global_y_offset = abs(min_dy) + 50
    margin = 5 

    for i, frame in enumerate(frames[:-1]):
        dx = abs(shifts[i][0])
        if abs(dx) < 0.1: 
            dx = 2.0 

        strip_width = int(dx + 2 * margin)
        safe_zone_start = strip_width // 2
        safe_zone_end = w - (strip_width // 2)

        cut_center = int(safe_zone_start + (safe_zone_end - safe_zone_start) * x_ratio)
        
        start_col = int(cut_center - strip_width // 2)
        s_start = max(0, start_col)
        s_end = min(w, start_col + strip_width)
        
        strip = frame[:, s_start:s_end].astype(np.float32)
        s_h, s_w, _ = strip.shape
        mask = np.ones((s_h, s_w, 3), dtype=np.float32)
        
        if s_w > 2 * margin:
            fade_curve = np.linspace(0, 1, margin)
            mask[:, :margin] = fade_curve[None, :, None]
            mask[:, -margin:] = fade_curve[::-1][None, :, None]
        
        target_x = int(current_x)
        paste_y = int(global_y_offset - y_positions[i])
        
        y_start, y_end = max(0, paste_y), min(paste_y + s_h, canvas_h)
        x_start, x_end = max(0, target_x), min(target_x + s_w, canvas_w)
        
        act_h, act_w = y_end - y_start, x_end - x_start
        if act_h > 0 and act_w > 0:
            s_y, s_x = y_start - paste_y, x_start - target_x
            canvas[y_start:y_end, x_start:x_end] += strip[s_y:s_y+act_h, s_x:s_x+act_w] * mask[s_y:s_y+act_h, s_x:s_x+act_w]
            weight_mask[y_start:y_end, x_start:x_end] += mask[s_y:s_y+act_h, s_x:s_x+act_w]
            
        current_x += dx
        
    weight_mask[weight_mask == 0] = 1.0
    final_pano = np.clip(canvas / weight_mask, 0, 255).astype(np.uint8)
    final_pano = final_pano[:, :int(current_x)]
    return final_pano

def build_all_panoramas(frames, shifts, n_out_frames):
    h, w, c = frames[0].shape
    
    y_positions = [0]
    cum_dy = 0
    for _, dy in shifts:
        cum_dy += dy
        y_positions.append(cum_dy)
    
    min_dy, max_dy = min(y_positions), max(y_positions)
    total_dx = sum([abs(s[0]) for s in shifts])
    
    canvas_w = int(total_dx) + w + 100
    canvas_h = int(h + max_dy - min_dy) + 100
    
    canvases = [np.zeros((canvas_h, canvas_w, 3), dtype=np.float32) for _ in range(n_out_frames)]
    weight_masks = [np.zeros((canvas_h, canvas_w, 1), dtype=np.float32) for _ in range(n_out_frames)]
    
    x_ratios = np.linspace(0.0, 1.0, n_out_frames)
    
    current_x = 0.0
    global_y_offset = abs(min_dy) + 50
    margin = 5 

    for i, frame in enumerate(frames[:-1]):
        dx = abs(shifts[i][0])
        if abs(dx) < 0.1: dx = 2.0 

        strip_width = int(dx + 2 * margin)
        safe_zone_start = strip_width // 2
        safe_zone_end = w - (strip_width // 2)
        
        start_col_base = safe_zone_start 
        span = safe_zone_end - safe_zone_start
        
        target_x = int(current_x)
        paste_y = int(global_y_offset - y_positions[i])
        
        for j, ratio in enumerate(x_ratios):
            cut_center = int(start_col_base + span * ratio)
            
            s_start_col = int(cut_center - strip_width // 2)
            s_start = max(0, s_start_col)
            s_end = min(w, s_start_col + strip_width)
            
            strip = frame[:, s_start:s_end].astype(np.float32)
            s_h, s_w, _ = strip.shape
            
            if s_w <= 0: continue

            mask = np.ones((s_h, s_w, 1), dtype=np.float32)
            if s_w > 2 * margin:
                fade = np.linspace(0, 1, margin).astype(np.float32)
                mask[:, :margin, 0] = fade[None, :]
                mask[:, -margin:, 0] = fade[::-1][None, :]
            
            y_start, y_end = max(0, paste_y), min(paste_y + s_h, canvas_h)
            x_start, x_end = max(0, target_x), min(target_x + s_w, canvas_w)
            
            act_h, act_w = y_end - y_start, x_end - x_start
            
            if act_h > 0 and act_w > 0:
                s_y, s_x = y_start - paste_y, x_start - target_x
                
                canvases[j][y_start:y_end, x_start:x_end] += strip[s_y:s_y+act_h, s_x:s_x+act_w] * mask[s_y:s_y+act_h, s_x:s_x+act_w]
                weight_masks[j][y_start:y_end, x_start:x_end] += mask[s_y:s_y+act_h, s_x:s_x+act_w]

        current_x += dx

    final_panos = []
    final_width = int(current_x)
    
    for k in range(n_out_frames):
        w_mask = weight_masks[k]
        w_mask[w_mask == 0] = 1.0
        pano = np.clip(canvases[k] / w_mask, 0, 255).astype(np.uint8)
        pano = pano[:, :final_width]
        final_panos.append(pano)
        
    return final_panos

def crop_for_alignment(pano, ratio, frame_width):
    """
    Performs alignment cropping according to the instructions:
    "Crop left columns from the left panorama. crop right columns from the right panorama".
    """
    h, w, c = pano.shape

    if w < 2 * frame_width:
        return pano
        
    max_crop = int(frame_width * 0.9)
    crop_left = int(max_crop * (1.0 - ratio))
    crop_right = int(max_crop * ratio)
   
    new_start = crop_left
    new_end = w - crop_right
    
    if new_end > new_start:
        return pano[:, new_start:new_end]
    else:
        return pano


def load_frames_from_folder(input_dir_path):
    frames = []
    if not os.path.exists(input_dir_path): return []
    files = sorted([f for f in os.listdir(input_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    for filename in files:
        path = os.path.join(input_dir_path, filename)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

# def generate_panorama(input_frames_path, n_out_frames=10):
#     """
#     Main entry point for ex4:
#     param input_frames_path : path to a dir with input video frames.
#     We will test your code with a dir that has K frames, each in the format
#     "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
#     param n_out_frames: number of generated panorama frames
#     return: A list of generated panorma frames (of size n_out_frames),
#     each list item should be a PIL image of a generated panorama.
#     """

#     print(f"Loading frames from {input_frames_path}...")
#     frames = load_frames_from_folder(input_frames_path)
#     if not frames:
#         print("Error: No images found.")
#         return []

#     print(f"Loaded {len(frames)} frames. Calculating shifts...")
#     shifts = calculate_all_shifts(frames)

#     pil_results = []
#     # Requirement: Exactly 10 panoramas
#     # 0.0 = Leftmost strip
#     # 1.0 = Rightmost strip
#     x_ratios = np.linspace(0.0, 1.0, n_out_frames)
#     frame_width = frames[0].shape[1]

#     print(f"Generating {n_out_frames} panoramas with alignment...")
#     for i, ratio in enumerate(x_ratios):
#         print(f"  Building panorama {i+1}/{n_out_frames} (Ratio: {ratio:.2f})")
#         pano_numpy = build_panorama_at_angle(frames, shifts, x_ratio=ratio)
#         pano_aligned = crop_for_alignment(pano_numpy, ratio, frame_width)
        
#         pil_img = Image.fromarray(pano_aligned)
#         pil_results.append(pil_img)
        
#     return pil_results

def generate_panorama(input_frames_path, n_out_frames=10):
    print(f"Loading frames from {input_frames_path}...")
    frames = load_frames_from_folder(input_frames_path)
    if not frames:
        print("Error: No images found.")
        return []

    print(f"Loaded {len(frames)} frames. Calculating shifts...")
    shifts = calculate_all_shifts(frames)

    print(f"Generating {n_out_frames} panoramas simultaneously...")
    raw_panoramas = build_all_panoramas(frames, shifts, n_out_frames)
    
    pil_results = []
    x_ratios = np.linspace(0.0, 1.0, n_out_frames)
    frame_width = frames[0].shape[1]

    print("Aligning and converting to PIL...")
    for i, pano_numpy in enumerate(raw_panoramas):
        ratio = x_ratios[i]
        pano_aligned = crop_for_alignment(pano_numpy, ratio, frame_width)
        pil_img = Image.fromarray(pano_aligned)
        pil_results.append(pil_img)
        
    return pil_results

def extract_frames_from_video(video_path, output_folder, step=1):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0
    print(f"Extracting frames from {os.path.basename(video_path)}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_idx += 1
    cap.release()
    print(f"Finished. Saved {saved_count} frames.")

def save_panoramas_to_video(pil_images, output_path, fps=5):
    """
    Saves a list of PIL images as an MP4 video.
    """
    if not pil_images:
        print("No images to save to video.")
        return

    # Get dimensions from the first image
    width, height = pil_images[0].size
    
    # Initialize VideoWriter
    # 'mp4v' is a common codec for .mp4. If it fails, try 'avc1' or 'XVID' (with .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Saving video to {output_path}...")
    
    for img in pil_images:
        # Convert PIL image (RGB) to OpenCV format (BGR)
        open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Ensure the image size matches the video writer size
        if (open_cv_image.shape[1], open_cv_image.shape[0]) != (width, height):
            open_cv_image = cv2.resize(open_cv_image, (width, height))
            
        video_writer.write(open_cv_image)

    video_writer.release()
    print(f"Video saved successfully: {output_path}")


def main():
    input_video_path = r"C:\Users\ASUS\Documents\University\image_processing\project\viewpoint_input.mp4"      
    frames_dir = "frames"              
    output_video_name = "evening___.mp4"   
    
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        extract_frames_from_video(input_video_path, frames_dir, step=2) 
    t0 = time.perf_counter()
    panorama_images = generate_panorama(frames_dir, n_out_frames=10) 
    t1 = time.perf_counter()
    print(f"total runtime: {t1-t0:.3f} seconds")
    if panorama_images:
        save_panoramas_to_video(panorama_images, output_video_name, fps=4) 
     
        # debug_folder = "debug_panoramas"
        # if not os.path.exists(debug_folder): os.makedirs(debug_folder)
        # for i, img in enumerate(panorama_images):
        #     img.save(os.path.join(debug_folder, f"pano_{i}.jpg"))
    pass
 

if __name__ == "__main__":
    main()
