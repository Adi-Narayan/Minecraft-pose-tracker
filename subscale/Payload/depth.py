import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Optimized constants
WINDOW_SIZE = 7
MAX_DISPARITY = 64
MIN_DISPARITY = 0
MAX_IMAGE_SIZE = 512
NUM_WORKERS = 6

def resize_images(left, right, max_size=MAX_IMAGE_SIZE):
    """
    Aggressive resize for speed optimization
    """
    scale = max_size / max(left.shape)
    new_width = int(left.shape[1] * scale)
    new_height = int(left.shape[0] * scale)
    
    left = cv2.resize(left, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return left, right

def preprocess_images(left, right):
    """
    Fast preprocessing pipeline
    """
    left = cv2.equalizeHist(left)
    right = cv2.equalizeHist(right)
    
    left = cv2.GaussianBlur(left, (3, 3), 0)
    right = cv2.GaussianBlur(right, (3, 3), 0)
    
    return left, right

def compute_block_difference(left_block, right_block):
    """
    Fast block matching using SAD
    """
    return np.sum(np.abs(left_block - right_block))

def process_chunk(args):
    """
    Process a single chunk of the image
    """
    y_start, y_end, height, width, left, right, half_window = args
    
    # Ensure we're processing the exact number of rows we're supposed to
    chunk_height = y_end - y_start
    disparity = np.zeros((chunk_height, width), dtype=np.float32)
    
    for i, y in enumerate(range(y_start, y_end)):
        if y < half_window or y >= height - half_window:
            continue
            
        for x in range(half_window, width - half_window):
            left_block = left[y-half_window:y+half_window+1, 
                            x-half_window:x+half_window+1]
            
            min_diff = float('inf')
            best_d = 0
            
            x_start = max(half_window, x - MAX_DISPARITY)
            x_end = x + 1
            
            for d in range(x_start, x_end):
                if d + half_window < width:
                    right_block = right[y-half_window:y+half_window+1, 
                                     d-half_window:d+half_window+1]
                    
                    diff = compute_block_difference(left_block, right_block)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_d = x - d
            
            disparity[i, x] = best_d
    
    return y_start, y_end, disparity

def compute_disparity_map(left, right):
    """
    Compute disparity map with fixed chunk sizes
    """
    height, width = left.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    
    half_window = WINDOW_SIZE // 2
    
    # Calculate chunk size
    valid_height = height - 2 * half_window
    chunk_size = max(valid_height // NUM_WORKERS, WINDOW_SIZE)
    
    # Create chunks with proper sizes
    chunks = []
    for y_start in range(half_window, height - half_window, chunk_size):
        y_end = min(y_start + chunk_size, height - half_window)
        chunk_args = (y_start, y_end, height, width, left, right, half_window)
        chunks.append(chunk_args)
    
    print(f"Processing {len(chunks)} chunks...")
    start_time = time.time()
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = list(executor.map(process_chunk, chunks))
        
        # Collect results
        for idx, (y_start, y_end, chunk_result) in enumerate(futures):
            disparity[y_start:y_end] = chunk_result
            print(f"Chunk {idx + 1}/{len(chunks)} "
                  f"({time.time() - start_time:.2f}s)")
    
    return disparity

def post_process(disparity_map):
    """
    Fast post-processing
    """
    disparity_map = cv2.medianBlur(disparity_map.astype(np.float32), 5)
    
    kernel = np.ones((3,3), np.uint8)
    disparity_map = cv2.morphologyEx(disparity_map, cv2.MORPH_CLOSE, kernel)
    
    disparity_map = cv2.GaussianBlur(disparity_map, (3,3), 0)
    
    disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    return disparity_map.astype(np.uint8)

def main():
    try:
        total_start_time = time.time()
        print("Starting high-speed stereo matching...")
        
        left_path = "left image path"
        right_path = "right image path"
        
        # Read and validate images
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left is None or right is None:
            raise FileNotFoundError("Cannot find input images")
        
        print(f"Original size: {left.shape}")
        
        # Resize
        left, right = resize_images(left, right)
        print(f"Resized to: {left.shape}")
        
        # Preprocess
        print("Preprocessing...")
        left, right = preprocess_images(left, right)
        
        # Compute disparity
        print("Computing disparity map...")
        disparity_map = compute_disparity_map(left, right)
        
        # Post-process
        print("Post-processing...")
        final_disparity = post_process(disparity_map)
        
        # Save results
        print("Saving results...")
        cv2.imwrite("disparity.png", final_disparity)
        cv2.imwrite("disparity_color.png", 
                   cv2.applyColorMap(final_disparity, cv2.COLORMAP_JET))
        
        total_time = time.time() - total_start_time
        print(f"Done! Total processing time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()