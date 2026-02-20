import cv2
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_visualizer")

# Import system modules
try:
    from hybrid_ocr import OCRFactory, HybridOCR
    from gpu_preprocessing import GPUPreprocessor
except ImportError:
    # Handle running from root directory
    sys.path.append(os.path.join(os.getcwd(), 'ocr-system'))
    from ocr_system.hybrid_ocr import OCRFactory, HybridOCR
    from ocr_system.gpu_preprocessing import GPUPreprocessor

def visualize_ocr(image_path: str, output_dir: str = "ocr-system/debug_output"):
    """
    Run OCR pipeline and visualize steps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Processing {image_path}...")
    
    # 1. Initialize
    preprocessor = GPUPreprocessor(use_gpu=False) # Force CPU for stability in debug
    ocr = OCRFactory.create_cpu_only() # Force CPU for stability in debug
    
    # 2. Preprocess
    original, processed, status = preprocessor.preprocess_for_ocr(image_path)
    if original is None:
        logger.error(f"Failed to load image: {status}")
        return
        
    # Save preprocessed
    cv2.imwrite(os.path.join(output_dir, "debug_preprocessed.png"), processed)
    logger.info("Saved debug_preprocessed.png")
    
    # 3. Resize (Mimic current production pipeline)
    # We target a height of 2560px for better detail.
    resized_processed = preprocessor.resize_for_ocr(processed, target_height=2560)
    cv2.imwrite(os.path.join(output_dir, "debug_resized_2560.png"), resized_processed)
    logger.info("Saved debug_resized_2560.png")
    
    # 4. Run OCR
    # We use the resized image for OCR as per current pipeline
    logger.info("Running OCR detection and recognition...")
    text, regions = ocr.process_image(resized_processed, return_regions=True)
    
    # 5. Visualize Regions
    vis_img = resized_processed.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
    for region in regions:
        bbox = region.bbox
        # bbox is usually list of tuples [(x,y), (x,y), (x,y), (x,y)]
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (0, 0, 255), 2)
        
        # Put text
        # cv2.putText(vis_img, region.text, (bbox[0][0], bbox[0][1]-5), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
    cv2.imwrite(os.path.join(output_dir, "debug_visualized.png"), vis_img)
    logger.info("Saved debug_visualized.png")
    
    # 6. Save Text
    with open(os.path.join(output_dir, "debug_text.txt"), "w") as f:
        f.write(text)
    logger.info("Saved debug_text.txt")
    
    logger.info("\n--- Extracted Text Preview ---")
    print(text[:500] + "..." if len(text) > 500 else text)
    logger.info("------------------------------")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to the watch folder image if exists
        img_path = "ocr-system/watch_folder/2026-02-20_18-05.jpg"
        
    if os.path.exists(img_path):
        visualize_ocr(img_path)
    else:
        print(f"Image not found: {img_path}")
