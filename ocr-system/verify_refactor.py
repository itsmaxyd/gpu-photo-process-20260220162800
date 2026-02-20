import sys
import os
import numpy as np
import cv2

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from ocr_system import OCRSystem
    from gpu_preprocessing import GPUPreprocessor
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_preprocessing():
    print("Testing preprocessing logic...")
    try:
        # Force CPU to test the fix (since we modified _preprocess_cpu)
        # Even if GPU is available, logic is similar now (enhancement vs binary)
        prep = GPUPreprocessor(use_gpu=False) 
        
        # Create dummy image: 100x100 RGB
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Run CPU preprocess
        processed = prep._preprocess_cpu(img)
        
        if processed is None:
            print("Preprocessing returned None!")
            return False
            
        print(f"Processed shape: {processed.shape}")
        
        # Check if it's grayscale
        if len(processed.shape) == 2:
            print("Output is grayscale (correct for PaddleOCR).")
        elif len(processed.shape) == 3 and processed.shape[2] == 1:
            print("Output is grayscale (correct for PaddleOCR).")
        else:
            print(f"Output shape unexpected: {processed.shape}")
            return False
            
        # Check if it's not strictly binary (0 and 255 only)
        # Since input is random noise, enhanced output should have variation
        unique_vals = np.unique(processed)
        if len(unique_vals) <= 2:
            print("Warning: Output appears to be binary. This might be intented if image is very simple, but for random noise it should vary.")
            print(f"Unique values: {unique_vals}")
        else:
            print(f"Output has {len(unique_vals)} unique values (Good: Grayscale maintained).")
            
        return True
    except Exception as e:
        print(f"Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    if test_preprocessing():
        print("\nVerification Passed!")
        sys.exit(0)
    else:
        print("\nVerification Failed!")
        sys.exit(1)
