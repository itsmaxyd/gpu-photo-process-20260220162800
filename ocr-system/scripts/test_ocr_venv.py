"""
Minimal end-to-end test for OCR pipeline in a Python venv.

Run from ocr-system/: python scripts/test_ocr_venv.py [image_path]
If no image_path is given, a small test image is created in watch_folder.
Asserts that results.csv is written with an "Extracted Text" column and at least one row.
"""

import sys
import os

# Run from ocr-system so imports and paths work
_ocr_system_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ocr_system_dir not in sys.path:
    sys.path.insert(0, _ocr_system_dir)
os.chdir(_ocr_system_dir)

import pandas as pd

from ocr_system import OCRSystem


def main():
    watch_folder = os.environ.get("WATCH_FOLDER", "./watch_folder")
    results_file = os.environ.get("RESULTS_FILE", "./results.csv")

    os.makedirs(watch_folder, exist_ok=True)
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    # Remove existing results so we test a clean write
    if os.path.exists(results_file):
        os.remove(results_file)

    system = OCRSystem(watch_folder=watch_folder, results_file=results_file)
    system.initialize()

    if not system.ocr_initialized:
        print("ERROR: OCR failed to initialize")
        sys.exit(1)

    if len(sys.argv) >= 2:
        image_path = os.path.abspath(sys.argv[1])
        if not os.path.isfile(image_path):
            print(f"ERROR: Not a file: {image_path}")
            sys.exit(1)
        filename = os.path.basename(image_path)
    else:
        # Create a minimal test image (small white rectangle)
        import cv2
        import numpy as np
        filename = "test_ocr_venv.png"
        image_path = os.path.join(watch_folder, filename)
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.imwrite(image_path, img)

    original, processed, extracted_text, stats = system.process_image(image_path)
    if original is None:
        extracted_text = extracted_text or "Preprocessing/OCR failed"
    system._save_result(filename, extracted_text, stats)

    if not os.path.exists(results_file):
        print("ERROR: results.csv was not created")
        sys.exit(1)

    df = pd.read_csv(results_file)
    if "Extracted Text" not in df.columns:
        print("ERROR: results.csv missing 'Extracted Text' column; columns:", list(df.columns))
        sys.exit(1)
    if len(df) < 1:
        print("ERROR: results.csv has no rows")
        sys.exit(1)

    first_text = df["Extracted Text"].iloc[0]
    s = str(first_text) if pd.notna(first_text) else ""
    preview = repr(s[:80]) + ("..." if len(s) > 80 else "")
    print("OK: results.csv has 'Extracted Text' column and at least one row")
    print(f"    Extracted text (first row): {preview}")
    sys.exit(0)


if __name__ == "__main__":
    main()
