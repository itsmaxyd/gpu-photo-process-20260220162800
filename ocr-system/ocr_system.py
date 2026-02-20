"""
OCR System Management Module

Handles initialization, monitoring, and processing logic for the OCR system.
Separated from the UI to ensure clean architecture and separation of concerns.
"""

import os
import time
import threading
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Import our optimized modules
from gpu_preprocessing import GPUPreprocessor
from hybrid_ocr import HybridOCR, OCRFactory

logger = logging.getLogger(__name__)

class OCRSystem:
    """
    Manages the OCR system components and processing pipeline.
    
    Encapsulates initialization, file monitoring, and image processing.
    Decoupled from Streamlit to allow easier testing and reuse.
    """
    
    def __init__(self, watch_folder: str = "/app/watch_folder", results_file: str = "/app/results.csv", poll_interval: int = 15):
        self.watch_folder = watch_folder
        self.results_file = results_file
        self.poll_interval = poll_interval
        
        self.preprocessor: Optional[GPUPreprocessor] = None
        self.ocr: Optional[HybridOCR] = None
        self.gpu_available = False
        self.ocr_initialized = False
        self.monitoring_started = False
        self._monitoring_thread = None
        
    def initialize(self):
        """Initialize system components."""
        self._init_preprocessor()
        self._init_ocr()
        
    def _init_preprocessor(self):
        """Initialize GPU preprocessor."""
        logger.info("Initializing GPU preprocessor...")
        try:
            self.preprocessor = GPUPreprocessor(use_gpu=True)
            if self.preprocessor.cuda_available:
                logger.info("GPU preprocessing enabled")
                self.gpu_available = True
            else:
                logger.warning("GPU preprocessing not available, using CPU")
                self.gpu_available = False
                
        except Exception as e:
            logger.error(f"Preprocessor initialization failed: {e}")
            self.gpu_available = False
            # Fallback to simple CPU stub if needed, but GPUPreprocessor handles fallback

    def _init_ocr(self):
        """Initialize hybrid OCR system with high-accuracy settings."""
        logger.info("Initializing hybrid OCR system with high-accuracy configuration...")
        try:
            # Use high-accuracy configuration for best OCR results
            self.ocr = OCRFactory.create_high_accuracy(lang='en')
            logger.info("Hybrid OCR initialized successfully with high-accuracy config")
            self.ocr_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize hybrid OCR: {e}")
            # Fallback to fast configuration
            logger.warning("Falling back to fast OCR configuration")
            try:
                self.ocr = OCRFactory.create_fast(lang='en')
                self.ocr_initialized = True
            except Exception as fallback_error:
                logger.critical(f"Critical failure: Fast OCR fallback also failed: {fallback_error}")
                self.ocr_initialized = False
        
    def process_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, float]]:
        """
        Process a single image through the full pipeline.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (original_image, processed_image, extracted_text, timing_stats)
        """
        stats = {
            'preprocess_time': 0.0,
            'ocr_time': 0.0,
            'total_time': 0.0
        }
        
        if not self.preprocessor or not self.ocr:
            return None, None, "System not initialized", stats
        
        total_start = time.time()
        
        # Step 1: Preprocessing with enhanced pipeline
        preprocess_start = time.time()
        
        # Use enhanced preprocessing (v2) for better accuracy
        if hasattr(self.preprocessor, 'preprocess_for_ocr_v2'):
            original_img, processed_img, status = self.preprocessor.preprocess_for_ocr_v2(image_path)
        else:
            original_img, processed_img, status = self.preprocessor.preprocess_for_ocr(image_path)
        
        stats['preprocess_time'] = time.time() - preprocess_start
        
        if original_img is None:
            return None, None, f"Preprocessing failed: {status}", stats
        
        # Step 1.5: Multi-scale OCR for better accuracy on varying text sizes
        # Instead of just resizing to one height, we process at multiple scales
        # This significantly improves detection of small or large text
        
        # Target multiple heights for better text detection
        target_heights = [1920, 2560]  # Process at 1920px AND 2560px height
        
        all_text_regions = []
        
        for target_height in target_heights:
            # Resize for this scale
            scaled_img = self.preprocessor.resize_for_ocr(processed_img, target_height=target_height)
            
            # Run OCR on this scale
            try:
                _, text_regions = self.ocr.process_image(scaled_img, return_regions=True)
                all_text_regions.extend(text_regions)
            except Exception as e:
                logger.warning(f"OCR at height {target_height} failed: {e}")
                continue
        
        # Step 2: Merge overlapping regions from multi-scale processing
        merged_regions = self._merge_overlapping_regions(all_text_regions)
        
        # Extract text from merged regions
        combined_text = '\n'.join([tr.text for tr in merged_regions if tr.text.strip()])
        
        # Step 3: Post-processing - add common corrections
        combined_text = self._post_process_text(combined_text)
        
        stats['ocr_time'] = time.time() - total_start
        stats['total_time'] = time.time() - total_start
        stats['regions_found'] = len(merged_regions)
        stats['scales_processed'] = len(target_heights)
        
        return original_img, processed_img, combined_text, stats

    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.monitoring_started:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
            self.monitoring_started = True
            logger.info("Monitoring thread started")

    def _monitoring_loop(self):
        """Internal background monitoring loop."""
        logger.info("Starting monitoring loop...")
        processed_files = set()
        
        # Load existing
        self._load_processed_history(processed_files)
        
        while True:
            try:
                if not os.path.exists(self.watch_folder):
                    # logger.warning(f"Watch folder does not exist: {self.watch_folder}") # Reduce noise
                    time.sleep(self.poll_interval)
                    continue
                
                # Check for files
                try:
                    files = sorted([
                        f for f in os.listdir(self.watch_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                    ])
                except OSError:
                    files = []
                    
                for filename in files:
                    if filename not in processed_files:
                        self._process_and_save(filename, processed_files)
                        
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.poll_interval)

    def _load_processed_history(self, processed_files: set):
        """Load history of processed files."""
        if os.path.exists(self.results_file):
            try:
                # Check file size first
                if os.path.getsize(self.results_file) == 0:
                    logger.warning("Results file is empty. Starting fresh.")
                    return

                df = pd.read_csv(self.results_file)
                if 'Filename' in df.columns:
                    # Update the set via update to avoid reference change issues if any
                    processed_files.update(df['Filename'].tolist())
                    logger.info(f"Loaded {len(processed_files)} previously processed files")
                else:
                    logger.warning("Results file format incorrect (missing Filename column). Starting fresh.")
            except pd.errors.EmptyDataError:
                logger.warning("Results file found but empty (pd error). Starting fresh.")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")

    def _process_and_save(self, filename: str, processed_files: set):
        """Process a file and save results."""
        file_path = os.path.join(self.watch_folder, filename)
        logger.info(f"Processing new file: {filename}")
        
        try:
            original_img, processed_img, text, stats = self.process_image(file_path)
            
            if original_img is not None:
                self._save_result(filename, text, stats)
                processed_files.add(filename)
                logger.info(f"Finished processing: {filename}")
            else:
                logger.error(f"Failed to process {filename}")
        except Exception as e:
            logger.error(f"Exception processing file {filename}: {e}")

    def _save_result(self, filename: str, text: str, stats: Dict[str, float]):
        """Save processing result to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = pd.DataFrame([{
                "Timestamp": timestamp,
                "Filename": filename,
                "Extracted Text": text,
                "Preprocess Time": f"{stats['preprocess_time']:.3f}s",
                "OCR Time": f"{stats['ocr_time']:.3f}s",
                "Total Time": f"{stats['total_time']:.3f}s"
            }])
            
            # Determine if header is needed
            header = True
            if os.path.exists(self.results_file) and os.path.getsize(self.results_file) > 0:
                 header = False

            new_entry.to_csv(self.results_file, mode='a', header=header, index=False)
        except Exception as e:
            logger.error(f"Failed to save result for {filename}: {e}")

    def _merge_overlapping_regions(self, regions: List) -> List:
        """
        Merge overlapping text regions from multi-scale OCR processing.
        
        Uses IoU (Intersection over Union) to identify and merge duplicate detections.
        """
        if not regions:
            return []
        
        merged = []
        used_indices = set()
        
        for i, region in enumerate(regions):
            if i in used_indices:
                continue
            
            current_region = region
            
            # Find all overlapping regions
            for j, other_region in enumerate(regions[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                if self._regions_overlap(current_region, other_region):
                    # Merge: keep the one with higher confidence
                    if other_region.confidence > current_region.confidence:
                        current_region = other_region
                    used_indices.add(j)
            
            merged.append(current_region)
            used_indices.add(i)
        
        # Sort by position (top to bottom, left to right)
        merged.sort(key=lambda r: (r.bbox[0][1] if r.bbox else 0, r.bbox[0][0] if r.bbox else 0))
        
        return merged
    
    def _regions_overlap(self, region1, region2) -> bool:
        """Check if two text regions overlap."""
        try:
            # Get bounding boxes
            bbox1 = region1.bbox
            bbox2 = region2.bbox
            
            if not bbox1 or not bbox2:
                return False
            
            # Calculate intersection
            x1_min = min(p[0] for p in bbox1)
            x1_max = max(p[0] for p in bbox1)
            y1_min = min(p[1] for p in bbox1)
            y1_max = max(p[1] for p in bbox1)
            
            x2_min = min(p[0] for p in bbox2)
            x2_max = max(p[0] for p in bbox2)
            y2_min = min(p[1] for p in bbox2)
            y2_max = max(p[1] for p in bbox2)
            
            # Check intersection
            overlap = not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
            
            return overlap
        except:
            return False
    
    def _post_process_text(self, text: str) -> str:
        """
        Apply post-processing corrections to OCR output.
        
        Handles common OCR errors and formatting issues.
        """
        if not text:
            return text
        
        # Fix common OCR character substitutions
        replacements = {
            '|': 'I',
            '—': '-',
            '–': '-',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '...': '...',
        }
        
        # Apply replacements carefully
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Fix multiple newlines
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
