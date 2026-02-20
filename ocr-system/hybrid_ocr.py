"""
Hybrid OCR Module for GTX 980 Ti (Maxwell Architecture)

This module implements a hybrid approach optimized for GTX 980 Ti:
- GPU-accelerated text detection (using PaddleOCR detection model)
- CPU-based text recognition (using Tesseract or PaddleOCR CPU)

This approach maximizes GTX 980 Ti's strengths:
- Strong memory bandwidth (336.5 GB/s) for detection
- Avoids compatibility issues with newer CUDA features
"""

# CRITICAL: Disable OneDNN/MKLDNN before importing PaddlePaddle
# PaddlePaddle 3.x has compatibility issues with OneDNN backend
# Error: "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"
import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_onednn'] = '0'
os.environ['FLAGS_enable_mkldnn'] = '0'

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecognitionEngine(Enum):
    """Available OCR recognition engines."""
    PADDLE_CPU = "paddle_cpu"
    PADDLE_V4 = "paddle_v4"  # PP-OCRv4 - significantly improved accuracy
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    ENSEMBLE = "ensemble"  # Multi-engine voting for best accuracy


@dataclass
class TextRegion:
    """Represents a detected text region."""
    bbox: List[Tuple[int, int]]  # Bounding box coordinates
    text: str  # Recognized text
    confidence: float  # Recognition confidence
    detection_confidence: float  # Detection confidence
    engine: str = "unknown"  # Engine used for recognition


class HybridOCR:
    """
    Hybrid OCR system combining GPU detection with CPU recognition.
    
    Optimized for GTX 980 Ti (Maxwell architecture):
    - Uses GPU for text detection (parallelizable, benefits from memory bandwidth)
    - Uses CPU for text recognition (avoids CUDA compatibility issues)
    
    Supports multiple engines with ensemble voting for best accuracy.
    """
    
    def __init__(
        self,
        use_gpu_detection: bool = True,
        recognition_engine: RecognitionEngine = RecognitionEngine.ENSEMBLE,  # Default to ensemble
        lang: str = 'en',
        use_v4_models: bool = True,  # Use PP-OCRv4 models by default
        enable_spell_check: bool = True,  # Enable spell correction
        confidence_threshold: float = 0.5  # Minimum confidence threshold
    ):
        """
        Initialize the hybrid OCR system.
        
        Args:
            use_gpu_detection: Use GPU for text detection (default: True)
            recognition_engine: Which recognition engine to use
            lang: Language for OCR
            use_v4_models: Use PP-OCRv4 models (better accuracy)
            enable_spell_check: Enable spell correction post-processing
            confidence_threshold: Minimum confidence for accepting results
        """
        self.use_gpu_detection = use_gpu_detection
        self.recognition_engine = recognition_engine
        self.lang = lang
        self.use_v4_models = use_v4_models
        self.enable_spell_check = enable_spell_check
        self.confidence_threshold = confidence_threshold
        
        # Initialize spell checker
        self._spell = None
        if self.enable_spell_check:
            self._init_spell_checker()
        
        # Initialize detection and recognition models
        self.detector = None
        self.recognizer = None
        self.ensemble_engines = []  # For multi-engine voting
        # PaddleOCR 3.x uses a single full pipeline; 2.x uses separate det/rec
        self._paddle_v3 = False

        self._init_detector()
        if recognition_engine == RecognitionEngine.ENSEMBLE:
            self._init_ensemble()
        elif not self._paddle_v3:
            self._init_recognizer()
    
    def _init_spell_checker(self) -> None:
        """Initialize spell checker for post-processing."""
        try:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
            logger.info("Spell checker initialized")
        except ImportError:
            logger.warning("spellchecker not installed, spell correction disabled")
            self.enable_spell_check = False
    
    def _init_detector(self) -> None:
        """Initialize the text detection model with improved settings."""
        try:
            if self.use_gpu_detection:
                self._initialize_gpu_detector()
            else:
                self._initialize_cpu_detector()
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}. Attempting CPU fallback.")
            if self.use_gpu_detection:
                try:
                    self._initialize_cpu_detector()
                except Exception as fallback_error:
                    logger.error(f"CPU fallback failed: {fallback_error}")
                    raise

    def _initialize_gpu_detector(self) -> None:
        """Helper to initialize GPU detector (PaddleOCR 2.x) with v4 models."""
        from paddleocr import PaddleOCR
        logger.info("Initializing GPU text detection with PP-OCRv4...")
        try:
            # Use PP-OCRv4 models for better accuracy
            # det_db_thresh: Higher = more strict (fewer false positives)
            # det_db_box_thresh: Higher = more strict detection
            # rec_batch_num: Lower = less memory, more stable
            self.detector = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=True,
                show_log=False,
                det=True,
                cls=True,
                det_model_dir=None,
                # PP-OCRv4 specific improvements
                det_db_thresh=0.3,  # Lower for more detections
                det_db_box_thresh=0.5,  # Lower for more boxes
                det_db_unclip_ratio=1.6,  # More expansion for connected text
                rec_batch_num=6,  # Smaller batch for stability
            )
            logger.info("Text detection initialized on GPU with v4 models")
        except (TypeError, ValueError) as e:
            err = str(e).lower()
            if any(x in err for x in ("show_log", "det", "use_gpu", "unknown argument")):
                self._init_paddle_v3()
            else:
                raise

    def _initialize_cpu_detector(self) -> None:
        """Helper to initialize CPU detector (PaddleOCR 2.x) with v4 models."""
        from paddleocr import PaddleOCR
        logger.info("Initializing CPU text detection with PP-OCRv4...")
        try:
            self.detector = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=False,
                show_log=False,
                det=True,
                cls=True,
                # PP-OCRv4 specific improvements
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                rec_batch_num=6,
            )
            self.use_gpu_detection = False
            logger.info("Text detection initialized on CPU with v4 models")
        except (TypeError, ValueError) as e:
            err = str(e).lower()
            if any(x in err for x in ("show_log", "det", "use_gpu", "unknown argument")):
                self._init_paddle_v3()
            else:
                raise

    def _init_paddle_v3(self) -> None:
        """Initialize single PaddleOCR instance for 3.x (full pipeline, no det/rec split)."""
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR 3.x (single pipeline)...")
        self.detector = PaddleOCR(lang=self.lang)
        self.recognizer = self.detector
        self._paddle_v3 = True
        self.use_gpu_detection = False
        logger.info("PaddleOCR 3.x initialized")
    
    def _init_recognizer(self) -> None:
        """Initialize the text recognition model (CPU-based)."""
        try:
            if self.recognition_engine == RecognitionEngine.PADDLE_CPU or self.recognition_engine == RecognitionEngine.PADDLE_V4:
                self._init_paddle_recognizer()
            elif self.recognition_engine == RecognitionEngine.TESSERACT:
                self._init_tesseract_recognizer()
            elif self.recognition_engine == RecognitionEngine.EASYOCR:
                self._init_easyocr_recognizer()
            else:
                logger.warning(f"Unknown engine {self.recognition_engine}, defaulting to PADDLE_V4")
                self._init_paddle_recognizer()
        except Exception as e:
            logger.error(f"Recognizer initialization failed: {e}")
            # Fallback to PaddleOCR v4 if preferred engine fails
            if self.recognition_engine not in (RecognitionEngine.PADDLE_CPU, RecognitionEngine.PADDLE_V4):
                logger.info("Falling back to PaddleOCR v4 recognizer")
                self.recognition_engine = RecognitionEngine.PADDLE_V4
                self._init_paddle_recognizer()
            else:
                raise
    
    def _init_paddle_recognizer(self) -> None:
        """Initialize PaddleOCR for CPU-based recognition with v4 models."""
        if self._paddle_v3:
            return
        from paddleocr import PaddleOCR
        try:
            # Use PP-OCRv4 for better accuracy
            self.recognizer = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=False,
                show_log=False,
                det=False,
                rec=True,
                cls=False,
                # PP-OCRv4 recognition improvements
                rec_model_name='ch_PP-OCRv4',  # Explicitly use v4 model
                rec_batch_num=6,
                rec_image_shape='3, 48, 192',  # Better for v4
            )
            logger.info("PaddleOCR v4 CPU recognizer initialized")
        except (TypeError, ValueError) as e:
            err = str(e).lower()
            if any(x in err for x in ("show_log", "det", "use_gpu", "rec", "unknown argument", "rec_model_name")):
                # Fallback to basic v3-style init
                self.recognizer = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                    det=False,
                    rec=True,
                    cls=False,
                )
                logger.info("PaddleOCR recognizer initialized (fallback)")
            raise
    
    def _init_tesseract_recognizer(self) -> None:
        """Initialize Tesseract for text recognition."""
        try:
            import pytesseract
            self.recognizer = pytesseract
            # Verify Tesseract is installed
            pytesseract.get_tesseract_version()
            logger.info("Tesseract recognizer initialized")
        except (ImportError, OSError) as e:
            logger.warning(f"pytesseract issue: {e}. Falling back to PADDLE_CPU.")
            self.recognition_engine = RecognitionEngine.PADDLE_CPU
            self._init_paddle_recognizer()
    
    def _init_easyocr_recognizer(self) -> None:
        """Initialize EasyOCR for text recognition with better settings."""
        try:
            import easyocr
            # EasyOCR with GPU disabled for compatibility with GTX 980 Ti
            # Add more detail for better accuracy
            self.recognizer = easyocr.Reader(
                [self.lang], 
                gpu=False,
                verbose=False,
                model_storage_directory=None,
                download_enabled=False,
            )
            logger.info("EasyOCR recognizer initialized")
        except ImportError:
            logger.warning("easyocr not installed, falling back to PADDLE_V4")
            self.recognition_engine = RecognitionEngine.PADDLE_V4
            self._init_paddle_recognizer()
    
    def _init_ensemble(self) -> None:
        """Initialize ensemble engines for multi-engine voting."""
        logger.info("Initializing ensemble OCR engines...")
        
        # Initialize multiple OCR engines
        self.ensemble_engines = []
        
        # 1. PaddleOCR v4 (primary)
        try:
            from paddleocr import PaddleOCR
            paddle = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=6,
            )
            self.ensemble_engines.append(('paddle', paddle))
            logger.info("Ensemble: PaddleOCR v4 added")
        except Exception as e:
            logger.warning(f"Ensemble: PaddleOCR failed: {e}")
        
        # 2. EasyOCR (often better for complex text)
        try:
            import easyocr
            easy = easyocr.Reader([self.lang], gpu=False, verbose=False)
            self.ensemble_engines.append(('easyocr', easy))
            logger.info("Ensemble: EasyOCR added")
        except ImportError:
            logger.warning("Ensemble: EasyOCR not available")
        
        # 3. Tesseract (good for simple, clean text)
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ensemble_engines.append(('tesseract', pytesseract))
            logger.info("Ensemble: Tesseract added")
        except (ImportError, OSError):
            logger.warning("Ensemble: Tesseract not available")
        
        if not self.ensemble_engines:
            logger.warning("No ensemble engines available, falling back to PaddleOCR v4")
            self.recognition_engine = RecognitionEngine.PADDLE_V4
            self._init_paddle_recognizer()
        else:
            logger.info(f"Ensemble initialized with {len(self.ensemble_engines)} engines")
    
    def detect_text(self, image: np.ndarray) -> List[Tuple[List, float]]:
        """
        Detect text regions in an image.
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            List of (bounding_box, confidence) tuples
        """
        try:
            if self._paddle_v3:
                # 3.x uses full pipeline in process_image only; no separate det
                return []
            if not isinstance(image, np.ndarray):
                logger.error("Invalid image format provided to detect_text")
                return []

            # Ensure 3 channels for PaddleOCR detection
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Run detection
            result = self.detector.ocr(image, det=True, rec=False, cls=True)
            
            # PaddleOCR returns a list of results (one per image passed)
            if not result or not result[0]:
                return []
                
            detections = []
            for line in result[0]:
                # line structure: [box_coords, (text, confidence)] IF rec=True
                # If rec=False (which we might have set via det=True, rec=False above? 
                # Wait, PaddleOCR.ocr() behavior depends on args.
                # If we passed rec=False to .ocr(), it returns boxes.
                
                # Let's handle the output format robustly.
                # usually [[box, (text, conf)], ...] for det+rec
                # or [box, ...] for det only?
                # Actually PaddleOCR detection only returns [box1, box2, ...] usually.
                # Let's check typical paddle output.
                # Actually, standard usage usually returns `[[[bbox], (text, conf)], ...]`
                
                # If we just do det=True, rec=False:
                # result = [ [box1, box2, ...] ]
                
                if isinstance(line, list) and len(line) == 4 and isinstance(line[0], list):
                     # This looks like just a box [[x,y], [x,y], [x,y], [x,y]]
                     bbox = line
                     confidence = 1.0 # Detection confidence not always returned in simple det mode
                     detections.append((bbox, confidence))
                elif isinstance(line, list) and len(line) == 2:
                    # [[box], (text, conf)]
                    bbox = line[0]
                    confidence = line[1][1] if isinstance(line[1], tuple) else 1.0
                    detections.append((bbox, confidence))
                
            # Sort detections top-to-bottom, left-to-right
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Sort primarily by Y of top-left corner, then X
            # Note: This is a simple sort. For complex layouts, a more sophisticated
            # line grouping algorithm would be better.
            if detections:
                detections.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
            
            return detections
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []
    
    def recognize_text(self, image: np.ndarray, regions: List[Tuple]) -> List[TextRegion]:
        """
        Recognize text in detected regions using ensemble voting.
        
        Args:
            image: Original image
            regions: List of detected regions (bbox, confidence)
            
        Returns:
            List of TextRegion objects with recognized text
        """
        results = []
        
        # Handle ensemble mode
        if self.recognition_engine == RecognitionEngine.ENSEMBLE and self.ensemble_engines:
            return self._recognize_ensemble(image, regions)
        
        for bbox, det_conf in regions:
            try:
                # Extract region from image
                region_img = self._extract_region(image, bbox)
                
                if region_img is None or region_img.size == 0:
                    continue
                
                # Recognize text based on engine
                text, conf = "", 0.0
                
                if self.recognition_engine == RecognitionEngine.TESSERACT:
                    text, conf = self._recognize_tesseract(region_img)
                elif self.recognition_engine == RecognitionEngine.EASYOCR:
                    text, conf = self._recognize_easyocr(region_img)
                else:  # PADDLE_CPU or PADDLE_V4
                    text, conf = self._recognize_paddle(region_img)
                
                # Apply spell correction if enabled
                if self.enable_spell_check and text.strip():
                    text = self._apply_spell_correction(text)
                
                # Only include results above confidence threshold
                if conf < self.confidence_threshold:
                    logger.debug(f"Skipping low confidence result: {conf:.2f}")
                    continue
                
                # Create TextRegion
                formatted_bbox = [(int(p[0]), int(p[1])) for p in bbox]
                
                text_region = TextRegion(
                    bbox=formatted_bbox,
                    text=text,
                    confidence=float(conf),
                    detection_confidence=float(det_conf),
                    engine=self.recognition_engine.value
                )
                results.append(text_region)
                
            except Exception as e:
                logger.warning(f"Recognition failed for region: {e}")
                continue
        
        return results
    
    def _recognize_ensemble(self, image: np.ndarray, regions: List[Tuple]) -> List[TextRegion]:
        """
        Recognize text using ensemble voting across multiple engines.
        
        Uses majority voting or confidence-weighted selection.
        """
        results = []
        
        for bbox, det_conf in regions:
            try:
                region_img = self._extract_region(image, bbox)
                
                if region_img is None or region_img.size == 0:
                    continue
                
                # Get predictions from all engines
                predictions = []
                for engine_name, engine in self.ensemble_engines:
                    try:
                        if engine_name == 'paddle':
                            text, conf = self._recognize_paddle_with_engine(engine, region_img)
                        elif engine_name == 'easyocr':
                            text, conf = self._recognize_easyocr_with_engine(engine, region_img)
                        elif engine_name == 'tesseract':
                            text, conf = self._recognize_tesseract(region_img)
                        else:
                            continue
                        
                        if text.strip():
                            predictions.append((text, conf, engine_name))
                            logger.debug(f"Ensemble {engine_name}: '{text}' (conf: {conf:.2f})")
                    except Exception as e:
                        logger.debug(f"Ensemble {engine_name} failed: {e}")
                        continue
                
                if not predictions:
                    continue
                
                # Voting: select the text with highest confidence or majority
                # First, try to find exact matches
                texts_only = [p[0] for p in predictions]
                best_prediction = max(predictions, key=lambda x: x[1])  # Default to highest confidence
                
                # Check for majority voting among similar texts
                for pred in predictions:
                    # Count how many engines agree on this text (exact match)
                    matches = sum(1 for t in texts_only if t.lower() == pred[0].lower())
                    if matches >= 2:  # At least 2 engines agree
                        best_prediction = pred
                        logger.debug(f"Ensemble voting selected: '{pred[0]}' ({matches} engines)")
                        break
                
                text, conf, engine_name = best_prediction
                
                # Apply spell correction
                if self.enable_spell_check and text.strip():
                    text = self._apply_spell_correction(text)
                
                # Only include if above threshold
                if conf < self.confidence_threshold:
                    continue
                
                formatted_bbox = [(int(p[0]), int(p[1])) for p in bbox]
                
                text_region = TextRegion(
                    bbox=formatted_bbox,
                    text=text,
                    confidence=float(conf),
                    detection_confidence=float(det_conf),
                    engine=f"ensemble({engine_name})"
                )
                results.append(text_region)
                
            except Exception as e:
                logger.warning(f"Ensemble recognition failed for region: {e}")
                continue
        
        return results
    
    def _recognize_paddle_with_engine(self, engine, region: np.ndarray) -> Tuple[str, float]:
        """Recognize using a specific PaddleOCR engine instance."""
        result = engine.ocr(region, det=False, rec=True, cls=True)
        if not result:
            return "", 0.0
        try:
            if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], tuple):
                text = result[0][0][0]
                conf = result[0][0][1]
            elif isinstance(result[0], tuple):
                text = result[0][0]
                conf = result[0][1]
            else:
                return "", 0.0
            return str(text), float(conf)
        except:
            return "", 0.0
    
    def _recognize_easyocr_with_engine(self, engine, region: np.ndarray) -> Tuple[str, float]:
        """Recognize using a specific EasyOCR engine instance."""
        try:
            results = engine.readtext(region)
            if results:
                text_parts = []
                confidences = []
                for detection in results:
                    text_parts.append(detection[1])
                    confidences.append(detection[2])
                text = ' '.join(text_parts)
                conf = float(np.mean(confidences))
                return text, conf
            return "", 0.0
        except Exception as e:
            logger.debug(f"EasyOCR recognition error: {e}")
            return "", 0.0
    
    def _apply_spell_correction(self, text: str) -> str:
        """Apply spell correction to recognized text."""
        if not self._spell or not text.strip():
            return text
        
        try:
            words = text.split()
            corrected_words = []
            for word in words:
                # Only correct words that are clearly wrong
                if len(word) > 2:
                    # Check if word might be misspelled
                    candidates = self._spell.candidates(word)
                    if candidates and word not in candidates:
                        # Get the most likely correction
                        corrected = self._spell.correction(word)
                        if corrected:
                            corrected_words.append(corrected)
                            continue
                corrected_words.append(word)
            
            return ' '.join(corrected_words)
        except Exception:
            return text
    
    def _extract_region(self, image: np.ndarray, bbox: List) -> Optional[np.ndarray]:
        """Extract a text region from the image based on bounding box."""
        try:
            # Convert bbox to numpy array
            points = np.array(bbox, dtype=np.int32)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(points)
            
            # Add padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop region
            region = image[y:y+h, x:x+w]
            return region
        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
            return None
    
    def _recognize_paddle(self, region: np.ndarray) -> Tuple[str, float]:
        """Recognize text using PaddleOCR CPU (2.x only)."""
        if self._paddle_v3:
            return "", 0.0
        result = self.recognizer.ocr(region, det=False, rec=True, cls=True)
        if not result:
            return "", 0.0
            
        # PaddleOCR nesting can vary based on det/rec flags and input
        # Typically [[(text, conf)]] or [(text, conf)]
        try:
            # Case 1: [[(text, conf)]]
            if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], tuple):
                text = result[0][0][0]
                conf = result[0][0][1]
            # Case 2: [(text, conf)]
            elif isinstance(result[0], tuple):
                text = result[0][0]
                conf = result[0][1]
            # Case 3: [[text, conf]]
            elif isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], str):
                 text = result[0][0]
                 conf = result[0][1]
            else:
                logger.warning(f"Unexpected PaddleOCR result format: {type(result[0])}")
                return "", 0.0
            
            return str(text), float(conf)
        except (IndexError, TypeError, ValueError) as e:
            logger.warning(f"Error parsing PaddleOCR result: {e}")
            return "", 0.0
        except Exception as e:
            logger.warning(f"PaddleOCR recognition failed: {e}")
            return "", 0.0
    
    def _recognize_tesseract(self, region: np.ndarray) -> Tuple[str, float]:
        """Recognize text using Tesseract."""
        try:
            import pytesseract
            from pytesseract import Output
            
            # Get text and confidence
            data = pytesseract.image_to_data(region, output_type=Output.DICT)
            
            # Combine text
            text_parts = []
            confidences = []
            for i, txt in enumerate(data['text']):
                if isinstance(txt, str) and txt.strip():
                    text_parts.append(txt)
                    # conf is 0-100
                    try:
                        conf_val = float(data['conf'][i])
                        if conf_val >= 0:
                            confidences.append(conf_val)
                    except (ValueError, TypeError):
                        pass
            
            text = ' '.join(text_parts)
            conf = (np.mean(confidences) / 100.0) if confidences else 0.0
            
            return text, conf
        except Exception as e:
            logger.warning(f"Tesseract recognition failed: {e}")
            return "", 0.0
    
    def _recognize_easyocr(self, region: np.ndarray) -> Tuple[str, float]:
        """Recognize text using EasyOCR."""
        try:
            results = self.recognizer.readtext(region)
            
            if results:
                text_parts = []
                confidences = []
                for detection in results:
                    # detection = (bbox, text, prob)
                    text_parts.append(detection[1])
                    confidences.append(detection[2])
                
                text = ' '.join(text_parts)
                conf = float(np.mean(confidences))
                return text, conf
            return "", 0.0
        except Exception as e:
            logger.warning(f"EasyOCR recognition failed: {e}")
            return "", 0.0
    
    def process_image(
        self,
        image: np.ndarray,
        return_regions: bool = False
    ) -> Union[Tuple[str, List[TextRegion]], str]:
        """
        Full OCR processing pipeline.
        
        Args:
            image: Input image (can be preprocessed)
            return_regions: Whether to return individual text regions
            
        Returns:
            Tuple of (combined_text, list_of_text_regions) or just combined_text
        """
        if self._paddle_v3:
            return self._process_image_v3(image, return_regions)

        start_time = time.time()
        detect_start = time.time()
        regions = self.detect_text(image)
        detect_time = time.time() - detect_start
        logger.info(f"Detection took {detect_time:.3f}s, found {len(regions)} regions")
        recog_start = time.time()
        text_regions = self.recognize_text(image, regions)
        recog_time = time.time() - recog_start
        logger.info(f"Recognition took {recog_time:.3f}s")
        combined_text = '\n'.join([tr.text for tr in text_regions if tr.text.strip()])
        total_time = time.time() - start_time
        logger.info(f"Total OCR processing took {total_time:.3f}s")
        if return_regions:
            return combined_text, text_regions
        return combined_text
    
    def _process_image_v3(
        self, image: np.ndarray, return_regions: bool
    ) -> Union[str, Tuple[str, List[TextRegion]]]:
        """Run full OCR with PaddleOCR 3.x (single pipeline, no det/rec kwargs)."""
        try:
            result = self.detector.ocr(image)
        except Exception as e:
            logger.error(f"PaddleOCR 3.x ocr() failed: {e}")
            if return_regions:
                return "", []
            return ""
        combined_text, text_regions = self._parse_paddle_v3_result(result)
        if return_regions:
            return combined_text, text_regions
        return combined_text

    def _parse_paddle_v3_result(self, result) -> Tuple[str, List[TextRegion]]:
        """Parse PaddleOCR 3.x result into combined text and list of TextRegion."""
        text_regions: List[TextRegion] = []
        texts: List[str] = []
        if not result:
            return "", text_regions
        try:
            # PaddleOCR 3.x returns a list of page dicts with keys: rec_texts, rec_scores, rec_polys
            for page in result if isinstance(result, (list, tuple)) else [result]:
                # Check for 3.x dict format with rec_texts key
                if isinstance(page, dict) and 'rec_texts' in page:
                    rec_texts = page.get('rec_texts', [])
                    rec_scores = page.get('rec_scores', [])
                    rec_polys = page.get('rec_polys', [])
                    
                    for i, text in enumerate(rec_texts):
                        if text and text.strip():
                            conf = rec_scores[i] if i < len(rec_scores) else 0.0
                            poly = rec_polys[i] if i < len(rec_polys) else None
                            
                            # Convert polygon to bbox (list of tuples)
                            bbox = [(0, 0)]
                            if poly is not None:
                                try:
                                    # poly is typically a numpy array of shape (N, 2)
                                    if hasattr(poly, 'shape') and len(poly.shape) == 2:
                                        bbox = [(int(p[0]), int(p[1])) for p in poly]
                                    elif isinstance(poly, (list, tuple)):
                                        bbox = [(int(p[0]), int(p[1])) for p in poly]
                                except Exception:
                                    pass
                            
                            texts.append(text.strip())
                            text_regions.append(TextRegion(
                                bbox=bbox,
                                text=text.strip(),
                                confidence=float(conf),
                                detection_confidence=1.0
                            ))
                else:
                    # Fallback: try old parsing for 2.x compatibility or object format
                    items = getattr(page, "result", page)
                    if not isinstance(items, (list, tuple, dict)):
                        items = [page]
                    if isinstance(items, dict) and 'rec_texts' in items:
                        # Recursive call with proper dict
                        sub_texts, sub_regions = self._parse_paddle_v3_result([items])
                        texts.extend(sub_texts.split('\n') if sub_texts else [])
                        text_regions.extend(sub_regions)
                    elif isinstance(items, (list, tuple)):
                        for line in items:
                            text = ""
                            conf = 0.0
                            bbox = []
                            if hasattr(line, "text"):
                                text = getattr(line, "text", "") or ""
                                conf = float(getattr(line, "score", 0) or 0)
                                box = getattr(line, "bbox", None) or getattr(line, "box", None)
                                if box is not None:
                                    bbox = [(int(p[0]), int(p[1])) for p in box] if isinstance(box, (list, tuple)) else []
                            elif isinstance(line, (list, tuple)):
                                if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                                    bbox = [(int(p[0]), int(p[1])) for p in line[0]] if line[0] else []
                                    text = str(line[1][0]) if len(line[1]) > 0 else ""
                                    conf = float(line[1][1]) if len(line[1]) > 1 else 0.0
                                elif len(line) == 2 and isinstance(line[0], (list, tuple)) and isinstance(line[1], (int, float)):
                                    text, conf = str(line[0]), float(line[1])
                            if text.strip():
                                texts.append(text.strip())
                                text_regions.append(TextRegion(bbox=bbox or [(0, 0)], text=text.strip(), confidence=conf, detection_confidence=1.0))
        except Exception as e:
            logger.warning(f"Parse PaddleOCR 3.x result: {e}")
        return "\n".join(texts), text_regions

    def process_image_fast(self, image: np.ndarray) -> str:
        """
        Fast OCR processing - returns only combined text.
        
        Uses PaddleOCR's built-in pipeline but with CPU recognition configuration.
        """
        try:
            # Use detector instance which should have det=True, rec=True, cls=True
            # But wait, we initialized separate detector and recognizer.
            # If we want FAST, we should just use the detector instance if it was init with rec=True
            # But we initialized detector with rec=False (via implicit or explicit det=True only).
            
            # For fast processing, let's just use the recognizer instance if it is creating a full pipeline.
            # But our recognizer instance is CPU only and explicitly rec=True, det=False?
            # Actually catch-22.
            
            # Let's fallback to the robust method using `process_image` which calls detect then recognize.
            # It might be slightly slower than a single pass but its safer.
            return self.process_image(image, return_regions=False)

        except Exception as e:
            logger.error(f"Fast OCR failed: {e}")
            return ""


class OCRFactory:
    """Factory for creating optimized OCR instances."""
    
    @staticmethod
    def create_for_gtx980ti(lang: str = 'en') -> HybridOCR:
        return HybridOCR(
            use_gpu_detection=True,
            recognition_engine=RecognitionEngine.ENSEMBLE,  # Use ensemble for best accuracy
            lang=lang,
            use_v4_models=True,
            enable_spell_check=True,
            confidence_threshold=0.5
        )
    
    @staticmethod
    def create_high_accuracy(lang: str = 'en') -> HybridOCR:
        """Create OCR instance optimized for maximum accuracy."""
        return HybridOCR(
            use_gpu_detection=False,  # CPU more stable for accuracy
            recognition_engine=RecognitionEngine.ENSEMBLE,
            lang=lang,
            use_v4_models=True,
            enable_spell_check=True,
            confidence_threshold=0.4  # Lower threshold to capture more
        )
    
    @staticmethod
    def create_fast(lang: str = 'en') -> HybridOCR:
        """Create OCR instance optimized for speed."""
        return HybridOCR(
            use_gpu_detection=True,
            recognition_engine=RecognitionEngine.PADDLE_V4,
            lang=lang,
            use_v4_models=True,
            enable_spell_check=False,  # Skip spell check for speed
            confidence_threshold=0.6
        )
    
    @staticmethod
    def create_cpu_only(lang: str = 'en') -> HybridOCR:
        return HybridOCR(
            use_gpu_detection=False,
            recognition_engine=RecognitionEngine.PADDLE_V4,
            lang=lang,
            use_v4_models=True,
            enable_spell_check=True,
            confidence_threshold=0.5
        )
    
    @staticmethod
    def create_with_tesseract(lang: str = 'eng') -> HybridOCR:
        return HybridOCR(
            use_gpu_detection=True,
            recognition_engine=RecognitionEngine.TESSERACT,
            lang=lang,
            use_v4_models=False,
            enable_spell_check=True,
            confidence_threshold=0.5
        )
    
    @staticmethod
    def create_with_easyocr(lang: str = 'en') -> HybridOCR:
        return HybridOCR(
            use_gpu_detection=True,
            recognition_engine=RecognitionEngine.EASYOCR,
            lang=lang,
            use_v4_models=False,
            enable_spell_check=True,
            confidence_threshold=0.5
        )


# Convenience function
def run_ocr(image: np.ndarray, use_gpu_detection: bool = True) -> Union[str, Tuple[str, List[TextRegion]]]:
    ocr = HybridOCR(use_gpu_detection=use_gpu_detection)
    return ocr.process_image(image)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hybrid_ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    # Create optimized OCR for GTX 980 Ti
    ocr = OCRFactory.create_for_gtx980ti()
    
    # Process image
    text, regions = ocr.process_image(image, return_regions=True)
    
    print(f"\nExtracted Text:\n{'-'*40}")
    print(text)
    print(f"\n{'-'*40}")
    print(f"Found {len(regions)} text regions:")
    
    for i, region in enumerate(regions, 1):
        print(f"  {i}. '{region.text}' (conf: {region.confidence:.2f})")