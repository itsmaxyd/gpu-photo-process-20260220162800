"""
REST API Server for Android Client Photo Upload

This FastAPI server provides endpoints for:
- Uploading photos from Android devices over LAN
- Getting OCR results
- Checking system status

The server runs alongside the Streamlit dashboard and processes
uploaded images using the same GPU-optimized pipeline.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, List
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WATCH_FOLDER = os.environ.get("WATCH_FOLDER", "/app/watch_folder")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "/app/results.csv")
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Create FastAPI app
app = FastAPI(
    title="GTX 980 Ti OCR API",
    description="REST API for Android clients to upload photos for OCR processing",
    version="1.0.0"
)

# Enable CORS for Android clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your LAN IP range
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class UploadResponse(BaseModel):
    """Response model for file upload."""
    success: bool
    message: str
    filename: str
    file_id: str
    timestamp: str


class OCRResult(BaseModel):
    """Model for OCR result."""
    timestamp: str
    filename: str
    extracted_text: str
    preprocess_time: Optional[str] = None
    ocr_time: Optional[str] = None
    total_time: Optional[str] = None


class SystemStatus(BaseModel):
    """Model for system status."""
    status: str
    gpu_available: bool
    watch_folder: str
    total_processed: int
    uptime: str


class ResultList(BaseModel):
    """Model for list of results."""
    count: int
    results: List[OCRResult]


# Global state
processing_queue = asyncio.Queue()
ocr_processor = None
gpu_preprocessor = None
start_time = datetime.now()


def init_ocr_components():
    """Initialize OCR components (lazy loading)."""
    global ocr_processor, gpu_preprocessor
    
    if ocr_processor is None:
        from hybrid_ocr import OCRFactory
        ocr_processor = OCRFactory.create_for_gtx980ti(lang='en')
        logger.info("OCR processor initialized")
    
    if gpu_preprocessor is None:
        from gpu_preprocessing import GPUPreprocessor
        gpu_preprocessor = GPUPreprocessor(use_gpu=True)
        logger.info("GPU preprocessor initialized")
    
    return ocr_processor, gpu_preprocessor


def process_image_sync(file_path: str) -> dict:
    """
    Process a single image synchronously.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dictionary with processing results
    """
    import time
    import pandas as pd
    import cv2
    
    ocr, preprocessor = init_ocr_components()
    
    stats = {
        'preprocess_time': 0.0,
        'ocr_time': 0.0,
        'total_time': 0.0
    }
    
    total_start = time.time()
    
    # Preprocess
    preprocess_start = time.time()
    original_img, processed_img, status = preprocessor.preprocess_for_ocr(file_path)
    stats['preprocess_time'] = time.time() - preprocess_start
    
    if original_img is None:
        return {
            'success': False,
            'error': f"Preprocessing failed: {status}",
            'text': ''
        }
    
    # OCR
    ocr_start = time.time()
    try:
        extracted_text = ocr.process_image_fast(processed_img)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        extracted_text = f"OCR Error: {str(e)}"
    stats['ocr_time'] = time.time() - ocr_start
    
    stats['total_time'] = time.time() - total_start
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(file_path)
    
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "Filename": filename,
        "Extracted Text": extracted_text,
        "Preprocess Time": f"{stats['preprocess_time']:.3f}s",
        "OCR Time": f"{stats['ocr_time']:.3f}s",
        "Total Time": f"{stats['total_time']:.3f}s"
    }])
    
    # Write header when file missing or empty (match ocr_system logic)
    header = not (os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0)
    new_entry.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
    
    return {
        'success': True,
        'text': extracted_text,
        'stats': stats,
        'timestamp': timestamp
    }


async def process_queue():
    """Background task to process images from queue."""
    while True:
        try:
            file_path = await processing_queue.get()
            logger.info(f"Processing queued file: {file_path}")
            
            # Run processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_image_sync, file_path)
            
            if result['success']:
                logger.info(f"Successfully processed: {file_path}")
            else:
                logger.error(f"Failed to process {file_path}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        
        await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Ensure watch folder exists
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    
    # Start background queue processor
    asyncio.create_task(process_queue())
    
    logger.info("API server started")


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "GTX 980 Ti OCR API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "upload_sync": "/upload/sync",
            "results": "/results",
            "result/{filename}": "/result/{filename}",
            "status": "/status",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    import pandas as pd
    
    total_processed = 0
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            total_processed = len(df)
        except Exception:
            pass
    
    # Check GPU availability
    gpu_available = False
    try:
        import cv2
        gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        pass
    
    uptime = datetime.now() - start_time
    
    return SystemStatus(
        status="running",
        gpu_available=gpu_available,
        watch_folder=WATCH_FOLDER,
        total_processed=total_processed,
        uptime=str(uptime).split('.')[0]  # Remove microseconds
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_async: bool = True
):
    """
    Upload a photo for OCR processing.
    
    The file will be saved to the watch folder and processed.
    If process_async is True (default), processing happens in background.
    If process_async is False, waits for processing to complete.
    
    Args:
        file: Uploaded file
        process_async: Whether to process asynchronously
        
    Returns:
        UploadResponse with file info
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file_id}_{file.filename}"
    file_path = os.path.join(WATCH_FOLDER, safe_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file: {safe_filename}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Process file
    if process_async:
        # Add to processing queue
        await processing_queue.put(file_path)
        message = "File uploaded and queued for processing"
    else:
        # Process immediately (blocking)
        result = process_image_sync(file_path)
        if result['success']:
            message = "File uploaded and processed successfully"
        else:
            message = f"File uploaded but processing failed: {result.get('error')}"
    
    return UploadResponse(
        success=True,
        message=message,
        filename=safe_filename,
        file_id=file_id,
        timestamp=datetime.now().isoformat()
    )


@app.post("/upload/sync", response_model=UploadResponse)
async def upload_file_sync(file: UploadFile = File(...)):
    """
    Upload a photo and wait for OCR processing to complete.
    
    This endpoint blocks until processing is done and returns
    the result immediately.
    
    Args:
        file: Uploaded file
        
    Returns:
        UploadResponse with file info
    """
    return await upload_file(file=file, process_async=False)


@app.get("/results", response_model=ResultList)
async def get_results(limit: int = 10):
    """
    Get recent OCR results.
    
    Args:
        limit: Maximum number of results to return (default: 10)
        
    Returns:
        ResultList with recent results
    """
    import pandas as pd
    
    if not os.path.exists(RESULTS_FILE):
        return ResultList(count=0, results=[])
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        df = df.tail(limit).iloc[::-1]  # Most recent first
        
        results = []
        for _, row in df.iterrows():
            results.append(OCRResult(
                timestamp=row.get('Timestamp', ''),
                filename=row.get('Filename', ''),
                extracted_text=row.get('Extracted Text', ''),
                preprocess_time=row.get('Preprocess Time'),
                ocr_time=row.get('OCR Time'),
                total_time=row.get('Total Time')
            ))
        
        return ResultList(count=len(results), results=results)
        
    except Exception as e:
        logger.error(f"Failed to read results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read results: {str(e)}")


@app.get("/result/{filename}")
async def get_result(filename: str):
    """
    Get OCR result for a specific file.
    
    Args:
        filename: Name of the file
        
    Returns:
        OCRResult for the specified file
    """
    import pandas as pd
    
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(status_code=404, detail="No results found")
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        match = df[df['Filename'] == filename]
        
        if match.empty:
            raise HTTPException(status_code=404, detail=f"No result found for: {filename}")
        
        row = match.iloc[0]
        return OCRResult(
            timestamp=row.get('Timestamp', ''),
            filename=row.get('Filename', ''),
            extracted_text=row.get('Extracted Text', ''),
            preprocess_time=row.get('Preprocess Time'),
            ocr_time=row.get('OCR Time'),
            total_time=row.get('Total Time')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")


@app.delete("/results")
async def clear_results():
    """Clear all OCR results."""
    try:
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        return {"success": True, "message": "Results cleared"}
    except Exception as e:
        logger.error(f"Failed to clear results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear results: {str(e)}")


@app.get("/image/{filename}")
async def get_image(filename: str):
    """
    Get the original image file.
    
    Args:
        filename: Name of the image file
        
    Returns:
        Image file
    """
    file_path = os.path.join(WATCH_FOLDER, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    return FileResponse(file_path)


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )