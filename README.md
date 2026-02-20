# 🚀 GTX 980 Ti Optimized OCR System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-20.10+-2496ed?style=flat&logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-ff4b4b?style=flat&logo=streamlit)](https://streamlit.io/)
[![CUDA](https://img.shields.io/badge/CUDA-11.2.2-76b900?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

A GPU-accelerated OCR system specifically optimized for **NVIDIA GTX 980 Ti (Maxwell architecture)**. This system monitors a folder for screen photos, processes them using GPU-accelerated OpenCV operations, and performs OCR using a hybrid approach (GPU detection + CPU recognition).

## ✨ Features

- 🖥️ **Streamlit Dashboard** - Web interface for monitoring and viewing OCR results
- 📱 **Android Client** - Send photos directly from your Android device over LAN
- 🚀 **REST API** - FastAPI endpoints for programmatic access
- ⚡ **GPU Preprocessing** - 4-10x speedup using OpenCV CUDA operations
- 🔄 **Hybrid OCR** - GPU detection + CPU recognition for Maxwell compatibility
- 📊 **Real-time Monitoring** - Track processing times and success rates
- 🔔 **Auto-notifications** - Get notified when OCR completes

## 🎯 Key Optimizations for GTX 980 Ti

### Why These Optimizations?

The GTX 980 Ti uses the **Maxwell architecture (compute capability 5.2)**, which has limited support for modern deep learning frameworks. This system is designed to work around these limitations:

1. **CUDA 11.2.2, cuDNN 8.2.1**: Older but compatible versions that support Maxwell architecture
2. **GPU-Accelerated Preprocessing**: Leverages the 980 Ti's strong memory bandwidth (336.5 GB/s) for OpenCV CUDA operations, achieving 4-10x speedups
3. **Hybrid OCR Approach**: 
   - **GPU**: Text detection (parallelizable, benefits from memory bandwidth)
   - **CPU**: Text recognition (avoids Maxwell compatibility issues, Tesseract doesn't benefit from GPU anyway)

## 📋 Prerequisites

### Hardware Requirements
- **NVIDIA GTX 980 Ti** (or other Maxwell-based GPU)
- Minimum 4GB VRAM (6GB recommended)
- 8GB+ System RAM

### Software Requirements
- **Docker** 20.10+
- **Docker Compose** 2.0+
- **NVIDIA Driver** 470.x+ (for CUDA 11.2)
- **NVIDIA Container Toolkit** (nvidia-docker2)

### Verify Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi

# Should show driver version 470+ and your GTX 980 Ti

# Check Docker
docker --version

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.2.2-base nvidia-smi
```

## 🚀 Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd gpu-photo-process

# Copy environment configuration (optional)
cp .env.example .env
# Edit .env to customize settings
```

### 2. Build and Run

```bash
# Build the Docker image (this may take 10-15 minutes)
docker-compose build

# Start the container
docker-compose up -d

# Check logs
docker-compose logs -f
```

### 3. Access the Dashboard

Open your browser to: [http://localhost:8501](http://localhost:8501)

### 4. Process Images

Place images (`.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`) into the `./data/images` directory:
- The system checks for new files every 15 seconds
- Results are saved to `./data/results/results.csv`
- View results in the web dashboard

## 📁 Project Structure

```
gpu-photo-process/
├── Dockerfile                 # CUDA 11.2.2 optimized container
├── docker-compose.yml         # Container orchestration
├── .dockerignore              # Docker build exclusions
├── .env.example               # Environment configuration template
├── README.md                  # This file
├── data/
│   ├── images/                # Input folder (mounted as watch_folder)
│   └── results/               # Output directory for results.csv
├── docker/
│   └── start.sh               # Container startup script
└── ocr-system/
    ├── app.py                 # Main Streamlit application
    ├── gpu_preprocessing.py   # GPU-accelerated preprocessing module
    ├── hybrid_ocr.py          # Hybrid OCR (GPU detection + CPU recognition)
    ├── ocr_system.py          # Core OCR system
    ├── api_server.py          # FastAPI REST server
    ├── requirements.txt       # Python dependencies
    ├── README.md              # Detailed technical documentation
    └── android-client/        # Android app source
        └── OCRClient/
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCH_FOLDER` | `/app/watch_folder` | Folder to monitor for images |
| `RESULTS_FILE` | `/app/results.csv` | Output CSV file |
| `POLL_INTERVAL` | `15` | Seconds between folder scans |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID to use |

### Using Environment File

```bash
# Copy the example file
cp .env.example .env

# Edit configuration
nano .env

# Restart container to apply changes
docker-compose down && docker-compose up -d
```

### GPU Selection

If you have multiple GPUs, select a specific one:

```yaml
# In docker-compose.yml, modify:
environment:
  - CUDA_VISIBLE_DEVICES=0  # Use first GPU
  # - CUDA_VISIBLE_DEVICES=1  # Use second GPU
```

### Adjusting for Other Maxwell GPUs

This system works with other Maxwell-based GPUs:
- GTX 750 Ti
- GTX 960, GTX 970, GTX 980
- Titan X (Maxwell)

No configuration changes needed - the system auto-detects GPU capabilities.

## 🧪 Local Development (without Docker)

Test the OCR pipeline locally before building the Docker image:

1. **Create and activate a venv** (from project root):
   ```bash
   cd ocr-system
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Create the watch folder**:
   ```bash
   mkdir -p watch_folder
   ```

3. **Run the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Optional: Run the API server** (in another terminal):
   ```bash
   uvicorn api_server:app --reload
   ```

## 🌐 REST API Endpoints

The server provides the following API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/upload` | POST | Upload photo (async processing) |
| `/upload/sync` | POST | Upload photo (sync processing) |
| `/results` | GET | Get recent OCR results |
| `/result/{filename}` | GET | Get result for specific file |
| `/image/{filename}` | GET | Get original image |
| `/results` | DELETE | Clear all results |

### API Usage Examples

**Upload a photo**:
```bash
curl -X POST "http://localhost:8000/upload/sync" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

**Get results**:
```bash
curl "http://localhost:8000/results?limit=5"
```

**Check status**:
```bash
curl "http://localhost:8000/status"
```

## 📱 Android Client

The Android client allows you to capture and send photos directly to the OCR server over your local network.

### Building the Android App

1. **Prerequisites**:
   - Android Studio Arctic Fox or later
   - Android SDK 24+ (Android 7.0)
   - Kotlin 1.9+

2. **Build Steps**:
   ```bash
   cd ocr-system/android-client/OCRClient
   # Open in Android Studio or build from command line
   ./gradlew assembleDebug
   ```

3. **Install on Device**:
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

### Using the Android Client

1. **Configure Server URL**:
   - Find your server's LAN IP address: `hostname -I` or `ipconfig`
   - Enter the URL in the app: `http://YOUR_SERVER_IP:8000`

2. **Test Connection**:
   - Tap "Test Connection" to verify connectivity

3. **Capture & Send**:
   - Use the camera to capture a photo, or pick from gallery
   - Tap "Send to OCR Server"

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size or image resolution
# Or disable GPU preprocessing in code:
# preprocessor = GPUPreprocessor(use_gpu=False)
```

### CUDA Initialization Failed

```bash
# Check CUDA compatibility
docker exec -it ocr_gpu_monitor python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Should return 1 or more
```

### PaddleOCR GPU Not Working

```bash
# Check PaddlePaddle GPU status
docker exec -it ocr_gpu_monitor python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Image too large | Resize images to < 4K resolution |
| `No CUDA devices found` | Driver issue | Update NVIDIA driver to 470+ |
| `cublas create failed` | CUDA version mismatch | Rebuild container |
| `Compute capability < 5.2` | Incompatible GPU | Use CPU-only mode |

## 📊 Monitoring

### Container Logs

```bash
# Follow logs
docker-compose logs -f

# Check specific time range
docker-compose logs --since 1h
```

### GPU Utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Or use nvtop
sudo apt install nvtop
nvtop
```

## 🔒 Security Considerations

- The container runs with GPU access but limited system permissions
- Only the `./data/images` and `./data/results.csv` are mounted
- CORS is enabled for all origins - restrict in production

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [OpenCV](https://opencv.org/) - Image processing with CUDA support
- [NVIDIA](https://developer.nvidia.com/) - CUDA toolkit and documentation

---

**Note**: This system is specifically optimized for GTX 980 Ti (Maxwell). For newer GPUs (Pascal, Turing, Ampere, etc.), consider using newer CUDA versions and full GPU acceleration for both detection and recognition.
