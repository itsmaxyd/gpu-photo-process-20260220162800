package com.ocr.client

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.ocr.client.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var capturedBitmap: Bitmap? = null
    
    // Server configuration
    private var serverUrl: String = "http://192.168.1.100:8000"  // Default, should be configurable
    
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .build()
    
    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] ?: false
        val storageGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions[Manifest.permission.READ_MEDIA_IMAGES] ?: false
        } else {
            permissions[Manifest.permission.READ_EXTERNAL_STORAGE] ?: false
        }
        
        if (cameraGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }
    
    // Gallery picker
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            val inputStream = contentResolver.openInputStream(it)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            
            bitmap?.let { bmp ->
                capturedBitmap = bmp
                displayImage(bmp)
                binding.btnSendPhoto.isEnabled = true
            }
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Load saved server URL
        val sharedPrefs = getSharedPreferences("OCRClient", MODE_PRIVATE)
        serverUrl = sharedPrefs.getString("server_url", serverUrl) ?: serverUrl
        binding.etServerUrl.setText(serverUrl)
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Setup UI listeners
        setupListeners()
        
        // Request permissions and start camera
        checkPermissionsAndStart()
    }
    
    private fun setupListeners() {
        // Save server URL
        binding.btnSaveUrl.setOnClickListener {
            serverUrl = binding.etServerUrl.text.toString().trim()
            if (serverUrl.isNotEmpty()) {
                val sharedPrefs = getSharedPreferences("OCRClient", MODE_PRIVATE)
                sharedPrefs.edit().putString("server_url", serverUrl).apply()
                Toast.makeText(this, "Server URL saved", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Capture photo
        binding.btnCapturePhoto.setOnClickListener {
            capturePhoto()
        }
        
        // Pick from gallery
        binding.btnPickGallery.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
        
        // Send photo
        binding.btnSendPhoto.setOnClickListener {
            capturedBitmap?.let { bitmap ->
                sendPhotoToServer(bitmap)
            } ?: run {
                Toast.makeText(this, "No photo captured", Toast.LENGTH_SHORT).show()
            }
        }
        
        // Clear result
        binding.btnClearResult.setOnClickListener {
            binding.tvResult.text = ""
            capturedBitmap = null
            binding.ivPreview.setImageResource(android.R.color.darker_gray)
            binding.btnSendPhoto.isEnabled = false
        }
        
        // Test connection
        binding.btnTestConnection.setOnClickListener {
            testServerConnection()
        }
    }
    
    private fun checkPermissionsAndStart() {
        val permissions = mutableListOf(
            Manifest.permission.CAMERA
        )
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        
        val needsRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (needsRequest.isEmpty()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(permissions.toTypedArray())
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }
            
            // Image capture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            
            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
                Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_LONG).show()
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun capturePhoto() {
        val imageCapture = imageCapture ?: return
        
        imageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed", exception)
                    runOnUiThread {
                        Toast.makeText(
                            this@MainActivity,
                            "Capture failed: ${exception.message}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
                
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxyToBitmap(imageProxy)
                    imageProxy.close()
                    
                    bitmap?.let { bmp ->
                        capturedBitmap = bmp
                        runOnUiThread {
                            displayImage(bmp)
                            binding.btnSendPhoto.isEnabled = true
                            Toast.makeText(
                                this@MainActivity,
                                "Photo captured",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                }
            }
        )
    }
    
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    
    private fun displayImage(bitmap: Bitmap) {
        // Scale down for preview
        val maxWidth = 800
        val scale = maxWidth.toFloat() / bitmap.width.coerceAtLeast(1)
        val scaledBitmap = if (scale < 1) {
            Bitmap.createScaledBitmap(bitmap, maxWidth, (bitmap.height * scale).toInt(), true)
        } else {
            bitmap
        }
        
        binding.ivPreview.setImageBitmap(scaledBitmap)
    }
    
    private fun sendPhotoToServer(bitmap: Bitmap) {
        showLoading(true)
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Compress bitmap to JPEG
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream)
                val imageBytes = outputStream.toByteArray()
                
                // Create multipart request
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart(
                        "file",
                        "photo_${System.currentTimeMillis()}.jpg",
                        imageBytes.toRequestBody("image/jpeg".toMediaType())
                    )
                    .build()
                
                val request = Request.Builder()
                    .url("$serverUrl/upload/sync")
                    .post(requestBody)
                    .build()
                
                val response = client.newCall(request).execute()
                
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string()
                        parseAndDisplayResult(responseBody)
                    } else {
                        binding.tvResult.text = "Error: ${response.code} - ${response.message}"
                        Toast.makeText(
                            this@MainActivity,
                            "Upload failed: ${response.code}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    binding.tvResult.text = "Error: ${e.message}"
                    Toast.makeText(
                        this@MainActivity,
                        "Error: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
                Log.e(TAG, "Upload error", e)
            }
        }
    }
    
    private fun parseAndDisplayResult(responseBody: String?) {
        try {
            val json = JSONObject(responseBody ?: "{}")
            val success = json.optBoolean("success", false)
            val message = json.optString("message", "")
            val filename = json.optString("filename", "")
            
            if (success) {
                // Fetch the result
                fetchResult(filename)
            } else {
                binding.tvResult.text = "Upload failed: $message"
            }
        } catch (e: Exception) {
            binding.tvResult.text = "Parse error: ${e.message}\n\nRaw: $responseBody"
        }
    }
    
    private fun fetchResult(filename: String) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Wait a bit for processing
                Thread.sleep(1000)
                
                val request = Request.Builder()
                    .url("$serverUrl/result/$filename")
                    .get()
                    .build()
                
                val response = client.newCall(request).execute()
                
                withContext(Dispatchers.Main) {
                    if (response.isSuccessful) {
                        val responseBody = response.body?.string()
                        val json = JSONObject(responseBody ?: "{}")
                        val text = json.optString("extracted_text", "")
                        val totalTime = json.optString("total_time", "")
                        
                        binding.tvResult.text = buildString {
                            append("File: $filename\n")
                            append("Time: $totalTime\n\n")
                            append("Extracted Text:\n")
                            append(text)
                        }
                    } else {
                        binding.tvResult.text = "File uploaded but result not yet available.\nFilename: $filename"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    binding.tvResult.text = "File uploaded: $filename\nError fetching result: ${e.message}"
                }
            }
        }
    }
    
    private fun testServerConnection() {
        showLoading(true)
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val request = Request.Builder()
                    .url("$serverUrl/health")
                    .get()
                    .build()
                
                val response = client.newCall(request).execute()
                
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    
                    if (response.isSuccessful) {
                        Toast.makeText(
                            this@MainActivity,
                            "Server connection successful!",
                            Toast.LENGTH_LONG
                        ).show()
                        binding.tvResult.text = "Server is reachable at $serverUrl"
                    } else {
                        Toast.makeText(
                            this@MainActivity,
                            "Server responded with ${response.code}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    Toast.makeText(
                        this@MainActivity,
                        "Connection failed: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    binding.tvResult.text = "Connection failed: ${e.message}\n\nMake sure:\n1. Server is running\n2. IP address is correct\n3. Both devices are on same network"
                }
            }
        }
    }
    
    private fun showLoading(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnSendPhoto.isEnabled = !show
        binding.btnCapturePhoto.isEnabled = !show
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
    
    companion object {
        private const val TAG = "OCRClient"
    }
}