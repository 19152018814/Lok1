from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import time
import platform
import psutil
from datetime import datetime

from infer import main_process


# ANSI color helper
class Fmt:
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    WHITE  = "\033[97m"
    END    = "\033[0m"


def print_banner():
    banner = f"""
{Fmt.CYAN}{Fmt.BOLD}
+======================================================================+
|                                                                      |
|               IMAR Inference API Server                             |
|                                                                      |
|     Video Image Colorimetric Detection & Analysis                    |
|              Trinder-1 Glucose Detection                            |
|                                                                      |
+======================================================================+
{Fmt.END}"""
    print(banner)


def print_system_info():
    print(f"{Fmt.YELLOW}{Fmt.BOLD}# System Information:{Fmt.END}")
    print(f"  OS:       {Fmt.CYAN}{platform.system()} {platform.release()}{Fmt.END}")
    print(f"  Python:   {Fmt.CYAN}{platform.python_version()}{Fmt.END}")
    print(f"  Memory:   {Fmt.CYAN}{psutil.virtual_memory().total // (1024**3)} GB{Fmt.END}")
    print(f"  CPU cores:{Fmt.CYAN}{psutil.cpu_count()}{Fmt.END}")
    print(f"  CWD:      {Fmt.CYAN}{os.getcwd()}{Fmt.END}")
    print()


def print_server_config():
    print(f"{Fmt.GREEN}{Fmt.BOLD}# Server Configuration:{Fmt.END}")
    print(f"  Host:           {Fmt.CYAN}0.0.0.0{Fmt.END}")
    print(f"  Port:           {Fmt.CYAN}8000{Fmt.END}")
    print(f"  Auto-reload:    {Fmt.CYAN}Enabled{Fmt.END}")
    print(f"  Log level:      {Fmt.CYAN}info{Fmt.END}")
    print(f"  Upload dir:     {Fmt.CYAN}{os.path.abspath('uploads')}{Fmt.END}")
    print()


def print_available_models():
    print(f"{Fmt.CYAN}{Fmt.BOLD}# Available Models:{Fmt.END}")
    print(f"  Trinder-1 Detection {Fmt.YELLOW}(mmol/L){Fmt.END}")
    print()


def print_endpoints():
    endpoints = [
        ("GET",  "/",                   "Root endpoint"),
        ("GET",  "/health",             "Health check"),
        ("POST", "/process_images",     "Process images with AI models"),
        ("POST", "/upload",             "Simple image upload"),
        ("GET",  "/models",             "Get available models"),
        ("GET",  "/results/{timestamp}", "Get processing results by timestamp"),
    ]
    print(f"{Fmt.CYAN}{Fmt.BOLD}# API Endpoints:{Fmt.END}")
    for method, path, desc in endpoints:
        color = Fmt.GREEN if method == "GET" else Fmt.YELLOW
        print(f"  {color}{method:<7}{Fmt.END} {Fmt.CYAN}{path:<27}{Fmt.END} {desc}")
    print()


def print_startup_complete():
    print(f"{Fmt.GREEN}{Fmt.BOLD}Server startup complete!{Fmt.END}")
    print(f"  Server:  {Fmt.CYAN}http://0.0.0.0:8000{Fmt.END}")
    print(f"  Docs:    {Fmt.CYAN}http://0.0.0.0:8000/docs{Fmt.END}")
    print(f"  ReDoc:   {Fmt.CYAN}http://0.0.0.0:8000/redoc{Fmt.END}")
    print()
    print(f"{Fmt.CYAN}{'='*70}{Fmt.END}")
    print(f"{Fmt.CYAN}Waiting for requests...{Fmt.END}")
    print(f"{Fmt.CYAN}{'='*70}{Fmt.END}")


app = FastAPI(
    title="IMAR API",
    version="1.0.0",
    description="Intelligent Microdroplet Analytical Robotics — Trinder-1 Inference Pipeline",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS from any origin (restrict to specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR   = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
request_count = 0
start_time    = time.time()


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request, call_next):
    global request_count
    request_count += 1

    start_t = time.time()
    print(f"{Fmt.CYAN}[{datetime.now().strftime('%H:%M:%S')}] Request #{request_count}{Fmt.END}")
    print(f"  {Fmt.YELLOW}{request.method}{Fmt.END} {Fmt.CYAN}{request.url.path}{Fmt.END}")
    print(f"  Client: {Fmt.CYAN}{request.client.host if request.client else 'unknown'}{Fmt.END}")

    response = await call_next(request)

    elapsed = time.time() - start_t
    status_color = Fmt.GREEN if response.status_code < 400 else Fmt.RED
    print(f"  Status: {status_color}{response.status_code}{Fmt.END} | "
          f"Time: {Fmt.CYAN}{elapsed:.3f}s{Fmt.END}")
    print(f"  Memory: {Fmt.CYAN}{psutil.Process().memory_info().rss // (1024*1024)} MB{Fmt.END}")
    print()
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message":     "IMAR Inference API is running",
        "version":     "1.0.0",
        "description": "Intelligent Microdroplet Analytical Robotics — Trinder-1 Glucose",
        "timestamp":   datetime.now().isoformat(),
        "status":     "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check — returns system resource usage."""
    mem       = psutil.virtual_memory()
    cpu_pct   = psutil.cpu_percent(interval=1)
    return {
        "status":   "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "memory_usage_percent": mem.percent,
            "memory_available_gb": round(mem.available / (1024**3), 2),
            "cpu_usage_percent":    cpu_pct,
            "disk_usage_percent":  psutil.disk_usage("/").percent,
        },
        "server": {
            "uptime_seconds":  time.time() - start_time,
            "total_requests":  request_count,
        },
    }


@app.post("/process_images")
async def process_images_endpoint(
    original_image:   UploadFile = File(..., description="Original sample image"),
    background_image: UploadFile = File(..., description="Background reference image"),
    model_name: str = Form(..., description="Model name (only 'trinder_1' is supported)"),
):
    """
    Process a pair of images (sample + background) and return detection and
    concentration results.
    """
    print(f"{Fmt.CYAN}Image processing request received{Fmt.END}")
    print(f"  Original:  {Fmt.CYAN}{original_image.filename}{Fmt.END} "
          f"({Fmt.YELLOW}{original_image.size} bytes{Fmt.END})")
    print(f"  Background: {Fmt.CYAN}{background_image.filename}{Fmt.END} "
          f"({Fmt.YELLOW}{background_image.size} bytes{Fmt.END})")
    print(f"  Model:    {Fmt.CYAN}{model_name}{Fmt.END}")

    try:
        if not original_image.content_type.startswith("image/"):
            raise HTTPException(400, "Original image must be an image file")
        if not background_image.content_type.startswith("image/"):
            raise HTTPException(400, "Background image must be an image file")

        valid_models = ["trinder_1"]
        if model_name not in valid_models:
            raise HTTPException(400, f"Invalid model name. Must be one of: {valid_models}")

        print(f"  File validation passed; starting AI processing...")

        with tempfile.TemporaryDirectory() as temp_dir:
            original_path   = os.path.join(temp_dir, "original.jpg")
            background_path = os.path.join(temp_dir, "background.jpg")

            with open(original_path, "wb") as f:
                shutil.copyfileobj(original_image.file, f)
            with open(background_path, "wb") as f:
                shutil.copyfileobj(background_image.file, f)

            print(f"  Images saved to temp directory")
            t0 = time.time()
            result = main_process(original_path, background_path, model_name)
            print(f"  AI processing completed in {Fmt.CYAN}{time.time()-t0:.2f}s{Fmt.END}")

            if result.get("success"):
                return JSONResponse(content=result)
            else:
                print(f"  Processing failed: {result.get('error', 'unknown')}")
                raise HTTPException(500, result.get("error", "Processing failed"))

    except HTTPException:
        raise
    except Exception as e:
        print(f"  Error: {Fmt.RED}{str(e)}{Fmt.END}")
        raise HTTPException(500, str(e))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Simple image upload endpoint (raw file storage, no inference)."""
    print(f"{Fmt.CYAN}File upload request{Fmt.END}")
    print(f"  File: {Fmt.CYAN}{file.filename}{Fmt.END} "
          f"({Fmt.YELLOW}{file.size} bytes{Fmt.END})")

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"{timestamp}_{file.filename}"
        file_path  = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"  Saved: {Fmt.CYAN}{filename}{Fmt.END}")
        return {
            "success":  True,
            "filename": filename,
            "file_path": file_path,
            "message":  "File uploaded successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"  Upload failed: {Fmt.RED}{str(e)}{Fmt.END}")
        raise HTTPException(500, str(e))


@app.get("/models")
async def get_available_models():
    """List all available models and their metadata."""
    print(f"{Fmt.CYAN}Models list requested{Fmt.END}")
    return {
        "models": [
            {
                "name":        "trinder_1",
                "description": "Glucose concentration (Trinder-1 colorimetric method)",
                "unit":        "mmol/L",
            }
        ]
    }


@app.get("/results/{timestamp}")
async def get_result(timestamp: str):
    """Retrieve processing results by timestamp."""
    print(f"{Fmt.CYAN}Result request: {timestamp}{Fmt.END}")
    result_dir = os.path.join(UPLOAD_DIR, timestamp)
    if not os.path.exists(result_dir):
        raise HTTPException(404, "Result not found")
    print(f"  Found: {result_dir}")
    return {
        "timestamp": timestamp,
        "status":   "completed",
        "result_dir": result_dir,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    print_banner()
    print_system_info()
    print_server_config()
    print_available_models()
    print_endpoints()
    print_startup_complete()

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
