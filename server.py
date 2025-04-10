import os
import uuid
import urllib.request
from pathlib import Path
import asyncio
from typing import Dict
import httpx
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import subprocess
from s3fs import S3FileSystem
from omegaconf import OmegaConf
import logging
import time

from types import SimpleNamespace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
logger.info("Loading environment variables")
AUTH_HEADER = os.getenv('AUTH_HEADER')
R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')

if not all([AUTH_HEADER, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
    logger.error("Missing required environment variables")
    raise ValueError("Missing required environment variables")

logger.info("Initializing FastAPI application")
app = FastAPI()

class GenerateRequest(BaseModel):
    audio: str
    video: str

def setup_s3():
    logger.info(f"Setting up S3 connection to Cloudflare R2 bucket: {R2_BUCKET_NAME}")
    return S3FileSystem(
        key=R2_ACCESS_KEY_ID,
        secret=R2_SECRET_ACCESS_KEY,
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
    )

async def download_file(url: str, path: str):
    logger.info(f"Downloading file from {url} to {path}")
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
    download_time = time.time() - start_time
    file_size = os.path.getsize(path)
    logger.info(f"Downloaded {file_size} bytes to {path} in {download_time:.2f} seconds")

def run_inference(video_path: str, audio_path: str, output_path: str):
    logger.info(f"Starting inference with video={video_path}, audio={audio_path}, output={output_path}")
    start_time = time.time()
    try:
        # Spawn subprocess similar to inference.sh
        cmd = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", "20",
            "--guidance_scale", "1.5",
            "--video_path", video_path,
            "--audio_path", audio_path,
            "--video_out_path", output_path,
            "--seed", "1247"
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the subprocess and capture output
        process = subprocess.run(
            cmd,
            check=True,  # This will raise CalledProcessError if the process fails
            capture_output=True,  # Capture stdout and stderr
            text=True  # Convert output to string
        )
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed successfully in {inference_time:.2f} seconds")
        logger.debug(f"Inference stdout: {process.stdout}")
        
    except subprocess.CalledProcessError as e:
        # Handle subprocess execution errors
        error_msg = f"Inference process failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        # Handle any other errors
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Inference failed: {str(e)}")

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    auth: str = Header(None)
):
    logger.info("Received generation request")
    
    # Check auth header
    if auth != AUTH_HEADER:
        logger.warning("Unauthorized request attempt with invalid auth header")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Generate unique ID for this request
        # Extract filename from video URL without extension
        video_filename = os.path.basename(request.video)
        request_id = os.path.splitext(video_filename)[0]
        logger.info(f"Processing request with ID: {request_id}")
        
        # Create assets directory if it doesn't exist
        logger.info("Ensuring assets directory exists")
        Path("assets").mkdir(exist_ok=True)
        
        # Define file paths
        video_path = f"assets/{request_id}_video.mp4"
        audio_path = f"assets/{request_id}_audio.wav"
        output_path = f"/tmp/{request_id}.mp4"
        logger.info(f"File paths - Video: {video_path}, Audio: {audio_path}, Output: {output_path}")

        # Download files
        logger.info("Starting parallel download of video and audio files")
        download_start = time.time()
        await asyncio.gather(
            download_file(request.video, video_path),
            download_file(request.audio, audio_path)
        )
        logger.info(f"Downloads completed in {time.time() - download_start:.2f} seconds")

        # Run inference
        logger.info("Starting inference process")
        run_inference(video_path, audio_path, output_path)

        # Upload to Cloudflare R2
        logger.info(f"Uploading result to R2 bucket: {R2_BUCKET_NAME}/output/{request_id}.mp4")
        upload_start = time.time()
        s3 = setup_s3()
        s3.put(output_path, f"{R2_BUCKET_NAME}/output/{request_id}.mp4")
        logger.info(f"Upload completed in {time.time() - upload_start:.2f} seconds")

        # Cleanup local files
        logger.info("Cleaning up temporary files")
        for file in [video_path, audio_path, output_path]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed temporary file: {file}")

        logger.info(f"Request {request_id} completed successfully")
        return {"output": f"{request_id}.mp4"}

    except httpx.HTTPError as e:
        logger.error(f"Failed to download files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download files: {str(e)}")
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:8081")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        timeout_keep_alive=300,  # 5 minutes keep-alive timeout
        timeout_graceful_shutdown=300,  # 5 minutes graceful shutdown
        limit_concurrency=10,  # Limit concurrent connections
    ) 