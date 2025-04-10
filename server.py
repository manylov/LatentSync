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

from types import SimpleNamespace

# Environment variables
AUTH_HEADER = os.getenv('AUTH_HEADER')
R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME')

if not all([AUTH_HEADER, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
    raise ValueError("Missing required environment variables")

app = FastAPI()

class GenerateRequest(BaseModel):
    audio: str
    video: str

def setup_s3():
    return S3FileSystem(
        key=R2_ACCESS_KEY_ID,
        secret=R2_SECRET_ACCESS_KEY,
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'
    )

async def download_file(url: str, path: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)

def run_inference(video_path: str, audio_path: str, output_path: str):
    try:
        # Create args namespace with the same parameters as CLI
        args = SimpleNamespace(
            unet_config_path='configs/unet/stage2.yaml',
            inference_ckpt_path='checkpoints/latentsync_unet.pt',
            inference_steps=20,
            guidance_scale=1.5,
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=output_path,
            seed=1247  # Using default from inference.py
        )
        
        # Load config
        config = OmegaConf.load(args.unet_config_path)
        
        # Call main directly
        main(config, args)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    auth: str = Header(None)
):
    # Check auth header
    if auth != AUTH_HEADER:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Generate unique ID for this request
        # Extract filename from video URL without extension
        video_filename = os.path.basename(request.video)
        request_id = os.path.splitext(video_filename)[0]
        
        # Create assets directory if it doesn't exist
        Path("assets").mkdir(exist_ok=True)
        
        # Define file paths
        # video_path = f"assets/{request_id}_video.mp4"
        # audio_path = f"assets/{request_id}_audio.wav"
        # output_path = f"/tmp/{request_id}.mp4"

        video_path = f"assets/demo1_video.mp4"
        audio_path = f"assets/demo1_audio.wav"
        output_path = f"assets/demo3_video.mp4"

        # Download files
        # await asyncio.gather(
        #     download_file(request.video, video_path),
        #     download_file(request.audio, audio_path)
        # )

        # Run inference
        # run_inference(video_path, audio_path, output_path)

        # Upload to Cloudflare R2
        s3 = setup_s3()
        s3.put(output_path, f"{R2_BUCKET_NAME}/output/{request_id}.mp4")

        # Cleanup local files
        # for file in [video_path, audio_path, output_path]:
        #     if os.path.exists(file):
        #         os.remove(file)

        return {"output": f"{request_id}.mp4"}

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download files: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081) 