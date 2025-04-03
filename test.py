import asyncio
from typing import Optional, List
import io
import json

from PIL import Image
import numpy as np

async def read_frame(input_path=r".\inputs\3-29-01.mov", frame_number=200) -> Optional[np.ndarray]:
    # The command to run
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"select=eq(n\\,{frame_number})",
        "-frames:v", '1',
        "-f", "image2pipe",  # output format to pipe the image
        "pipe:1"            # write to stdout
    ]
    
    # Create subprocess; stdout, stderr are captured asynchronously
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Wait for the process to finish, capturing output
    stdout, stderr = await process.communicate()
    
    # Handle success or error
    if process.returncode != 0:
        print("Error during FFmpeg execution. Return code:", process.returncode)
        if stderr:
            print("FFmpeg error output:\n", stderr.decode())
        return None
    return np.frombuffer(stdout, np.uint8)


async def read_video_info(input_path=r".\inputs\3-29-01.mov"):
    cmd = [
        "ffprobe",
        "-show_format", "-show_streams",
        "-of", "json",
        input_path
    ]
    output = await std_async_call(cmd)
    if output is None:
        return
    probe = json.loads(output.decode('utf-8'))
    info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return (info['nb_frames'], info['width'], info['height'])

async def std_async_call(cmd: List[str]) -> Optional[bytes]:
    """Run a command asynchronously and return the output."""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        print("Error during command execution. Return code:", process.returncode)
        if stderr:
            print("Command error output:\n", stderr.decode())
        return None
    
    return stdout

async def main():
    frame_bytes = await read_frame()
    if frame_bytes is None:
        return
    
    # Display the image in Jupyter Notebook
    print(frame_bytes.shape)  # Print the shape of the image array

asyncio.run(read_video_info())