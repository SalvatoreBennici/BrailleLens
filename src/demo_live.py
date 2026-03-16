import sys
import os
import cv2
import time
import subprocess
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.pipeline_model import EndToEndPipeline

CAMERA_INDEX = 10
HW_FRAME_WIDTH = 1920
HW_FRAME_HEIGHT = 1080
WINDOW_NAME = "Braille E2E Real-Time Translation"

def start_scrcpy_stream() -> subprocess.Popen:
    cmd = [
        "scrcpy",
        "--video-source=camera",
        "--camera-id=0", 
        f"--camera-size={HW_FRAME_WIDTH}x{HW_FRAME_HEIGHT}",
        f"--v4l2-sink=/dev/video{CAMERA_INDEX}",
        "--no-playback"
    ]
    
    clean_env = os.environ.copy()
    clean_env.pop('LD_LIBRARY_PATH', None)
        
    process = subprocess.Popen(cmd, env=clean_env)
    time.sleep(4)
    return process

def draw_results(frame: np.ndarray, results: list[dict[str, Any]], fps: float) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    for res in results:
        x1, y1, x2, y2 = res["bbox"]
        label = f"{res.get('char', '?')} ({res.get('dots', '')})"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + max(60, len(label) * 10), y1), (0, 255, 0), -1)
        
        cv2.putText(frame, label, (x1 + 2, max(15, y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    info_text = f"FPS: {fps:.1f} | Chars: {len(results)} | Res: {frame.shape[1]}x{frame.shape[0]}"
    cv2.putText(frame, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
    return frame

def main() -> None:
    print("Starting scrcpy stream...")
    scrcpy_proc = start_scrcpy_stream()
    
    cap = cv2.VideoCapture(f"/dev/video{CAMERA_INDEX}", cv2.CAP_V4L2)
    
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Critical error: Cannot open /dev/video{CAMERA_INDEX}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, HW_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HW_FRAME_HEIGHT)

        print("Loading AI Pipeline...")
        pipeline = EndToEndPipeline("configs/pipeline_config.yaml")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 540, 960)

        dummy_frame = np.zeros((HW_FRAME_WIDTH, HW_FRAME_HEIGHT, 3), dtype=np.uint8)
        pipeline.process_image(dummy_frame, translate=True)
        
        fps_buffer = deque(maxlen=10)
        prev_time = time.perf_counter()
        
        print("Stream ready. Press 'q' to exit.")
        while True:
            ret, hw_frame_bgr = cap.read()
            if not ret: 
                print("Frame dropped or stream ended.")
                break

            portrait_frame_bgr = cv2.rotate(hw_frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            inference_results = pipeline.process_image(portrait_frame_bgr, translate=True)
            
            curr_time = time.perf_counter()
            fps_buffer.append(1.0 / (curr_time - prev_time))
            prev_time = curr_time
            avg_fps = sum(fps_buffer) / len(fps_buffer)

            display_frame = draw_results(portrait_frame_bgr, inference_results, avg_fps)
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        if scrcpy_proc.poll() is None:
            scrcpy_proc.terminate()
            scrcpy_proc.wait()

if __name__ == "__main__":
    main()