import argparse
import cv2
import sys
import os
import time
import yaml
from FrameBufferDisplay import FrameBufferDisplay
from OpticalFlow import OpticalFlow
from OpticalFlowSender import OpticalFlowSender

DEBUG_PRINT_INTERVAL_S = 0.25
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
            
def main():
    parser = argparse.ArgumentParser(description="Optical flow sender for FPV drop mission")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path (default: 0)")
    
    parser.add_argument("--partition", default="",
                        help="DDS partition (must match handler)")

    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain id (default: 0)")
    
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml next to this script)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    frame_width = config["frame_width"]
    frame_height = config["frame_height"]
    capture_fps = int(config.get("capture_fps", 30))
    capture_codec = str(config.get("capture_codec", "MJPG")).upper()
    capture_buffer_size = int(config.get("capture_buffer_size", 1))
    camera = args.device
    
    optical_flow_computatator = OpticalFlow(config)
    optical_flow_sender = OpticalFlowSender(args.partition, args.domain_id,
                                            on_activate=optical_flow_computatator.reset_state)
    frame_buffer_handler = FrameBufferDisplay(frame_width, frame_height)
    
    
    #Openning camera
    if isinstance(camera, int) or camera.isdigit():
        cap = cv2.VideoCapture(int(camera), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Failed to open camera device: {camera}")
        sys.exit(1)

    if capture_codec in ("MJPEG", "MJPG"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif capture_codec == "YUYV":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    cap.set(cv2.CAP_PROP_FPS, capture_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, capture_buffer_size)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    print(
        f"Camera opened: {camera}: ({frame_width}x{frame_height}), "
        f"codec={capture_codec}, fps={capture_fps}, buffer={capture_buffer_size}"
    )
    
    #Connecting messenger
    optical_flow_sender.connect_messenger()

    #Initializing FrameBufferDisplay
    frame_buffer_handler.initialize()
    
    last_result = None
    last_debug_print = 0.0
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            if optical_flow_sender.active:
                of_data = optical_flow_computatator.process_frame(frame)
                if of_data is not None:
                    optical_flow_sender.send_flow(*of_data)
                    last_result = of_data

                now = time.monotonic()
                if now - last_debug_print >= DEBUG_PRINT_INTERVAL_S:
                    if last_result is not None:
                        dx, dy, dt, q = last_result
                        print(f"[FLOW_TX] dx={dx:+7.3f}px dy={dy:+7.3f}px "
                              f"dt={dt*1000:6.2f}ms q={q:.2f} "
                              f"pts={optical_flow_computatator.tracked_count}")
                    else:
                        print(f"[FLOW_TX] no result yet "
                              f"pts={optical_flow_computatator.tracked_count}")
                    last_debug_print = now

                optical_flow_computatator.draw_overlay(
                    frame, last_result,
                    tracked_count=optical_flow_computatator.tracked_count)
            else:
                last_result = None
                cv2.putText(frame, "FLOW IDLE (waiting for FLOW_CONTROL)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4)
                cv2.putText(frame, "FLOW IDLE (waiting for FLOW_CONTROL)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 2)

            frame_buffer_handler.imshow(frame)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        frame_buffer_handler.close()


if __name__ == "__main__":
    main()
