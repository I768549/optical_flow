import argparse
import cv2
import sys
import os
import yaml
from .FrameBufferDisplay import FrameBufferDisplay
from .OpticalFlow import OpticalFlow
from .OpticalFlowSender import OpticalFlowSender
def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
            
def main():
    parser = argparse.ArgumentParser(description="Optical flow sender for FPV drop mission")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path (default: 0)")
    
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain id (default: 0)")
    
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: config.yaml next to this script)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    frame_width = config["frame_width"]
    frame_height = config["frame_height"]
    camera = args.device
    
    optical_flow_computatator = OpticalFlow(config)
    optical_flow_sender = OpticalFlowSender(args.domain_id,
                                            on_activate=optical_flow_computatator.reset_state)
    frame_buffer_handler = FrameBufferDisplay(frame_width, frame_height)
    
    
    #Openning camera
    if isinstance(camera, int) or camera.isdigit():
        cap = cv2.VideoCapture(int(camera))
    else:
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Failed to open camera device: {camera}")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    print(f"Camera opened: {camera}: ({frame_width}x{frame_height})")
    
    #Connecting messenger
    optical_flow_sender.connect_messenger()

    #Initializing FrameBufferDisplay
    frame_buffer_handler.initialize()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame_buffer_handler.imshow(frame)

            if optical_flow_sender.active:
                of_data = optical_flow_computatator.process_frame(frame)
                if of_data is not None:
                    optical_flow_sender.send_flow(*of_data)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        frame_buffer_handler.close()


if __name__ == "__main__":
    main()
