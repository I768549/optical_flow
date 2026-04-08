import argparse
import time
import json
import sys
from messenger_async import DDSMessenger


FAKE_FLOW = {
    "dx": 1.2345,
    "dy": -0.6789,
    "dt": 0.033333,
    "quality": 0.95,
}


def main():
    parser = argparse.ArgumentParser(description="Test optical flow sender (fake data)")
    parser.add_argument("--device", default="0",
                        help="Camera device index or path (ignored, kept for compat)")

    parser.add_argument("--partition", default="", help="DDS partition (must match handler)")

    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain id (default: 0)")

    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (ignored, kept for compat)")

    parser.add_argument("--interval", type=float, default=0.1,
                        help="Send interval in seconds (default: 0.1)")
    args = parser.parse_args()

    active = False

    def on_flow_control(message):
        nonlocal active
        try:
            data = json.loads(message.data()) if hasattr(message, 'data') else json.loads(message)
            was_active = active
            active = bool(data.get("active", False))
            if active != was_active:
                state = "ACTIVE" if active else "IDLE"
                print(f"Flow control: {state}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Bad FLOW_CONTROL message: {e}")

    messenger = DDSMessenger(args.partition, domain_id=args.domain_id)
    try:
        messenger.init()
        messenger.subscribe("FLOW_CONTROL", on_flow_control)
    except Exception as e:
        print(f"Failed to initialize DDS messenger: {e}")
        sys.exit(1)

    print(f"DDS messenger initialized (domain_id={args.domain_id})")
    print(f"Waiting for FLOW_CONTROL active=true. Interval={args.interval}s. Ctrl+C to stop.")

    try:
        seq = 0
        while True:
            if active:
                payload = json.dumps(FAKE_FLOW)
                messenger.send("FLOW_DATA", payload)
                seq += 1
                print(f"[{seq}] sent: {payload}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        messenger.shutdown()


if __name__ == "__main__":
    main()
