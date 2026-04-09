from messenger_async import DDSMessenger
import json

class OpticalFlowSender:
    def __init__(self, domain_id: int = 0, on_activate=None):
        self._domain_id = domain_id
        self._messenger_ready = False
        self.active = False
        self._on_activate = on_activate
        self._messenger = DDSMessenger(domain_id=domain_id)

    def connect_messenger(self):
        try:
            self._messenger.init()
            self._messenger.subscribe("FLOW_CONTROL", self._on_flow_control)
            self._messenger_ready = True
            print(f"DDS messenger initialized (domain_id={self._domain_id})")
        except Exception as e:
            self._messenger_ready = False
            print(f"Failed to initialize DDS messenger: {e}")
            print("Running without publisher connection (will skip sends).")

    def _on_flow_control(self, message):
        try:
            data = json.loads(message.data()) if hasattr(message, 'data') else json.loads(message)
            was_active = self.active
            self.active = bool(data.get("active", False))
            if self.active != was_active:
                state = "ACTIVE" if self.active else "IDLE"
                print(f"Flow control: {state}")
                if self.active and self._on_activate:
                    self._on_activate()
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Bad FLOW_CONTROL message: {e}")
    
    def send_flow(self, dx, dy, dt, quality):
        payload = json.dumps({
            "dx": round(dx, 4),
            "dy": round(dy, 4),
            "dt": round(dt, 6),
            "quality": round(quality, 3),
        })
        if self._messenger_ready:
            try:
                #topic-based publish with payload only
                self._messenger.send("FLOW_DATA", payload)
            except Exception as e:
                print(f"Send error: {e}")

