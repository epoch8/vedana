import signal
import subprocess
import sys
from pathlib import Path


def shutdown(signum, frame):
    """Handle shutdown gracefully"""
    print("Shutting down services...")
    sys.exit(0)


def main():
    """Start jims backoffice services"""
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    caddyfile_path = Path(__file__).parent / "Caddyfile"

    print("Starting Caddy...")
    caddy = subprocess.Popen(["caddy", "run", "--config", str(caddyfile_path)])

    print("Starting backend application...")
    backend = subprocess.Popen(["reflex", "run", "--env", "prod", "--backend-only"])

    print(f"Services started. PIDs: Caddy={caddy.pid}, Backend={backend.pid}")
    print("Waiting for services...")

    try:
        backend.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
