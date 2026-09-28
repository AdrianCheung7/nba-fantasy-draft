"""Start the pick server (8502) and the Streamlit app (8501) without console windows.

python start.pyw               start whichever isn't already running
python start.pyw --install     run this at Windows login (Startup-folder entry)
python start.pyw --uninstall   remove that entry
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYW = Path(sys.executable).with_name("pythonw.exe")  # no console, even if launched from python.exe
SERVICES = {8502: ["server.py"],
            8501: ["-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]}


def in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def startup_entry():
    startup = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return startup / "Fantasy Draft App.vbs"


def main(argv):
    if "--install" in argv:
        # window style 0 = hidden
        startup_entry().write_text(f'CreateObject("WScript.Shell").Run """{PYW}"" ""{ROOT / "start.pyw"}""", 0, False\r\n')
    elif "--uninstall" in argv:
        startup_entry().unlink(missing_ok=True)
    else:
        for port, args in SERVICES.items():
            if not in_use(port):
                subprocess.Popen([str(PYW), *args], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)


if __name__ == "__main__":
    os.chdir(ROOT)
    main(sys.argv[1:])
