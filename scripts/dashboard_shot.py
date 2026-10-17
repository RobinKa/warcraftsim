"""A dashboard screenshot through headless Firefox + geckodriver (node is not installed here: scripts/dashboard_shot.mjs needs it).

    python3 scripts/dashboard_shot.py URL OUT.png [--wait SELECTOR] [--js "code run after the wait"] [--width W --height H]
"""
import argparse
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--wait", default="body")
ap.add_argument("--js", default="")
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=1000)
ap.add_argument("--port", type=int, default=4471)
a = ap.parse_args()

drv = subprocess.Popen(["/snap/bin/geckodriver", "--port", str(a.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
base = f"http://127.0.0.1:{a.port}"


def call(method, path, body=None):
    req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["value"]
    except urllib.error.HTTPError as e:
        print("webdriver error:", e.read()[:600])
        raise


try:
    for _ in range(50):
        try:
            call("GET", "/status")
            break
        except Exception:
            time.sleep(0.2)
    sid = call("POST", "/session", {"capabilities": {"alwaysMatch": {"moz:firefoxOptions": {
        "args": ["-headless", f"--width={a.width}", f"--height={a.height}"]}}}})["sessionId"]
    try:
        call("POST", f"/session/{sid}/window/rect", {"width": a.width, "height": a.height})
        call("POST", f"/session/{sid}/url", {"url": a.url})
        for _ in range(120):
            if call("POST", f"/session/{sid}/execute/sync", {"script": f"return !!document.querySelector({json.dumps(a.wait)})", "args": []}):
                break
            time.sleep(0.5)
        time.sleep(1.5)
        if a.js:
            print("js:", call("POST", f"/session/{sid}/execute/sync", {"script": a.js, "args": []}))
            time.sleep(4)
        errs = call("POST", f"/session/{sid}/execute/sync", {"script": "return [document.body.dataset.dbg, document.body.dataset.dbg2]", "args": []})
        if errs:
            print("errors:", errs)
        png = call("GET", f"/session/{sid}/screenshot")
        open(a.out, "wb").write(base64.b64decode(png))
        print("wrote", a.out)
    finally:
        call("DELETE", f"/session/{sid}")
finally:
    drv.terminate()
