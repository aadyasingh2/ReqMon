"""Script to launch live FastAPI backend server and test cross-origin fetch requests with Origin headers."""

import os
import sys
import time
import requests
import uvicorn
import threading

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ramma_backend.main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # Start uvicorn server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)  # Wait for server startup

    base_url = "http://127.0.0.1:8000"
    origin = "http://localhost:5173"

    print("============================================================")
    print("        RAMMA FASTAPI CORS CROSS-ORIGIN VERIFICATION        ")
    print("============================================================")

    # 1. Test POST /interpret with Origin header
    print(f"\n[TEST 1] Simulated browser fetch() POST {base_url}/interpret")
    print(f"         Request Header -> Origin: {origin}")
    res_interpret = requests.post(
        f"{base_url}/interpret",
        json={"text": "Recall must remain above 93%"},
        headers={"Origin": origin},
    )
    allow_origin_1 = res_interpret.headers.get("Access-Control-Allow-Origin")
    print(f"         HTTP Status: {res_interpret.status_code}")
    print(f"         Response Header -> Access-Control-Allow-Origin: {allow_origin_1}")
    print(f"         Response JSON: {res_interpret.json()}")
    assert res_interpret.status_code == 200
    assert allow_origin_1 == origin, f"Expected {origin}, got {allow_origin_1}"

    # 2. Test GET /requirements with Origin header
    print(f"\n[TEST 2] Simulated browser fetch() GET {base_url}/requirements")
    print(f"         Request Header -> Origin: {origin}")
    res_reqs = requests.get(f"{base_url}/requirements", headers={"Origin": origin})
    allow_origin_2 = res_reqs.headers.get("Access-Control-Allow-Origin")
    print(f"         HTTP Status: {res_reqs.status_code}")
    print(f"         Response Header -> Access-Control-Allow-Origin: {allow_origin_2}")
    print(f"         Response JSON Count: {len(res_reqs.json())}")
    assert res_reqs.status_code == 200
    assert allow_origin_2 == origin, f"Expected {origin}, got {allow_origin_2}"

    # 3. Test OPTIONS preflight request for /requirements
    print(f"\n[TEST 3] Simulated browser OPTIONS preflight request to {base_url}/requirements")
    print(f"         Request Header -> Origin: {origin}")
    print("         Request Header -> Access-Control-Request-Method: POST")
    res_preflight = requests.options(
        f"{base_url}/requirements",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    allow_origin_3 = res_preflight.headers.get("Access-Control-Allow-Origin")
    allow_methods = res_preflight.headers.get("Access-Control-Allow-Methods")
    print(f"         HTTP Status: {res_preflight.status_code}")
    print(f"         Response Header -> Access-Control-Allow-Origin: {allow_origin_3}")
    print(f"         Response Header -> Access-Control-Allow-Methods: {allow_methods}")
    assert res_preflight.status_code == 200
    assert allow_origin_3 == origin

    print("\n============================================================")
    print("SUCCESS: All CORS cross-origin requests verified cleanly!")
    print("============================================================")
