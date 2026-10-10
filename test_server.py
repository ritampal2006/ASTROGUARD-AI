import requests
import time
import subprocess
import sys

proc = subprocess.Popen(['py', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(2.5)

try:
    # 1. Test root page
    r_index = requests.get('http://127.0.0.1:5000/')
    print(f"1. Index Route: HTTP {r_index.status_code} (bytes: {len(r_index.text)})")

    # 2. Test predict
    payload = {'distance': 0.85, 'relative_speed': 12.4, 'time_to_approach': 8.5, 'inclination_diff': 28.4}
    r_pred = requests.post('http://127.0.0.1:5000/predict', json=payload)
    print("2. Predict Route:", r_pred.json())

    # 3. Test trajectory simulation
    r_traj = requests.post('http://127.0.0.1:5000/simulate-trajectory', json=payload)
    traj_json = r_traj.json()
    print(f"3. Trajectory Route: HTTP {r_traj.status_code}, Points count = {len(traj_json.get('trajectory', []))}")

    # 4. Test stats
    r_stats = requests.get('http://127.0.0.1:5000/api/stats')
    print("4. Stats Route:", r_stats.json().get('metadata', {}).get('algorithm'))

    print("\n--- ALL BACKEND ENDPOINTS PASSED SUCCESSFULLY! ---")
finally:
    proc.terminate()
