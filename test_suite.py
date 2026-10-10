import json
from app import app

client = app.test_client()

print("--- Testing AstroGuard AI Flask Application In-Process ---")

# 1. Test GET /
r_index = client.get('/')
print(f"1. GET / -> Status: {r_index.status_code}, Length: {len(r_index.data)} bytes")
assert r_index.status_code == 200

# 2. Test POST /predict with High Risk scenario
payload_high = {
    'distance': 0.85,
    'relative_speed': 12.4,
    'time_to_approach': 8.5,
    'inclination_diff': 28.4
}
r_pred_high = client.post('/predict', json=payload_high)
data_high = r_pred_high.get_json()
print(f"2. POST /predict (Critical Encounter) -> Status: {r_pred_high.status_code}")
print(f"   Risk Level: {data_high['risk_level']} (Code: {data_high['risk_code']})")
print(f"   Probabilities: {data_high['probabilities']}")
print(f"   Action Protocol: {data_high['protocol']['title']}")
print(f"   Delta-V Burn Est: {data_high['telemetry']['estimated_delta_v_ms']} m/s")
assert r_pred_high.status_code == 200
assert data_high['risk_level'] == 'High'

# 3. Test POST /predict with Nominal Safe scenario
payload_low = {
    'distance': 42.0,
    'relative_speed': 2.1,
    'time_to_approach': 95.0,
    'inclination_diff': 2.1
}
r_pred_low = client.post('/predict', json=payload_low)
data_low = r_pred_low.get_json()
print(f"\n3. POST /predict (Safe Encounter) -> Status: {r_pred_low.status_code}")
print(f"   Risk Level: {data_low['risk_level']} (Code: {data_low['risk_code']})")
print(f"   Probabilities: {data_low['probabilities']}")
assert r_pred_low.status_code == 200
assert data_low['risk_level'] == 'Low'

# 4. Test POST /simulate-trajectory
r_traj = client.post('/simulate-trajectory', json=payload_high)
data_traj = r_traj.get_json()
print(f"\n4. POST /simulate-trajectory -> Status: {r_traj.status_code}")
print(f"   Points returned: {len(data_traj['trajectory'])}")
print(f"   Step 1 Dist: {data_traj['trajectory'][0]['distance_km']} km -> Step 10 Dist: {data_traj['trajectory'][9]['distance_km']} km")
assert r_traj.status_code == 200
assert len(data_traj['trajectory']) == 10

# 5. Test GET /api/stats
r_stats = client.get('/api/stats')
data_stats = r_stats.get_json()
print(f"\n5. GET /api/stats -> Status: {r_stats.status_code}")
print(f"   Model Algorithm: {data_stats['metadata']['algorithm']}")
print(f"   Accuracy: {data_stats['metadata']['metrics']['accuracy']}%")
assert r_stats.status_code == 200

# 6. Test GET /api/sample-scenarios
r_scen = client.get('/api/sample-scenarios')
data_scen = r_scen.get_json()
print(f"\n6. GET /api/sample-scenarios -> Status: {r_scen.status_code}")
print(f"   Presets count: {len(data_scen['scenarios'])}")
assert r_scen.status_code == 200

print("\n=======================================================")
print("  ALL ASTROGUARD AI TESTS PASSED WITH 100% SUCCESS!  ")
print("=======================================================")
