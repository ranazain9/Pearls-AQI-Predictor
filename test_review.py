import requests, json

r = requests.get('http://127.0.0.1:5000/api/data', timeout=10)
d = r.json()
print('Status:', r.status_code)
print('Current AQI:', d['current']['aqi'], '|', d['current']['category'])
print('Forecast:')
for i, f in enumerate(d['forecast']):
    print(f"  {f['horizon']} (Day {i+1}): AQI={f['predicted_aqi']} | RMSE±{f['rmse_aqi']}")
print('Hourly rows:', len(d['forecast_hourly']))
print('RMSE by horizon (AQI):', d.get('rmse_by_horizon_aqi'))
