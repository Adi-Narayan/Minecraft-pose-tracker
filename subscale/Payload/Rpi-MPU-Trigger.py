import os
import subprocess
import time
from mpu6050 import mpu6050
import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_estimate_error=1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = initial_estimate_error
        
    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

# Initialize MPU6050 sensor
sensor = mpu6050(0x68)

# Define acceleration threshold (2G)
ACCEL_THRESHOLD = 2.0  

# Timer duration in seconds
TIMER_DURATION = 200

# Path to Hailo subprocess
hailo_path = "Hailo.py"  

# Initialize Kalman filters for each axis
# Tune these parameters based on your noise characteristics
process_variance = 0.01
measurement_variance = 0.1

kf_x = KalmanFilter(process_variance, measurement_variance)
kf_y = KalmanFilter(process_variance, measurement_variance)
kf_z = KalmanFilter(process_variance, measurement_variance)

triggered = False
start_time = time.time()

print("Starting MPU Monitoring with Kalman filtering...")
print(f"Timer set for {TIMER_DURATION} seconds")
time.sleep(1)  

while True:
    try:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= TIMER_DURATION:
            print("Timer expired! Stopping monitoring...")
            break
            
        accel_data = sensor.get_accel_data()
        
        # Apply Kalman filter to each axis
        filtered_x = kf_x.update(accel_data['x'])
        filtered_y = kf_y.update(accel_data['y'])
        filtered_z = kf_z.update(accel_data['z'])
        
        print(f"Raw Accel: X={accel_data['x']:.2f}, Y={accel_data['y']:.2f}, Z={accel_data['z']:.2f} G")
        print(f"Filtered Accel: X={filtered_x:.2f}, Y={filtered_y:.2f}, Z={filtered_z:.2f} G")
        print(f"Time remaining: {TIMER_DURATION - elapsed_time:.1f} seconds")
        
        # Check if filtered acceleration threshold is exceeded
        if (abs(filtered_x) > ACCEL_THRESHOLD or 
            abs(filtered_y) > ACCEL_THRESHOLD or 
            abs(filtered_z) > ACCEL_THRESHOLD) and not triggered:
            
            print("2G THRESHOLD EXCEEDED! Launching Hailo...")
            try:
                subprocess.run([hailo_path], check=True)
                triggered = True  # Prevent multiple triggers
                print(f"Triggered set to: {triggered}")
            except subprocess.CalledProcessError as e:
                print(f"Error launching Hailo: {e}")
            except FileNotFoundError:
                print("Error: Hailo executable not found")
        else:
            print("Launch conditions not met...")
            
        time.sleep(0.5)  # Delay to prevent excessive CPU usage
        
    except Exception as e:
        print(f"Error reading sensor data: {e}")
        time.sleep(1)  # Wait before retrying in case of failure

print("Capture signaling complete!")