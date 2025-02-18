#include <Wire.h>
#include <MPU6050.h>
#include <ESP32Servo.h>

// Kalman filter parameters
const float Q_ANGLE = 0.001;   // Process noise variance for angle
const float Q_BIAS = 0.003;    // Process noise variance for bias
const float R_MEASURE = 0.03;  // Measurement noise variance

// Kalman filter state variables
float angle = 0;
float bias = 0;
float rate = 0;
float P[2][2] = {{0, 0}, {0, 0}};

const int SERVO_PIN = 13;  // Change to your servo pin
const float ACCEL_THRESHOLD = 1.5;  // 1.5G threshold
const int ROTATION_DURATION = 200000;  // 200 seconds in milliseconds
const int ROTATIONS_PER_SECOND = 3;
const int ROTATION_STEP = 6;  // Degrees to step each update (adjust for smooth rotation)

MPU6050 mpu;
Servo myservo;

bool isRotating = false;
unsigned long startTime = 0;
int currentAngle = 0;

// Kalman filter implementation
float kalmanFilter(float newValue) {
    // Prediction step
    float P00_temp = P[0][0];
    float P01_temp = P[0][1];
    
    P[0][0] += Q_ANGLE - (P[0][1] + P[1][0]) + P[1][1];
    P[0][1] -= P[1][1];
    P[1][0] -= P[1][1];
    P[1][1] += Q_BIAS;
    
    // Measurement update
    float S = P[0][0] + R_MEASURE;
    float K[2] = {P[0][0] / S, P[1][0] / S};
    
    float y = newValue - angle;
    
    angle += K[0] * y;
    bias += K[1] * y;
    
    P[0][0] -= K[0] * P[0][0];
    P[0][1] -= K[0] * P[0][1];
    P[1][0] -= K[1] * P[0][0];
    P[1][1] -= K[1] * P[0][1];
    
    return angle;
}

void setup() {
    Serial.begin(115200);
    
    Wire.begin();
    mpu.initialize();
    
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);
    }
    
    // Initialize servo
    ESP32PWM::allocateTimer(0);
    myservo.setPeriodHertz(333);  // 333Hz servo frequency for KST Servo DS215MG V8.0
    myservo.attach(SERVO_PIN);
    myservo.write(0);
    
    Serial.println("Setup complete");
}

void loop() {
    // Read acceleration data
    int16_t ax, ay, az;
    mpu.getAcceleration(&ax, &ay, &az);
    
    // Convert raw acceleration to G's
    float accelX = ax / 16384.0;  // ±2G range
    float accelY = ay / 16384.0;
    float accelZ = az / 16384.0;
    
    // Calculate total acceleration magnitude
    float totalAccel = sqrt(accelX*accelX + accelY*accelY + accelZ*accelZ);
    
    // Apply Kalman filter to smooth the acceleration reading
    float filteredAccel = kalmanFilter(totalAccel);
    
    // Check if filtered acceleration threshold is exceeded and rotation isn't already happening
    if (filteredAccel > ACCEL_THRESHOLD && !isRotating) {
        Serial.println("Acceleration trigger detected! Starting rotation sequence");
        Serial.print("Filtered acceleration: ");
        Serial.println(filteredAccel);
        isRotating = true;
        startTime = millis();
    }
    
    // Handle rotation sequence
    if (isRotating) {
        unsigned long currentTime = millis();
        
        // Check if rotation duration is complete
        if (currentTime - startTime >= ROTATION_DURATION) {
            isRotating = false;
            myservo.write(0);
            Serial.println("Rotation sequence complete");
            return;
        }
        
        // Calculate delay needed for desired rotation speed
        int delayTime = (1000 / (ROTATIONS_PER_SECOND * (360 / ROTATION_STEP)));
        
        // Update servo position
        currentAngle = (currentAngle + ROTATION_STEP) % 360;
        myservo.write(currentAngle);
        delay(delayTime);
    }
}