from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

last_input_time = time.time()
input_delay = 0.5

def detect_hand_gestures(frame):
    global last_input_time
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            if len(results.multi_hand_landmarks) == 1:
                current_time = time.time()
                
                if current_time - last_input_time >= input_delay:
                    last_input_time = current_time

                    landmarks = results.multi_hand_landmarks[0]
                    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    if thumb_tip.y < index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
                        pyautogui.press('right')
                        print("Right arrow key pressed")

                    elif thumb_tip.y > index_tip.y > middle_tip.y > ring_tip.y > pinky_tip.y:
                        pyautogui.press('left')
                        print("Left arrow key pressed")

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = detect_hand_gestures(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
