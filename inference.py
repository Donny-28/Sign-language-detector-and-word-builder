import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
import time
from config import (MODEL_PATH, LABELS_DICT, MIN_DETECTION_CONFIDENCE, 
                    MIN_TRACKING_CONFIDENCE, PREDICTION_BUFFER_SIZE,
                    CONFIDENCE_THRESHOLD, STABILITY_THRESHOLD,
                    COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_WHITE, COLOR_YELLOW)

# Load model
try:
    model_dict = pickle.load(open(MODEL_PATH, 'rb'))
    model = model_dict['model']
    print(f" Model loaded from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found. Please run train.py first.")
    exit(1)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit(1)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)


# Word builder variables
current_word = ""
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
last_letter_time = 0
letter_hold_time = 1.5  

print("=" * 50)
print("ASL Word Builder - Controls:")
print("=" * 50)
print("SPACE      - Add current letter to word")
print("TAB        - Add space between words")
print("ENTER      - Speak the word/sentence")
print("BACKSPACE  - Delete last character")
print("DELETE     - Clear entire word")
print("Q          - Quit")
print("=" * 50)

def get_stable_prediction(buffer):
    #Get most common prediction from buffer if consistent enough
    if len(buffer) < 5:
        return None, 0
    
    # Count occurrences
    from collections import Counter
    counts = Counter(buffer)
    most_common = counts.most_common(1)[0]
    letter, count = most_common
    
    # Check if prediction is stable 
    stability = (count / len(buffer)) * 100
    return letter, stability

def speak_word(word):
    #Convert word to speech
    if not word:
            print("No word to speak!")
            return

    print(f"Speaking: {word}")
    
    # Initialize the TTS engine 
    try:
        temp_engine = pyttsx3.init()
        temp_engine.setProperty('rate', 150)
        temp_engine.say(word)
        temp_engine.runAndWait()
        
        # to ensure it releases resources fully before the function exits.
        temp_engine.stop() 

    except Exception as e:
        print(f"TTS Error: {e}")

def draw_word_panel(frame, word, W, H):
    #Draw word building panel
    panel_height = 100
    cv2.rectangle(frame, (0, 0), (W, panel_height), (50, 50, 50), -1)
    
    # Display current word
    word_display = word if word else "[Empty]"
    cv2.putText(frame, f"Word: {word_display}", (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 2, cv2.LINE_AA)
    
    # Display letter count
    cv2.putText(frame, f"Letters: {len(word)}", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_controls(frame, W, H):
    """Draw control instructions"""
    instructions = [
        "SPACE: Add Letter | Underscore: Add Space | ENTER: Speak | BKSP: Delete | DEL: Clear | Q: Quit"
    ]
    
    y_pos = H - 15
    cv2.putText(frame, instructions[0], (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

while True:
    data_aux = []
    x_ = []
    y_ = []
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    H, W, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    
    current_prediction = None
    current_confidence = 0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        
        try:
            prediction = model.predict([np.asarray(data_aux)])
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(prediction_proba) * 100
            
            predicted_character = LABELS_DICT[int(prediction[0])]
            current_prediction = predicted_character
            current_confidence = confidence
            
            # Add to buffer if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_buffer.append(predicted_character)
            
            # Draw bounding box it is green if confident
            box_color = COLOR_GREEN if confidence >= CONFIDENCE_THRESHOLD else COLOR_YELLOW
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            # Display prediction
            text = f'{predicted_character} ({confidence:.1f}%)'
            cv2.putText(frame, text, (x1, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3, cv2.LINE_AA)
            
            # Show instruction to add letter
            if confidence >= CONFIDENCE_THRESHOLD:
                cv2.putText(frame, "Press SPACE to add", (x1, y2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        cv2.putText(frame, "Show hand sign", (W//2 - 150, H//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 2, cv2.LINE_AA)
        prediction_buffer.clear()
        current_prediction = None
    
    # Draw UI panels
    draw_word_panel(frame, current_word, W, H)
    draw_controls(frame, W, H)
    
    cv2.imshow('ASL Word Builder', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' '):  # SPACE to Add 
        if current_prediction and current_confidence >= CONFIDENCE_THRESHOLD:
            current_word += current_prediction
            print(f"Added: {current_prediction} | Word: {current_word}")
            prediction_buffer.clear()
        else:
            print("No confident prediction to add!")
    elif key == ord('_'):  # Underscore
        current_word += " "
        print(f"Added space | Word: '{current_word}'")
    elif key == 13:  # ENTER to Speak word
        if current_word:
            speak_word(current_word)
        else:
            print("No word to speak!")
    elif key == 8:  # BACKSPACE to Delete last letter
        if current_word:
            removed = current_word[-1]
            current_word = current_word[:-1]
            print(f"Removed: {removed} | Word: {current_word}")
    elif key == 127 or key == 46:  # DELETE key to Clear word
        if current_word:
            print(f"Cleared word: {current_word}")
            current_word = ""
    elif key == ord('s'):  # S to Save word to file
        if current_word:
            with open('asl_words.txt', 'a') as f:
                f.write(current_word + '\n')
            print(f"Saved: {current_word}")

cap.release()
cv2.destroyAllWindows()
print("Word Builder stopped.")