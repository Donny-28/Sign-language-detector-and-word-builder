import os
import pickle
import mediapipe as mp
import cv2
from config import DATA_DIR, DATA_PICKLE_PATH, ASL_LETTERS

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []
skipped_images = 0

print("=" * 60)
print("Extracting hand landmarks from images...")
print("=" * 60)

for dir_ in sorted(os.listdir(DATA_DIR)):
    # Skip non-directory files
    if not os.path.isdir(os.path.join(DATA_DIR, dir_)):
        continue
    
    dir_path = os.path.join(DATA_DIR, dir_)
    images = os.listdir(dir_path)
    
    print(f"\nProcessing class {dir_} ({len(images)} images)...")
    processed = 0
    
    for img_path in images:
        data_aux = []
        x_ = []
        y_ = []
        
        img = cv2.imread(os.path.join(dir_path, img_path))
        
        if img is None:
            print(f"   Could not read: {img_path}")
            skipped_images += 1
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
            
            data.append(data_aux)
            labels.append(dir_)
            processed += 1
        else:
            skipped_images += 1
    
    print(f"  Processed {processed}/{len(images)} images")

print("\n" + "=" * 60)
print("Dataset Statistics:")
print("=" * 60)
print(f"Total samples collected: {len(data)}")
print(f"Feature vector length: {len(data[0]) if data else 0}")
print(f"Skipped images (no hand detected): {skipped_images}")
print(f"Success rate: {(len(data)/(len(data)+skipped_images)*100):.1f}%")
print("=" * 60)

# Save dataset
f = open(DATA_PICKLE_PATH, 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"\n Dataset saved to: {DATA_PICKLE_PATH}")
print("Next step: Run 'python train.py' to train the model")
print("=" * 60)