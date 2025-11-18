import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ASL letters to collect
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'O', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Map letters to class numbers
letter_to_class = {letter: idx for idx, letter in enumerate(letters)}

dataset_size = 100  

cap = cv2.VideoCapture(0)

print("=" * 60)
print("ASL Letter Data Collection")
print("=" * 60)
print(f"Letters to collect: {', '.join(letters)}")
print(f"Images per letter: {dataset_size}")
print(f"Total images: {len(letters) * dataset_size}")
print("=" * 60)
print("\nInstructions:")
print("1. Show the ASL sign for the letter")
print("2. Press 'Q' when ready to start capturing")
print("3. Hold the sign steady while images are captured")
print("4. Press 'S' to skip a letter")
print("=" * 60)

for letter in letters:
    class_num = letter_to_class[letter]
    class_dir = os.path.join(DATA_DIR, str(class_num))
    
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f'\nCollecting data for letter: {letter} (Class {class_num})')
    print(f"   Show the ASL sign for letter '{letter}'")
    
    # Wait for user to be ready
    skip = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # Display letter information
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)
        cv2.putText(frame, f"Letter: {letter} (Class {class_num})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'Q' to start | 'S' to skip", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw a guide box
        h, w = frame.shape[:2]
        box_size = 300
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Position hand here", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Collect ASL Data', frame)
        
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            skip = True
            break
    
    if skip:
        print(f"    Skipped letter {letter}")
        continue
    
    # Collect images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # Progress bar
        progress = (counter / dataset_size) * 100
        bar_length = 400
        bar_filled = int((counter / dataset_size) * bar_length)
        
        # Display collection progress
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (50, 50, 50), -1)
        cv2.putText(frame, f"Collecting: {letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Progress: {counter}/{dataset_size} ({progress:.1f}%)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Progress bar
        cv2.rectangle(frame, (10, 70), (10 + bar_length, 90), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 70), (10 + bar_filled, 90), (0, 255, 0), -1)
        
        # Guide box
        h, w = frame.shape[:2]
        box_size = 300
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Collect ASL Data', frame)
        cv2.waitKey(10)  # Reduced delay for faster capture
        
        # Save image
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
    
    print(f"   Completed! Collected {dataset_size} images for '{letter}'")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(" Data collection complete!")
print(f" Data saved to: {DATA_DIR}")
print(f" Total classes: {len(letters)}")
print(f" Total images: {len(letters) * dataset_size}")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python createData_set.py")
print("2. Run: python train.py")
print("3. Run: python inference_word_builder.py")
print("=" * 60)