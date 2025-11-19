# Sign-language-detector-and-word-builder
A real-time American Sign Language (ASL) letter recognition system that detects hand signs via webcam and builds words/sentences with text-to-speech functionality.

# Features
🎯 Real-time Detection

📝 Word Builder: Spell out complete words and sentences letter by letter

🔊 Text-to-Speech: Hear your spelled words spoken aloud

🎨 Visual Feedback: Confidence scores and color-coded indicators

💾 Save Words: Optional feature to save sentences to file

🖐️ Hand Tracking: Uses MediaPipe for robust hand landmark detection

🤖 Machine Learning: Random Forest classifier 

# How It Works
1. Data Collection (collect.py)
* Captures 100 images per letter (Can be adjusted)
* Uses webcam to record hand signs
* Stores images in organized class folders

2. Feature Extraction (createData_set.py)
* Uses MediaPipe Hands to detect hand landmarks per image
* Extracts (x, y) coordinates for each landmark 
* Saves features to data.pickle

3. Model Training (train.py)
* Trains Random Forest Classifier on extracted features
* 80/20 train-test split
* Outputs accuracy and classification report

4. Real-time Detection (inference.py)
* Captures webcam feed
* Detects hand landmarks with MediaPipe
* Extracts features
* Predicts letter using trained model
* Displays prediction with confidence score
* Builds words/sentences on demand

# Quick Start
The trained random forrest classifier is provided (model.p) and dataset is also provided (data.pickle), the model has been trained on my hand signs (excluding J, M, N, P, Q, R, Z) so the inference.py can be used directly. 
If you wish to train the model on your own hand sign signals replace the current model.p and data.pickle files by removing them and follow the steps below:
1. Collect Training Data ( run collect.py)
* Collect images for each ASL letter:
  * Press 'Q' when ready to start capturing
  * Press 'S' to skip a letter
  
2. Extract Hand Landmarks (run createData_set.py)
* Process images and extract MediaPipe hand features
* This creates data.pickle with normalized hand landmark coordinates.

3. Train the Model (run train.py)
* Train the Random Forest classifier

4. Start the Word Builder (run inference.py)
* Launch the real-time detection application

Adjust config.py to change the parameters/constants used in the program 

# Dependencies
* Opencv: 4.11.0
* Mediapipe: 0.10.21
* Pyttsx3: 2.99
* Scikit-learn: 1.7.2
