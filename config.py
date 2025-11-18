# config.py - Centralized configuration for ASL Detection

# ASL letters to use (excluding J and Z which require motion)
ASL_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'O', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Create labels dictionary
LABELS_DICT = {idx: letter for idx, letter in enumerate(ASL_LETTERS)}

# Reverse mapping for encoding
LETTER_TO_CLASS = {letter: idx for idx, letter in enumerate(ASL_LETTERS)}

# Data collection settings
DATA_DIR = './data'
DATASET_SIZE = 100  # Images per class

# Model settings
MODEL_PATH = 'model.p'
DATA_PICKLE_PATH = 'data.pickle'

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Word builder settings
PREDICTION_BUFFER_SIZE = 10
CONFIDENCE_THRESHOLD = 20
STABILITY_THRESHOLD = 70

# Display settings
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GRAY = (128, 128, 128)

# Print configuration
def print_config():
    print("=" * 60)
    print("ASL Detection System Configuration")
    print("=" * 60)
    print(f"Letters: {', '.join(ASL_LETTERS)}")
    print(f"Total classes: {len(ASL_LETTERS)}")
    print(f"Images per class: {DATASET_SIZE}")
    print(f"Total images needed: {len(ASL_LETTERS) * DATASET_SIZE}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()