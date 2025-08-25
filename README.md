# Kill Team Buddy

A computer vision assistant for playing the Kill Team tabletop miniature game. Kill Team Buddy helps players by automatically detecting game board boundaries from camera feeds, enabling precise measurements and game state tracking.

## 🚀 Features

### Current Capabilities
- **Board Corner Detection**: Automatically identifies the four corners of the Kill Team game board from photographs

## 🛠 Installation

### Requirements
- Python 3.8+
- Camera or smartphone for board photography

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd kt-buddy

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Basic Board Detection

1. **Take a photo** of your Kill Team board from above (slight angles are fine)
2. **Save the image** in the `tests/test-data/` directory
3. **Run detection**:
   ```bash
   python3 board_detector.py
   ```
4. **View results** in the generated `board_detection_result.png`

### Accuracy Report

Generate detailed accuracy reports for all test images:

```bash
# Generate comprehensive accuracy report
python3 board_detector.py --report
```

This will analyze all test images and display:
- Distance between expected and detected corners
- Pass/fail status for each corner
- Per-image and overall statistics
- Success rates and accuracy metrics

### Example Output

The detection system will identify all four corners of your board and show:
- Original image
- Detected edges (Canny edge detection)
- Identified lines and corners with visual markers
- Detection statistics

### Adding Custom Test Images

To test with your own board images:

1. Add your image to `tests/test-data/`
2. Update the image path in `board_detector.py`:
   ```python
   image_path = "tests/test-data/your-image.jpg"
   ```
3. Run the detection script

## 🧪 Testing

Run the test suite to validate detection accuracy:

```bash
# Run all tests
python3 test_board_detector.py

# Verbose output
python3 test_board_detector.py -v
```

### Test Results
The current test suite achieves:
- ✅ All corners detected within 15-pixel tolerance
- ✅ Average error: 6.5 pixels
- ✅ Consistent results across multiple runs


## 🏗 Project Structure

```
kt-buddy/
├── board_detector.py           # Main detection algorithm
├── report.py                   # Accuracy reporting functionality
├── requirements.txt            # Python dependencies
├── tests/
│   ├── test_board_corner_detector.py  # Unit tests
│   ├── test-data/              # Test images
│   │   ├── board001.jpg
│   │   ├── board003.jpg
│   │   └── test_board_corners.json # Test data configuration
│   └── __init__.py
├── CLAUDE.md                   # Developer documentation
└── README.md                   # Project documentation
```

