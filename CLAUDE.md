# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kill Team Buddy is a computer vision tool designed to assist with playing the Kill Team board game. The tool processes camera feeds of the 22"x30" cardboard game board to detect board boundaries, make measurements, and track game state. Currently implements board corner detection as the foundational computer vision capability.

## Commands

### Dependencies and Setup
```bash
# Install Python dependencies
python3 -m pip install -r requirements.txt

# Required packages: opencv-python, numpy, matplotlib
```

### Running Detection
```bash
# Run board corner detection on default test image
python3 board_detector.py

# Run board corner detection on specific image
python3 board_detector.py path/to/your/image.jpg

# Run with custom output file
python3 board_detector.py path/to/image.jpg --output custom_result.png

# Show help and available options
python3 board_detector.py --help

# This will:
# - Load the specified image (or tests/test-data/board001.jpg by default)
# - Detect the 4 board corners
# - Save visualization to specified output file (or board_detection_result.png by default)
```

### Testing
```bash
# Run all unit tests
python3 test_board_detector.py

# Run with verbose output
python3 test_board_detector.py -v

# Run specific test method
python3 -m unittest test_board_detector.TestBoardCornerDetector.test_board_corner_detection
```

## Code Architecture

### Core Detection System
The `BoardCornerDetector` class in `board_detector.py` implements a computer vision pipeline:

1. **Preprocessing**: Image grayscale conversion and Gaussian blur
2. **Edge Detection**: Canny edge detection to find board boundaries
3. **Line Detection**: Hough Line Transform to identify straight board edges
4. **Corner Extraction**: Line intersection calculation to locate corners
5. **Filtering**: Geometric validation and rectangle corner selection

Key configurable parameters:
- `canny_low/canny_high`: Edge detection thresholds
- `hough_threshold`: Line detection sensitivity  
- `min_line_length/max_line_gap`: Line filtering constraints

### Test Framework Structure
The testing system uses a data-driven approach:

- `tests/test_board_corners.json`: Contains test images with expected corner coordinates and tolerance values
- `test_board_detector.py`: Implements corner matching logic that finds closest detected corners to expected positions within tolerance
- Tests validate both accuracy (corner positions) and robustness (consistency across runs)

### Test Data Format
```json
{
  "test_images": [
    {
      "filename": "tests/test-data/board001.jpg",
      "expected_corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "tolerance": 15,
      "description": "Description of test case"
    }
  ]
}
```

## Key Implementation Details

### Corner Detection Algorithm
The system detects rectangular board corners by:
1. Filtering detected lines into horizontal/vertical categories based on angle
2. Selecting longest lines in each category (typically 4-6 lines each)
3. Computing intersections between horizontal and vertical lines
4. Removing duplicate corners within 30-pixel radius
5. If >4 corners found, selecting 4 extreme points (top-left, top-right, bottom-left, bottom-right)

### Computer Vision Challenges Addressed
- **Perspective distortion**: Board photographed at angles
- **Noise filtering**: Miniatures and terrain pieces on board surface
- **Edge quality**: Sandy texture and lighting variations
- **Robustness**: Consistent detection across multiple runs

### Adding New Test Cases
To add new board images for testing:
1. Place image in `tests/test-data/`
2. Manually identify corner coordinates
3. Add entry to `tests/test_board_corners.json` with appropriate tolerance
4. Run tests to validate detection accuracy