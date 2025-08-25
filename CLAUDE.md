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

# Generate accuracy report for all test images
python3 board_detector.py --report

# Show help and available options
python3 board_detector.py --help

# Single image mode will:
# - Load the specified image (or tests/test-data/board001.jpg by default)
# - Detect the 4 board corners
# - Save visualization to specified output file (or board_detection_result.png by default)

# Report mode will:
# - Load all test images from tests/test-data/test_board_corners.json
# - Run corner detection on each image
# - Calculate distances between expected and detected corners
# - Display detailed accuracy report in terminal with statistics
```

### Testing
```bash
# Run all unit tests
python3 -m unittest discover tests/ -v

# Run all tests in the corner detection test file
python3 -m unittest tests.test_board_corner_detector -v

# Run specific test methods
python3 -m unittest tests.test_board_corner_detector.TestBoardCornerDetector.test_board_corner_detection -v
python3 -m unittest tests.test_board_corner_detector.TestBoardCornerDetector.test_board_corner_detection_5px_tolerance -v
python3 -m unittest tests.test_board_corner_detector.TestBoardCornerDetector.test_corner_detection_robustness -v

# Run with pytest (alternative test runner)
python3 -m pytest tests/ -v
python3 -m pytest tests/test_board_corner_detector.py -v
```

## Code Architecture

### Core Detection System
The `BoardCornerDetector` class in `board_detector.py` implements a computer vision pipeline:

1. **Preprocessing**: Image grayscale conversion and Gaussian blur
2. **Edge Detection**: Canny edge detection to find board boundaries
3. **Line Detection**: Hough Line Transform to identify straight board edges
4. **Corner Extraction**: Line intersection calculation to locate corners
5. **Filtering**: Geometric validation and rectangle corner selection

Key configurable parameters (optimized for 5px accuracy):
- `canny_low/canny_high`: Edge detection thresholds (30/90, optimized from 30/100)
- `hough_threshold`: Line detection sensitivity (40, optimized from 60)
- `min_line_length/max_line_gap`: Line filtering constraints (60/20, optimized from 80/30)

### Report System
The `report.py` module provides accuracy analysis functionality:

1. **Distance Calculation**: Euclidean distance between expected and detected corners
2. **Corner Matching**: Finds closest detected corners to expected positions within tolerance
3. **Statistical Analysis**: Per-image and overall statistics including success rates
4. **Formatted Output**: Clean terminal formatting with pass/fail indicators

Key functions:
- `calculate_corner_distance()`: Computes distance between corner pairs
- `find_closest_corner_match()`: Matches detected corners to expected positions
- `generate_report()`: Main report generation function with comprehensive analysis

### Test Framework Structure
The testing system uses a data-driven approach with organized test structure:

- `tests/test-data/test_board_corners.json`: Contains test images with expected corner coordinates and tolerance values
- `tests/test_board_corner_detector.py`: Unit test suite with corner matching logic that finds closest detected corners to expected positions within tolerance
- `tests/__init__.py`: Makes tests directory a proper Python package

Test categories:
- **General Detection Test**: 15px tolerance for overall functionality validation
- **Strict 5px Tolerance Test**: Validates optimized parameter precision
- **Robustness Test**: Ensures consistent detection across multiple runs
- **Initialization Test**: Validates detector parameter setup

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