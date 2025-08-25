import unittest
import cv2
import json
import os
import math
from board_detector import BoardCornerDetector


class TestBoardCornerDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = BoardCornerDetector()
        self.test_data_file = "tests/test-data/test_board_corners.json"
        
    def load_test_data(self):
        """Load test data from JSON file."""
        with open(self.test_data_file, 'r') as f:
            return json.load(f)
    
    def calculate_corner_distance(self, detected_corner, expected_corner):
        """Calculate Euclidean distance between detected and expected corner."""
        x1, y1 = detected_corner
        x2, y2 = expected_corner
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def find_closest_corner_match(self, detected_corners, expected_corners, tolerance):
        """Find the best matching between detected and expected corners."""
        matches = []
        used_detected = set()
        used_expected = set()
        
        # For each expected corner, find the closest detected corner within tolerance
        for i, expected in enumerate(expected_corners):
            best_match = None
            best_distance = float('inf')
            
            for j, detected in enumerate(detected_corners):
                if j in used_detected:
                    continue
                    
                distance = self.calculate_corner_distance(detected, expected)
                if distance < tolerance and distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is not None:
                matches.append((best_match, i, best_distance))
                used_detected.add(best_match)
                used_expected.add(i)
        
        return matches
    
    def test_board_corner_detection(self):
        """Test board corner detection on test images."""
        test_data = self.load_test_data()
        
        for test_case in test_data["test_images"]:
            filename = test_case["filename"]
            expected_corners = test_case["expected_corners"]
            tolerance = test_case["tolerance"]
            description = test_case["description"]
            
            with self.subTest(filename=filename, description=description):
                # Load image
                image = cv2.imread(filename)
                self.assertIsNotNone(image, f"Could not load image: {filename}")
                
                # Detect corners
                detected_corners, _ = self.detector.detect_board_corners(image)
                
                # Check that we detected the expected number of corners
                self.assertEqual(len(detected_corners), len(expected_corners),
                               f"Expected {len(expected_corners)} corners, got {len(detected_corners)}")
                
                # Find matches between detected and expected corners
                matches = self.find_closest_corner_match(detected_corners, expected_corners, tolerance)
                
                # Check that all expected corners were matched
                self.assertEqual(len(matches), len(expected_corners),
                               f"Could only match {len(matches)} out of {len(expected_corners)} corners")
                
                # Print detailed results
                print(f"\nTest: {description}")
                print(f"Image: {filename}")
                print(f"Tolerance: {tolerance} pixels")
                print("Corner matching results:")
                for detected_idx, expected_idx, distance in matches:
                    detected = detected_corners[detected_idx]
                    expected = expected_corners[expected_idx]
                    print(f"  Expected ({expected[0]:4}, {expected[1]:4}) -> "
                          f"Detected ({detected[0]:6.1f}, {detected[1]:6.1f}) "
                          f"[Distance: {distance:.1f}px]")
                
                # Verify all matches are within tolerance
                for _, _, distance in matches:
                    self.assertLess(distance, tolerance,
                                  f"Corner distance {distance:.1f} exceeds tolerance {tolerance}")
    
    def test_board_corner_detection_5px_tolerance(self):
        """Test board corner detection with 5 pixel tolerance."""
        test_data = self.load_test_data()
        STRICT_TOLERANCE = 5.0  # 5 pixel tolerance for strict validation
        
        for test_case in test_data["test_images"]:
            filename = test_case["filename"]
            expected_corners = test_case["expected_corners"]
            description = test_case["description"]
            
            with self.subTest(filename=filename, description=f"{description} (5px tolerance)"):
                # Load image
                image = cv2.imread(filename)
                self.assertIsNotNone(image, f"Could not load image: {filename}")
                
                # Detect corners
                detected_corners, _ = self.detector.detect_board_corners(image)
                
                # Check that we detected the expected number of corners
                self.assertEqual(len(detected_corners), len(expected_corners),
                               f"Expected {len(expected_corners)} corners, got {len(detected_corners)}")
                
                # Find matches between detected and expected corners with 5px tolerance
                matches = self.find_closest_corner_match(detected_corners, expected_corners, STRICT_TOLERANCE)
                
                # Print detailed results for 5px tolerance test (before assertion)
                print(f"\nStrict Test (5px tolerance): {description}")
                print(f"Image: {filename}")
                print("Corner matching results:")
                
                # Calculate all distances for detailed reporting
                all_distances = []
                for i, expected in enumerate(expected_corners):
                    min_dist = float('inf')
                    closest_detected = None
                    closest_idx = None
                    for j, detected in enumerate(detected_corners):
                        dist = self.calculate_corner_distance(detected, expected)
                        if dist < min_dist:
                            min_dist = dist
                            closest_detected = detected
                            closest_idx = j
                    
                    status = "✓" if min_dist <= STRICT_TOLERANCE else "✗"
                    all_distances.append(min_dist)
                    print(f"  {status} Expected ({expected[0]:4}, {expected[1]:4}) -> "
                          f"Detected ({closest_detected[0]:6.1f}, {closest_detected[1]:6.1f}) "
                          f"[Distance: {min_dist:.1f}px]")
                
                passed_count = sum(1 for d in all_distances if d <= STRICT_TOLERANCE)
                print(f"Result: {passed_count}/{len(expected_corners)} corners within {STRICT_TOLERANCE}px tolerance")
                
                # Check that all expected corners were matched within 5px tolerance
                self.assertEqual(len(matches), len(expected_corners),
                               f"Could only match {len(matches)} out of {len(expected_corners)} corners within {STRICT_TOLERANCE}px tolerance")
                
                # Verify all matches are within strict 5px tolerance
                for _, _, distance in matches:
                    self.assertLessEqual(distance, STRICT_TOLERANCE,
                                       f"Corner distance {distance:.1f}px exceeds strict tolerance {STRICT_TOLERANCE}px")

    def test_corner_detection_robustness(self):
        """Test that corner detection is consistent across multiple runs."""
        test_data = self.load_test_data()
        
        if not test_data["test_images"]:
            self.skipTest("No test images available")
        
        filename = test_data["test_images"][0]["filename"]
        image = cv2.imread(filename)
        self.assertIsNotNone(image, f"Could not load image: {filename}")
        
        # Run detection multiple times
        results = []
        for i in range(3):
            corners, _ = self.detector.detect_board_corners(image)
            results.append(corners)
        
        # Check that results are consistent (within 1 pixel)
        for i in range(1, len(results)):
            self.assertEqual(len(results[0]), len(results[i]),
                           "Number of detected corners varies between runs")
            
            for j in range(len(results[0])):
                distance = self.calculate_corner_distance(results[0][j], results[i][j])
                self.assertLess(distance, 1.0,
                              f"Corner detection varies by {distance:.1f}px between runs")


if __name__ == "__main__":
    unittest.main(verbosity=2)