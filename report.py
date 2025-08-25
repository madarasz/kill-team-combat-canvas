import cv2
import json
import math
import time
from pathlib import Path
from board_detector import BoardCornerDetector


def calculate_corner_distance(detected_corner, expected_corner):
    """Calculate Euclidean distance between detected and expected corner."""
    x1, y1 = detected_corner
    x2, y2 = expected_corner
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_closest_corner_match(detected_corners, expected_corners, tolerance):
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
                
            distance = calculate_corner_distance(detected, expected)
            if distance < tolerance and distance < best_distance:
                best_distance = distance
                best_match = j
        
        if best_match is not None:
            matches.append((best_match, i, best_distance))
            used_detected.add(best_match)
            used_expected.add(i)
    
    return matches


def generate_report():
    """Generate a detailed report of corner detection accuracy for all test images."""
    # Load test data
    test_data_file = Path(__file__).parent / "tests" / "test-data" / "test_board_corners.json"
    
    try:
        with open(test_data_file, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find test data file at {test_data_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse test data file at {test_data_file}")
        return
    
    # Create detector with optimized parameters
    detector = BoardCornerDetector(
        canny_low=30,
        canny_high=90,
        hough_threshold=35,
        min_line_length=60,
        max_line_gap=30
    )
    
    print("=" * 80)
    print("KILL TEAM BOARD CORNER DETECTION ACCURACY REPORT")
    print("=" * 80)
    print()
    
    overall_stats = {
        'total_images': 0,
        'total_corners': 0,
        'successful_matches': 0,
        'distances': [],
        'within_tolerance': 0,
        'processing_times': []
    }
    
    for test_case in test_data["test_images"]:
        filename = test_case["filename"]
        expected_corners = test_case["expected_corners"]
        tolerance = test_case["tolerance"]
        description = test_case["description"]
        
        # Convert relative path to absolute path from project root
        image_path = Path(__file__).parent / filename
        
        print(f"Image: {filename}")
        print(f"Description: {description}")
        print(f"Tolerance: {tolerance} pixels")
        print("-" * 60)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"ERROR: Could not load image: {image_path}")
            print()
            continue
        
        # Detect corners with performance measurement
        start_time = time.time()
        detected_corners, _ = detector.detect_board_corners(image)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time*1000:.1f} ms")
        print()
        print("Corner Distance Analysis:")
        print(f"{'Expected':<20} {'Detected':<20} {'Distance':<12} {'Status'}")
        print(f"{'(x, y)':<20} {'(x, y)':<20} {'(pixels)':<12}")
        print("-" * 65)
        
        # Find matches
        matches = find_closest_corner_match(detected_corners, expected_corners, tolerance * 2)  # Use 2x tolerance for matching
        
        distances_this_image = []
        within_tolerance_count = 0
        
        for detected_idx, expected_idx, distance in matches:
            detected = detected_corners[detected_idx]
            expected = expected_corners[expected_idx]
            
            status = "✓ PASS" if distance <= tolerance else "✗ FAIL"
            if distance <= tolerance:
                within_tolerance_count += 1
            
            print(f"({expected[0]:4}, {expected[1]:4}){'':<8} "
                  f"({detected[0]:6.1f}, {detected[1]:6.1f}){'':<6} "
                  f"{distance:8.1f}{'':<4} {status}")
            
            distances_this_image.append(distance)
        
        # Handle unmatched corners
        matched_expected = {match[1] for match in matches}
        matched_detected = {match[0] for match in matches}
        
        for i, expected in enumerate(expected_corners):
            if i not in matched_expected:
                print(f"({expected[0]:4}, {expected[1]:4}){'':<8} "
                      f"{'NO MATCH':<20} {'N/A':<12} ✗ FAIL")
        
        for i, detected in enumerate(detected_corners):
            if i not in matched_detected:
                print(f"{'NO MATCH':<20} "
                      f"({detected[0]:6.1f}, {detected[1]:6.1f}){'':<6} "
                      f"{'N/A':<12} ✗ EXTRA")
        
        # Statistics for this image
        if distances_this_image:
            avg_distance = sum(distances_this_image) / len(distances_this_image)
            max_distance = max(distances_this_image)
            
            print()
            print(f"Image Statistics:")
            print(f"  Processing time: {processing_time*1000:.1f} ms")
            print(f"  Average distance: {avg_distance:.1f} pixels")
            print(f"  Maximum distance: {max_distance:.1f} pixels")
            print(f"  Within tolerance: {within_tolerance_count}/{len(expected_corners)} corners ({within_tolerance_count/len(expected_corners)*100:.1f}%)")
        else:
            print()
            print(f"Image Statistics:")
            print(f"  Processing time: {processing_time*1000:.1f} ms")
            print(f"  No successful corner matches found")
        
        # Update overall stats
        overall_stats['total_images'] += 1
        overall_stats['total_corners'] += len(expected_corners)
        overall_stats['successful_matches'] += len(matches)
        overall_stats['distances'].extend(distances_this_image)
        overall_stats['within_tolerance'] += within_tolerance_count
        overall_stats['processing_times'].append(processing_time)
        
        print()
        print("=" * 80)
        print()
    
    # Overall summary
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {overall_stats['total_images']}")
    print(f"Total corners within tolerance: {overall_stats['within_tolerance']}")
    
    if overall_stats['distances'] and overall_stats['processing_times']:
        avg_distance = sum(overall_stats['distances']) / len(overall_stats['distances'])
        max_distance = max(overall_stats['distances'])
        min_distance = min(overall_stats['distances'])
        avg_processing_time = sum(overall_stats['processing_times']) / len(overall_stats['processing_times'])
        max_processing_time = max(overall_stats['processing_times'])
        min_processing_time = min(overall_stats['processing_times'])
        
        print()
        print(f"Performance Statistics:")
        print(f"  Average processing time: {avg_processing_time*1000:.1f} ms")
        print(f"  Minimum processing time: {min_processing_time*1000:.1f} ms")
        print(f"  Maximum processing time: {max_processing_time*1000:.1f} ms")
        
        print()
        print(f"Distance Statistics:")
        print(f"  Average distance: {avg_distance:.1f} pixels")
        print(f"  Minimum distance: {min_distance:.1f} pixels")  
        print(f"  Maximum distance: {max_distance:.1f} pixels")
        print(f"  Match success rate: {overall_stats['successful_matches']/overall_stats['total_corners']*100:.1f}%")
        print(f"  Tolerance success rate: {overall_stats['within_tolerance']/overall_stats['total_corners']*100:.1f}%")
    else:
        print()
        print("No successful corner matches found across all images")