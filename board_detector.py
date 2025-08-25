import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math
import argparse


class BoardCornerDetector:
    def __init__(self, 
                 canny_low: int = 30,
                 canny_high: int = 90,
                 hough_threshold: int = 35,
                 min_line_length: int = 60,
                 max_line_gap: int = 30):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and apply Gaussian blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        edges = cv2.Canny(gray_image, self.canny_low, self.canny_high)
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lines using Hough Line Transform."""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=self.hough_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        if lines is None:
            return []
        
        return [tuple(line[0]) for line in lines]
    
    def line_angle(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate angle of a line in degrees."""
        x1, y1, x2, y2 = line
        return math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    def line_length(self, line: Tuple[int, int, int, int]) -> float:
        """Calculate length of a line."""
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def cluster_parallel_lines(self, lines: List[Tuple[int, int, int, int]], 
                              distance_threshold: float = 25, 
                              angle_threshold: float = 8) -> List[Tuple[int, int, int, int]]:
        """Cluster parallel lines that are close together and return representative lines."""
        if not lines:
            return []
        
        # Group lines by similarity
        clusters = []
        used = [False] * len(lines)
        
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
                
            # Start a new cluster with this line
            cluster = [line1]
            used[i] = True
            angle1 = abs(self.line_angle(line1))
            
            # Find similar lines to add to this cluster
            for j, line2 in enumerate(lines):
                if used[j] or i == j:
                    continue
                    
                angle2 = abs(self.line_angle(line2))
                angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
                
                if angle_diff < angle_threshold:
                    # Check distance between lines
                    dist = self.line_to_line_distance(line1, line2)
                    if dist < distance_threshold:
                        cluster.append(line2)
                        used[j] = True
            
            clusters.append(cluster)
        
        # Create representative lines for each cluster
        representative_lines = []
        for cluster in clusters:
            if len(cluster) == 1:
                representative_lines.append(cluster[0])
            else:
                # Create a representative line by averaging cluster endpoints
                # This gives better results than just picking the longest line
                total_length = sum(self.line_length(line) for line in cluster)
                
                # Weight by line length for better averaging
                weighted_x1 = weighted_y1 = weighted_x2 = weighted_y2 = 0
                for line in cluster:
                    weight = self.line_length(line) / total_length
                    x1, y1, x2, y2 = line
                    weighted_x1 += x1 * weight
                    weighted_y1 += y1 * weight
                    weighted_x2 += x2 * weight
                    weighted_y2 += y2 * weight
                
                avg_line = (int(weighted_x1), int(weighted_y1), int(weighted_x2), int(weighted_y2))
                representative_lines.append(avg_line)
        
        return representative_lines

    def line_to_line_distance(self, line1: Tuple[int, int, int, int], 
                             line2: Tuple[int, int, int, int]) -> float:
        """Calculate minimum distance between two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate distance from midpoint of line1 to line2
        mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Distance from point to line formula
        A = y4 - y3
        B = x3 - x4  
        C = x4 * y3 - x3 * y4
        
        distance = abs(A * mid1_x + B * mid1_y + C) / math.sqrt(A*A + B*B + 1e-10)
        return distance

    def filter_board_lines(self, lines: List[Tuple[int, int, int, int]]) -> Tuple[List, List]:
        """Filter lines into horizontal and vertical categories for rectangular board."""
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            angle = abs(self.line_angle(line))
            length = self.line_length(line)
            
            # Only consider longer lines for board edges
            if length < 60:
                continue
                
            # Classify as horizontal (near 0° or 180°) or vertical (near 90°)
            if angle < 20 or angle > 160:
                horizontal_lines.append(line)
            elif 70 < angle < 110:
                vertical_lines.append(line)
        
        # Sort by length first
        horizontal_lines.sort(key=self.line_length, reverse=True)
        vertical_lines.sort(key=self.line_length, reverse=True)
        
        # Apply clustering only if we have insufficient long lines
        # This prevents clustering from affecting images that already work well
        horizontal_result = horizontal_lines[:6]
        vertical_result = vertical_lines[:6]
        
        # If we don't have enough good lines, try with shorter lines and clustering
        if len(horizontal_result) < 2 or len(vertical_result) < 2:
            # Try again with shorter lines
            horizontal_short = []
            vertical_short = []
            
            for line in lines:
                angle = abs(self.line_angle(line))
                length = self.line_length(line)
                
                if length < 40:  # Reduced threshold for clustering attempt
                    continue
                    
                if angle < 20 or angle > 160:
                    horizontal_short.append(line)
                elif 70 < angle < 110:
                    vertical_short.append(line)
            
            # Apply clustering to the shorter lines
            if len(horizontal_result) < 2:
                horizontal_clusters = self.cluster_parallel_lines(horizontal_short)
                horizontal_clusters.sort(key=self.line_length, reverse=True)
                horizontal_result = horizontal_clusters[:6]
            
            if len(vertical_result) < 2:
                vertical_clusters = self.cluster_parallel_lines(vertical_short)
                vertical_clusters.sort(key=self.line_length, reverse=True)
                vertical_result = vertical_clusters[:6]
        
        return horizontal_result, vertical_result
    
    def line_intersection(self, line1: Tuple[int, int, int, int], 
                         line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
        
        return (px, py)
    
    def find_corners(self, horizontal_lines: List, vertical_lines: List, 
                    image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Find board corners by intersecting horizontal and vertical lines."""
        corners = []
        height, width = image_shape[:2]
        
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self.line_intersection(h_line, v_line)
                if intersection:
                    x, y = intersection
                    # Check if intersection is within image bounds with some margin
                    if -50 <= x <= width + 50 and -50 <= y <= height + 50:
                        corners.append((x, y))
        
        # Remove duplicate corners (within 30 pixels of each other)
        filtered_corners = []
        for corner in corners:
            is_duplicate = False
            for existing in filtered_corners:
                distance = math.sqrt((corner[0] - existing[0])**2 + (corner[1] - existing[1])**2)
                if distance < 30:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_corners.append(corner)
        
        # If we have more than 4 corners, select the 4 that form the best rectangle
        if len(filtered_corners) > 4:
            filtered_corners = self.select_best_rectangle_corners(filtered_corners, image_shape)
        
        return filtered_corners
    
    def select_best_rectangle_corners(self, corners: List[Tuple[float, float]], 
                                    image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Select 4 corners that best represent a rectangle."""
        if len(corners) <= 4:
            return corners
        
        # Sort corners by position to get approximate rectangle corners
        corners_array = np.array(corners)
        _ = image_shape  # Suppress unused parameter warning
        
        # Find corners that are most spread out (extreme points)
        top_left = corners_array[np.argmin(corners_array[:, 0] + corners_array[:, 1])]
        top_right = corners_array[np.argmax(corners_array[:, 0] - corners_array[:, 1])]  
        bottom_left = corners_array[np.argmax(corners_array[:, 1] - corners_array[:, 0])]
        bottom_right = corners_array[np.argmax(corners_array[:, 0] + corners_array[:, 1])]
        
        return [tuple(top_left), tuple(top_right), tuple(bottom_left), tuple(bottom_right)]
    
    def detect_board_corners(self, image: np.ndarray) -> Tuple[List[Tuple[float, float]], dict]:
        """Main function to detect board corners and return debug info."""
        # Preprocessing
        gray = self.preprocess_image(image)
        edges = self.detect_edges(gray)
        
        # Line detection
        all_lines = self.detect_lines(edges)
        horizontal_lines, vertical_lines = self.filter_board_lines(all_lines)
        
        # Corner detection
        corners = self.find_corners(horizontal_lines, vertical_lines, image.shape)
        
        # Return corners and debug information
        debug_info = {
            'gray': gray,
            'edges': edges,
            'all_lines': all_lines,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines
        }
        
        return corners, debug_info
    
    def visualize_results(self, image: np.ndarray, corners: List[Tuple[float, float]], 
                         debug_info: dict, save_path: Optional[str] = None):
        """Visualize the detection results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with results
        result_img = image.copy()
        
        # Draw all detected lines in light gray
        for line in debug_info['all_lines']:
            x1, y1, x2, y2 = line
            cv2.line(result_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        # Draw filtered horizontal lines in blue
        for line in debug_info['horizontal_lines']:
            x1, y1, x2, y2 = line
            cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw filtered vertical lines in green
        for line in debug_info['vertical_lines']:
            x1, y1, x2, y2 = line
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detected corners in red
        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(result_img, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(result_img, (x, y), 12, (0, 0, 255), 2)
        
        # Plot results
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(debug_info['edges'], cmap='gray')
        axes[0, 1].set_title('Canny Edges')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Detected Lines and Corners')
        axes[1, 0].axis('off')
        
        # Summary
        axes[1, 1].text(0.1, 0.8, f'Detected Corners: {len(corners)}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f'Horizontal Lines: {len(debug_info["horizontal_lines"])}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f'Vertical Lines: {len(debug_info["vertical_lines"])}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Total Lines: {len(debug_info["all_lines"])}', fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Detection Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        _ = fig  # Suppress unused variable warning
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detect board corners in Kill Team game board images')
    parser.add_argument('image_path', nargs='?', default="tests/test-data/board001.jpg",
                        help='Path to the board image file (default: tests/test-data/board001.jpg)')
    parser.add_argument('--output', '-o', default="board_detection_result.png",
                        help='Output path for visualization image (default: board_detection_result.png)')
    
    args = parser.parse_args()
    
    # Load the image
    image = cv2.imread(args.image_path)
    
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    print(f"Loaded image: {image.shape}")
    
    # Create detector with optimized parameters
    detector = BoardCornerDetector(
        canny_low=30,
        canny_high=90,
        hough_threshold=35,
        min_line_length=60,
        max_line_gap=30
    )
    
    # Detect corners
    corners, debug_info = detector.detect_board_corners(image)
    
    print(f"\nDetected {len(corners)} corners:")
    for i, (x, y) in enumerate(corners):
        print(f"Corner {i+1}: ({x:.1f}, {y:.1f})")
    
    # Visualize results
    detector.visualize_results(image, corners, debug_info, args.output)


if __name__ == "__main__":
    main()