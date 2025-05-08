import os
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import shutil
from tqdm import tqdm


class VehicleDamageScorer:
    """
    A class to analyze and score images for vehicle damage detection.
    Focuses on finding whole vehicles with visible damage.
    """
    
    def __init__(self, min_image_size=(300, 300)):
        """Initialize the scorer with minimum image dimensions."""
        self.min_image_size = min_image_size
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.jfif', '.gif']
    
    def load_image(self, image_path):
        """Load an image and convert to BGR format."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Check if image is large enough
        if img.shape[0] < self.min_image_size[0] or img.shape[1] < self.min_image_size[1]:
            return None
            
        return img
    
    def preprocess_image(self, img):
        """Preprocess image for analysis."""
        # Create a copy to avoid modifying original
        processed = img.copy()
        
        # Create grayscale version
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Create blurred version for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return {
            'original': img,
            'gray': gray,
            'blurred': blurred
        }
    
    def calculate_image_quality_score(self, img):
        """Calculate image quality based on brightness, contrast, and blur."""
        # Convert to HSV and extract V channel (brightness)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate brightness score (0-100)
        mean_brightness = np.mean(v)
        brightness_score = self._scale_score(mean_brightness, 
                                            optimal_min=70, optimal_max=200, 
                                            range_min=0, range_max=255)
        
        # Calculate contrast score using standard deviation of brightness
        contrast = np.std(v)
        contrast_score = self._scale_score(contrast, 
                                          optimal_min=40, optimal_max=100, 
                                          range_min=0, range_max=127)
        
        # Calculate blur score using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Higher variance = sharper image
        blur_score = self._scale_score(lap_var, 
                                      optimal_min=100, optimal_max=5000, 
                                      range_min=0, range_max=10000)
        
        # Combine scores with equal weights
        quality_score = (brightness_score + contrast_score + blur_score) / 3
        
        return {
            'quality_score': quality_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'sharpness_score': blur_score,
            'brightness_value': float(mean_brightness),
            'contrast_value': float(contrast),
            'sharpness_value': float(lap_var)
        }
    
    def calculate_vehicle_shape_score(self, img):
        """
        Analyze image for vehicle-like shapes using:
        - Line detection (vehicles have strong horizontal/vertical lines)
        - Contour analysis (vehicles have specific aspect ratios)
        - Symmetry detection
        """
        processed = self.preprocess_image(img)
        gray = processed['gray']
        height, width = gray.shape
        
        # 1. Line Detection
        edges = cv2.Canny(processed['blurred'], 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=width/10, maxLineGap=20)
        
        # Calculate line score
        line_score = 0
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                # Count horizontal lines (0° or 180°)
                if angle < 20 or angle > 160:
                    horizontal_lines += 1
                
                # Count vertical lines (90°)
                if 70 < angle < 110:
                    vertical_lines += 1
            
            # Vehicles typically have a good mix of horizontal and vertical lines
            total_lines = len(lines)
            if total_lines > 0:
                # Calculate ratio of horizontal and vertical lines
                h_ratio = horizontal_lines / total_lines
                v_ratio = vertical_lines / total_lines
                
                # Ideal vehicle has roughly 60% horizontal, 30% vertical lines
                line_score = 100 * (1 - (abs(h_ratio - 0.6) + abs(v_ratio - 0.3)))
        
        # 2. Contour Analysis
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_score = 0
        if contours:
            # Get largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Calculate contour features
            if area > 0:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Calculate aspect ratio (width/height)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Vehicle aspect ratios typically between 1.5:1 and 3:1
                if 1.3 < aspect_ratio < 3.5:
                    contour_score = 100 * (1 - (min(abs(aspect_ratio - 2.3), 1.0)))
                    
                    # Adjust score based on size (vehicles should occupy significant portion of image)
                    area_ratio = area / (width * height)
                    if 0.2 < area_ratio < 0.9:
                        contour_score *= (1 - abs(area_ratio - 0.6)) * 1.5
                    else:
                        contour_score *= 0.5
        
        # 3. Symmetry Analysis (simplified)
        # Check vertical symmetry (vehicles are typically symmetrical around vertical axis)
        flipped = cv2.flip(gray, 1)  # flip horizontally
        symmetry_diff = cv2.absdiff(gray, flipped)
        symmetry_score = 100 * (1 - (np.mean(symmetry_diff) / 255))
        
        # Combine scores with weights
        vehicle_shape_score = (0.4 * line_score + 0.4 * contour_score + 0.2 * symmetry_score)
        
        return {
            'vehicle_shape_score': vehicle_shape_score,
            'line_score': line_score,
            'contour_score': contour_score,
            'symmetry_score': symmetry_score,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines
        }
    
    def calculate_damage_context_score(self, img):
        """
        Analyze potential damage patterns focusing on:
        - Edge irregularities typical of damage
        - Local texture variance
        - Color inconsistencies
        """
        processed = self.preprocess_image(img)
        
        # Edge analysis for damage detection
        edges = cv2.Canny(processed['blurred'], 50, 150)
        
        # Divide image into a grid (5x5)
        h, w = edges.shape
        grid_h, grid_w = h // 5, w // 5
        
        # Calculate edge density for each cell
        max_edge_density = 0
        edge_variance = 0
        edge_densities = []
        
        for i in range(5):
            for j in range(5):
                # Get grid cell
                cell = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                
                # Calculate edge density
                edge_count = np.sum(cell > 0)
                edge_density = edge_count / (grid_h * grid_w)
                edge_densities.append(edge_density)
                
                # Track maximum density
                max_edge_density = max(max_edge_density, edge_density)
        
        # Variance in edge density (damage often creates regions with varying edge density)
        edge_variance = np.var(edge_densities) * 1000  # Scale up for better discrimination
        
        # Edge irregularity score (damage has high local variance in edges)
        edge_score = self._scale_score(edge_variance, 
                                     optimal_min=0.5, optimal_max=5, 
                                     range_min=0, range_max=10)
        
        # Color consistency analysis
        # Convert to LAB color space (better for color differences)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate local color variance (damage often shows color inconsistencies)
        l_var = cv2.Laplacian(l, cv2.CV_64F).var()
        a_var = cv2.Laplacian(a, cv2.CV_64F).var()
        b_var = cv2.Laplacian(b, cv2.CV_64F).var()
        
        # Combined color variance
        color_var = (l_var + a_var + b_var) / 3
        
        # Color variance score (higher variance often indicates damage)
        color_score = self._scale_score(color_var, 
                                       optimal_min=100, optimal_max=600, 
                                       range_min=0, range_max=1000)
        
        # Local texture analysis using LBP-like approach (simplified)
        gray = processed['gray']
        texture_var = 0
        
        # Simple texture variance estimation
        for i in range(1, h-1):
            for j in range(1, w-1, 10):  # Skip pixels for efficiency
                # Get 3x3 neighborhood
                neighborhood = gray[i-1:i+2, j-1:j+2]
                # Calculate variance in this neighborhood
                texture_var += np.var(neighborhood)
        
        # Normalize by number of samples
        texture_var /= ((h-2) * (w-2) / 10)
        
        # Texture score
        texture_score = self._scale_score(texture_var, 
                                         optimal_min=100, optimal_max=1000, 
                                         range_min=0, range_max=2000)
        
        # Combined damage context score
        damage_score = (0.5 * edge_score + 0.3 * color_score + 0.2 * texture_score)
        
        return {
            'damage_context_score': damage_score,
            'edge_score': edge_score,
            'color_score': color_score,
            'texture_score': texture_score,
            'edge_variance': float(edge_variance),
            'color_variance': float(color_var),
            'texture_variance': float(texture_var)
        }
    
    def calculate_composition_score(self, img):
        """
        Analyze image composition to find well-framed vehicle shots:
        - Subject position (rule of thirds)
        - Subject size (proportion of frame)
        - Overall composition quality
        """
        # Get image dimensions
        h, w, _ = img.shape
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find edges for contour detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'composition_score': 0,
                'subject_size_score': 0,
                'subject_position_score': 0,
                'subject_size_ratio': 0,
                'subject_center_dist': 1.0  # worst possible
            }
        
        # Find largest contour (likely the main subject)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)
        
        # Calculate subject size ratio (what percentage of frame is filled)
        subject_size_ratio = (cw * ch) / (w * h)
        
        # Score subject size (ideal: 30-70% of frame)
        if 0.3 <= subject_size_ratio <= 0.7:
            subject_size_score = 100 * (1 - abs(subject_size_ratio - 0.5) * 2)
        else:
            # Penalize if too small or too large
            subject_size_score = 50 * (1 - min(abs(subject_size_ratio - 0.5) * 2, 1.0))
        
        # Calculate subject position (distance from center)
        img_center_x, img_center_y = w / 2, h / 2
        subject_center_x = x + cw / 2
        subject_center_y = y + ch / 2
        
        # Normalize distance from center (0 = centered, 1 = at edge)
        center_dist_x = abs(subject_center_x - img_center_x) / (w / 2)
        center_dist_y = abs(subject_center_y - img_center_y) / (h / 2)
        subject_center_dist = (center_dist_x + center_dist_y) / 2
        
        # Score subject position (ideal: near center for damage assessment)
        subject_position_score = 100 * (1 - subject_center_dist)
        
        # Combined composition score
        composition_score = (0.6 * subject_size_score + 0.4 * subject_position_score)
        
        return {
            'composition_score': composition_score,
            'subject_size_score': subject_size_score,
            'subject_position_score': subject_position_score,
            'subject_size_ratio': float(subject_size_ratio),
            'subject_center_dist': float(subject_center_dist)
        }
    
    def calculate_information_intensity(self, img):
        """
        Calculate information intensity based on:
        - Image entropy
        - Edge density
        - Texture complexity
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate texture complexity using GLCM-like approach (simplified)
        texture_complexity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize scores
        entropy_score = self._scale_score(entropy, 
                                         optimal_min=3, optimal_max=7, 
                                         range_min=0, range_max=8)
        
        edge_density_score = self._scale_score(edge_density, 
                                              optimal_min=0.05, optimal_max=0.2, 
                                              range_min=0, range_max=0.3)
        
        texture_score = self._scale_score(texture_complexity, 
                                         optimal_min=100, optimal_max=1000, 
                                         range_min=0, range_max=2000)
        
        # Combined information intensity score
        info_score = (0.4 * entropy_score + 0.4 * edge_density_score + 0.2 * texture_score)
        
        return {
            'information_intensity': info_score,
            'entropy_score': entropy_score,
            'edge_density_score': edge_density_score,
            'texture_complexity_score': texture_score,
            'entropy_value': float(entropy),
            'edge_density_value': float(edge_density),
            'texture_complexity_value': float(texture_complexity)
        }
    
    def calculate_combined_vehicle_damage_metric(self, damage_context, vehicle_shape):
        """
        Calculate combined metric indicating vehicle damage correlation.
        Higher score means damage is likely on a vehicle.
        """
        # Extract values
        damage_score = damage_context['damage_context_score']
        vehicle_score = vehicle_shape['vehicle_shape_score']
        
        # Combine with emphasis on both being high
        # This creates a multiplier effect when both damage and vehicle are detected
        combined_score = np.sqrt(damage_score * vehicle_score)
        
        return combined_score
    
    def analyze_image(self, image_path):
        """Perform complete analysis of image and return scores."""
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return None
        
        # Run all metrics
        quality_metrics = self.calculate_image_quality_score(img)
        vehicle_metrics = self.calculate_vehicle_shape_score(img)
        damage_metrics = self.calculate_damage_context_score(img)
        composition_metrics = self.calculate_composition_score(img)
        info_metrics = self.calculate_information_intensity(img)
        
        # Calculate combined vehicle-damage metric
        combined_metric = self.calculate_combined_vehicle_damage_metric(
            damage_metrics, vehicle_metrics)
        
        # Calculate final score with weighted components
        final_score = (
            0.3 * vehicle_metrics['vehicle_shape_score'] + 
            0.3 * damage_metrics['damage_context_score'] + 
            0.2 * composition_metrics['composition_score'] + 
            0.1 * quality_metrics['quality_score'] + 
            0.1 * info_metrics['information_intensity']
        )
        
        # Apply bonus for images that have both vehicle shape and damage indicators
        # This promotes whole vehicle with damage photos
        final_score *= (1 + 0.2 * (combined_metric / 100))
        
        # Clamp score to 0-100
        final_score = max(0, min(100, final_score))
        
        # Create results dictionary
        results = {
            'filename': os.path.basename(image_path),
            'path': str(image_path),
            'file_size_kb': os.path.getsize(image_path) / 1024,
            'dimensions': f"{img.shape[1]}x{img.shape[0]}",
            'final_score': float(final_score),
            'combined_vehicle_damage_metric': float(combined_metric),
            'vehicle_shape_score': float(vehicle_metrics['vehicle_shape_score']),
            'damage_context_score': float(damage_metrics['damage_context_score']),
            'composition_score': float(composition_metrics['composition_score']),
            'quality_score': float(quality_metrics['quality_score']),
            'information_intensity': float(info_metrics['information_intensity']),
            'detailed_metrics': {
                'vehicle_metrics': vehicle_metrics,
                'damage_metrics': damage_metrics,
                'composition_metrics': composition_metrics,
                'quality_metrics': quality_metrics,
                'info_metrics': info_metrics
            }
        }
        
        return results
    
    def _scale_score(self, value, optimal_min, optimal_max, range_min, range_max):
        """
        Scale a value to a 0-100 score, with highest scores in optimal range.
        
        Args:
            value: The value to scale
            optimal_min: Lower bound of optimal range
            optimal_max: Upper bound of optimal range
            range_min: Minimum possible value
            range_max: Maximum possible value
            
        Returns:
            Score from 0-100
        """
        # Clamp value to range
        value = max(range_min, min(range_max, value))
        
        # If in optimal range, give high score
        if optimal_min <= value <= optimal_max:
            # Score based on position within optimal range
            return 80 + 20 * (1 - abs(value - (optimal_min + optimal_max) / 2) / 
                            ((optimal_max - optimal_min) / 2))
        
        # If below optimal range
        elif value < optimal_min:
            # Scale from range_min to optimal_min
            return 80 * (value - range_min) / (optimal_min - range_min)
        
        # If above optimal range
        else:
            # Scale from optimal_max to range_max
            return 80 * (1 - (value - optimal_max) / (range_max - optimal_max))


def process_claim_folder(folder_path, output_folder=None, top_n=5, save_json=True):
    """
    Process all images in a claim folder, score them, and optionally copy top images.
    
    Args:
        folder_path: Path to the claim folder
        output_folder: Path to save selected images (optional)
        top_n: Number of top images to select
        save_json: Whether to save results to JSON
        
    Returns:
        Dictionary with analysis results
    """
    folder_path = Path(folder_path)
    scorer = VehicleDamageScorer()
    
    # Find all image files in folder
    image_files = []
    for ext in scorer.supported_formats:
        image_files.extend(folder_path.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img_result = scorer.analyze_image(img_path)
            if img_result:
                results.append(img_result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Sort results by score
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Get top N images
    top_results = results[:top_n] if len(results) >= top_n else results
    
    # Create claim result summary
    claim_result = {
        'claim_folder': str(folder_path),
        'total_images': len(image_files),
        'processed_images': len(results),
        'top_images': top_results,
        'all_images': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to JSON
    if save_json:
        json_path = folder_path / "image_analysis_results.json"
        with open(json_path, 'w') as f:
            json.dump(claim_result, f, indent=2)
        print(f"Results saved to {json_path}")
    
    # Copy top images to output folder if specified
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for img_result in top_results:
            src_path = Path(img_result['path'])
            dst_path = output_path / f"{img_result['final_score']:.1f}_{src_path.name}"
            
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        print(f"Top {len(top_results)} images copied to {output_path}")
    
    return claim_result


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Analyze and score vehicle damage images.')
    parser.add_argument('input_folder', help='Folder containing claim images')
    parser.add_argument('--output', '-o', help='Folder to save top images')
    parser.add_argument('--top', '-n', type=int, default=5, 
                        help='Number of top images to select')
    parser.add_argument('--json_only', action='store_true',
                        help='Only generate JSON, don\'t copy images')
    
    args = parser.parse_args()
    
    output_folder = args.output if not args.json_only else None
    process_claim_folder(args.input_folder, output_folder, args.top)


if __name__ == "__main__":
    main()
