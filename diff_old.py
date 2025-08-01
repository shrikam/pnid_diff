import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime
from pathlib import Path
import json
from typing import Tuple, Dict, List, Any
import logging

# Configure logging for industrial safety compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PIDEquipmentRecognizer:
    """
    Advanced P&ID equipment recognition system for industrial safety applications
    Specifically identifies: pumps, drums, vessels, pipes, instruments, valves, heat exchangers
    """
    
    def __init__(self):
        # Equipment classification criteria based on geometric properties and P&ID standards
        self.equipment_classifiers = {
            'pump': {
                'vertices_range': (6, 12),
                'aspect_ratio_range': (0.8, 1.3),
                'area_range': (1000, 8000),
                'solidity_range': (0.7, 1.0),
                'circularity_range': (0.6, 1.0),
                'description': 'Centrifugal pump or positive displacement pump'
            },
            'drum': {
                'vertices_range': (8, 20),
                'aspect_ratio_range': (0.7, 1.4),
                'area_range': (5000, 50000),
                'solidity_range': (0.8, 1.0),
                'circularity_range': (0.7, 1.0),
                'description': 'Storage drum or process vessel'
            },
            'vessel': {
                'vertices_range': (4, 12),
                'aspect_ratio_range': (0.3, 3.0),
                'area_range': (2000, 100000),
                'solidity_range': (0.7, 1.0),
                'description': 'Process vessel, tank, or reactor'
            },
            'heat_exchanger': {
                'vertices_range': (4, 8),
                'aspect_ratio_range': (2.0, 6.0),
                'area_range': (3000, 25000),
                'solidity_range': (0.8, 1.0),
                'description': 'Shell and tube heat exchanger'
            },
            'valve': {
                'vertices_range': (3, 6),
                'aspect_ratio_range': (0.5, 2.0),
                'area_range': (200, 3000),
                'solidity_range': (0.5, 1.0),
                'description': 'Control valve or isolation valve'
            },
            'instrument': {
                'vertices_range': (6, 20),
                'aspect_ratio_range': (0.7, 1.3),
                'area_range': (100, 2000),
                'solidity_range': (0.6, 1.0),
                'circularity_range': (0.6, 1.0),
                'description': 'Process instrument or indicator'
            },
            'pipe': {
                'vertices_range': (2, 8),
                'aspect_ratio_range': (3.0, 50.0),  # Very elongated
                'area_range': (500, 20000),
                'solidity_range': (0.3, 1.0),
                'description': 'Process piping or pipeline'
            }
        }
        
        # Safety criticality for different equipment types
        self.equipment_safety_levels = {
            'pump': 'HIGH',           # Critical for process flow
            'drum': 'MEDIUM',         # Storage, important but not critical
            'vessel': 'HIGH',         # Process vessels are critical
            'heat_exchanger': 'MEDIUM', # Important for process efficiency
            'valve': 'CRITICAL',      # Flow control - safety critical
            'instrument': 'HIGH',     # Monitoring and control
            'pipe': 'MEDIUM'          # Transport, important
        }
        
        # Color coding for equipment types
        self.equipment_colors = {
            'pump': (255, 100, 0),        # Orange - Rotating equipment
            'drum': (0, 200, 0),          # Green - Storage
            'vessel': (0, 150, 255),      # Blue - Process vessels
            'heat_exchanger': (255, 200, 0), # Yellow - Heat transfer
            'valve': (255, 0, 0),         # Red - Critical control
            'instrument': (255, 0, 255),  # Magenta - Instrumentation
            'pipe': (0, 255, 255),       # Cyan - Piping
            'unknown': (128, 128, 128)   # Gray - Unidentified
        }
    
    def recognize_equipment(self, contour, img_context=None) -> Dict[str, Any]:
        """
        Advanced equipment recognition for P&ID symbols
        Returns detailed equipment analysis including safety criticality
        """
        try:
            # Calculate geometric properties
            area = cv2.contourArea(contour)
            if area < 50:  # Filter tiny contours
                return None
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate advanced geometric properties
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Circularity (4π*Area/Perimeter²) - perfect circle = 1.0
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Polygon approximation for vertex counting
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)
            
            # Special pipe detection (very elongated shapes)
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                if self._matches_criteria('pipe', vertices, aspect_ratio, area, solidity, circularity):
                    equipment_type = 'pipe'
                else:
                    equipment_type = 'unknown'
            else:
                # Classify other equipment types
                equipment_type = self._classify_equipment_type(vertices, aspect_ratio, area, solidity, circularity)
            
            # Determine safety criticality
            safety_level = self.equipment_safety_levels.get(equipment_type, 'LOW')
            
            # Get equipment description
            description = self.equipment_classifiers.get(equipment_type, {}).get('description', 'Unknown equipment')
            
            return {
                'equipment_type': equipment_type,
                'vertices': vertices,
                'area': area,
                'aspect_ratio': round(aspect_ratio, 2),
                'solidity': round(solidity, 2),
                'circularity': round(circularity, 2),
                'perimeter': round(perimeter, 2),
                'bounding_box': (x, y, w, h),
                'position': (x + w//2, y + h//2),  # Center position
                'safety_level': safety_level,
                'description': description,
                'confidence': self._calculate_confidence(equipment_type, vertices, aspect_ratio, area, solidity, circularity)
            }
            
        except Exception as e:
            logger.warning(f"Error in equipment recognition: {str(e)}")
            return None
    
    def _classify_equipment_type(self, vertices: int, aspect_ratio: float, area: int, 
                               solidity: float, circularity: float) -> str:
        """
        Classify equipment type based on geometric properties
        """
        best_match = 'unknown'
        best_score = 0
        
        for eq_type, criteria in self.equipment_classifiers.items():
            if eq_type == 'pipe':  # Handled separately
                continue
                
            score = self._calculate_match_score(eq_type, vertices, aspect_ratio, area, solidity, circularity)
            
            if score > best_score and score > 0.6:  # Minimum confidence threshold
                best_score = score
                best_match = eq_type
        
        return best_match
    
    def _matches_criteria(self, eq_type: str, vertices: int, aspect_ratio: float, 
                         area: int, solidity: float, circularity: float) -> bool:
        """
        Check if geometric properties match equipment criteria
        """
        criteria = self.equipment_classifiers.get(eq_type, {})
        
        # Check each criterion
        if 'vertices_range' in criteria:
            v_min, v_max = criteria['vertices_range']
            if not (v_min <= vertices <= v_max):
                return False
        
        if 'aspect_ratio_range' in criteria:
            ar_min, ar_max = criteria['aspect_ratio_range']
            if not (ar_min <= aspect_ratio <= ar_max):
                return False
        
        if 'area_range' in criteria:
            a_min, a_max = criteria['area_range']
            if not (a_min <= area <= a_max):
                return False
        
        if 'solidity_range' in criteria:
            s_min, s_max = criteria['solidity_range']
            if not (s_min <= solidity <= s_max):
                return False
        
        if 'circularity_range' in criteria:
            c_min, c_max = criteria['circularity_range']
            if not (c_min <= circularity <= c_max):
                return False
        
        return True
    
    def _calculate_match_score(self, eq_type: str, vertices: int, aspect_ratio: float,
                              area: int, solidity: float, circularity: float) -> float:
        """
        Calculate confidence score for equipment classification
        """
        criteria = self.equipment_classifiers.get(eq_type, {})
        score = 0
        total_criteria = 0
        
        # Vertices score
        if 'vertices_range' in criteria:
            v_min, v_max = criteria['vertices_range']
            if v_min <= vertices <= v_max:
                score += 1
            total_criteria += 1
        
        # Aspect ratio score
        if 'aspect_ratio_range' in criteria:
            ar_min, ar_max = criteria['aspect_ratio_range']
            if ar_min <= aspect_ratio <= ar_max:
                score += 1
            total_criteria += 1
        
        # Area score
        if 'area_range' in criteria:
            a_min, a_max = criteria['area_range']
            if a_min <= area <= a_max:
                score += 1
            total_criteria += 1
        
        # Solidity score
        if 'solidity_range' in criteria:
            s_min, s_max = criteria['solidity_range']
            if s_min <= solidity <= s_max:
                score += 1
            total_criteria += 1
        
        # Circularity score (if applicable)
        if 'circularity_range' in criteria:
            c_min, c_max = criteria['circularity_range']
            if c_min <= circularity <= c_max:
                score += 1
            total_criteria += 1
        
        return score / total_criteria if total_criteria > 0 else 0
    
    def _calculate_confidence(self, eq_type: str, vertices: int, aspect_ratio: float,
                            area: int, solidity: float, circularity: float) -> float:
        """
        Calculate overall confidence in equipment classification
        """
        if eq_type == 'unknown':
            return 0.0
        
        return self._calculate_match_score(eq_type, vertices, aspect_ratio, area, solidity, circularity)

class PIDPipelineDetector:
    """
    Specialized detector for piping systems in P&ID diagrams
    """
    
    def __init__(self):
        self.line_detector_params = {
            'rho': 1,
            'theta': np.pi/180,
            'threshold': 50,
            'min_line_length': 30,
            'max_line_gap': 10
        }
    
    def detect_pipes(self, img: np.ndarray, contours: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect piping systems using line detection and contour analysis
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Edge detection for line finding
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough Line Transform
            lines = cv2.HoughLinesP(edges, **self.line_detector_params)
            
            pipe_elements = []
            
            if lines is not None:
                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Filter for pipe-like lines (sufficient length and proper orientation)
                    if length > 50:  # Minimum pipe length
                        pipe_element = {
                            'pipe_id': f'PIPE_{i+1}',
                            'start_point': (x1, y1),
                            'end_point': (x2, y2),
                            'position': ((x1+x2)//2, (y1+y2)//2),
                            'length': round(length, 2),
                            'angle': round(angle, 2),
                            'pipe_type': self._classify_pipe_type(angle, length),
                            'equipment_type': 'pipe',
                            'safety_level': 'MEDIUM',
                            'description': f'Process piping - {self._get_orientation(angle)}'
                        }
                        
                        pipe_elements.append(pipe_element)
            
            # Also check elongated contours for pipes
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Very elongated shapes are likely pipes
                if (aspect_ratio > 4.0 or aspect_ratio < 0.25) and area > 500:
                    pipe_element = {
                        'pipe_id': f'PIPE_CONTOUR_{i+1}',
                        'bounding_box': (x, y, w, h),
                        'position': (x + w//2, y + h//2),
                        'area': area,
                        'aspect_ratio': round(aspect_ratio, 2),
                        'pipe_type': 'process_pipe',
                        'equipment_type': 'pipe',
                        'safety_level': 'MEDIUM',
                        'description': f'Process piping - {"horizontal" if aspect_ratio > 1 else "vertical"}'
                    }
                    
                    pipe_elements.append(pipe_element)
            
            return pipe_elements
            
        except Exception as e:
            logger.warning(f"Error in pipe detection: {str(e)}")
            return []
    
    def _classify_pipe_type(self, angle: float, length: float) -> str:
        """
        Classify pipe type based on orientation and length
        """
        abs_angle = abs(angle)
        
        if abs_angle < 15 or abs_angle > 165:
            pipe_type = "horizontal_main"
        elif 75 < abs_angle < 105:
            pipe_type = "vertical_main"
        else:
            pipe_type = "connecting_line"
        
        if length > 200:
            pipe_type += "_long"
        
        return pipe_type
    
    def _get_orientation(self, angle: float) -> str:
        """
        Get human-readable orientation description
        """
        abs_angle = abs(angle)
        
        if abs_angle < 15 or abs_angle > 165:
            return "horizontal"
        elif 75 < abs_angle < 105:
            return "vertical"
        else:
            return "diagonal"

class EnhancedPIDDifferenceAnalyzer:
    """
    Enhanced P&ID difference analyzer with specific equipment recognition and missing/additional tracking
    """
    
    def __init__(self, sensitivity: float = 30.0, min_area: int = 100):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.equipment_recognizer = PIDEquipmentRecognizer()
        self.pipe_detector = PIDPipelineDetector()
        
    def analyze_equipment_differences(self, img1: np.ndarray, img2: np.ndarray,
                                    img1_path: str = "Image 1", img2_path: str = "Image 2") -> Dict[str, Any]:
        """
        Enhanced difference analysis with specific P&ID equipment recognition and missing/additional detection
        """
        try:
            logger.info("Starting enhanced P&ID equipment difference analysis")
            
            # Step 1: Extract equipment from both images separately
            equipment_img1 = self._extract_equipment_from_image(img1)
            equipment_img2 = self._extract_equipment_from_image(img2)
            
            # Step 2: Perform difference analysis
            difference_analysis = self._perform_difference_analysis(img1, img2)
            
            # Step 3: Identify missing and additional equipment
            missing_additional = self._compute_missing_additional_equipment(equipment_img1, equipment_img2)
            
            # Combine all results
            analysis_results = {
                **difference_analysis,
                'missing_additional_analysis': missing_additional,
                'equipment_img1': equipment_img1,
                'equipment_img2': equipment_img2,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image1_path': img1_path,
                    'image2_path': img2_path,
                    'image_dimensions': f"{img1.shape[1]}x{img1.shape[0]}",
                    'sensitivity_threshold': self.sensitivity,
                    'minimum_area_threshold': self.min_area,
                    'equipment_recognition_enabled': True
                }
            }
            
            logger.info("Enhanced analysis completed with missing/additional equipment tracking")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in enhanced equipment difference analysis: {str(e)}")
            raise
    
    def _extract_equipment_from_image(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract all equipment from a single image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Apply thresholding to find all objects
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and recognize equipment
            equipment_list = []
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) >= self.min_area:
                    equipment_info = self.equipment_recognizer.recognize_equipment(contour, img)
                    if equipment_info:
                        equipment_info['symbol_id'] = f"EQ_{i+1}"
                        equipment_list.append(equipment_info)
            
            # Also detect pipes
            detected_pipes = self.pipe_detector.detect_pipes(img, contours)
            for pipe in detected_pipes:
                pipe['symbol_id'] = pipe['pipe_id']
                equipment_list.append(pipe)
            
            return equipment_list
            
        except Exception as e:
            logger.warning(f"Error extracting equipment from image: {str(e)}")
            return []
    
    def _perform_difference_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """
        Perform standard difference analysis
        """
        # Ensure images are the same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale for analysis
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        dilated = cv2.dilate(cleaned, np.ones((7, 7), np.uint8), iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
        
        # Create marked image
        marked_img = img2.copy()
        
        # Analyze differences
        difference_regions = []
        equipment_statistics = {}
        safety_statistics = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for i, contour in enumerate(valid_contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Recognize equipment in difference region
            equipment_info = self.equipment_recognizer.recognize_equipment(contour, img2)
            
            if equipment_info:
                equipment_type = equipment_info['equipment_type']
                safety_level = equipment_info['safety_level']
                
                equipment_statistics[equipment_type] = equipment_statistics.get(equipment_type, 0) + 1
                safety_statistics[safety_level] += 1
                
                # Draw marking
                color = self.equipment_recognizer.equipment_colors.get(equipment_type, (128, 128, 128))
                thickness = 5 if safety_level == 'CRITICAL' else 4 if safety_level == 'HIGH' else 3
                
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, thickness)
                
                # Add labels
                labels = [
                    f"D{i+1}: {equipment_type.upper()}",
                    f"SAFETY: {safety_level}",
                    f"CONF: {equipment_info['confidence']:.2f}"
                ]
                
                label_y_start = max(y - 60, 10)
                for idx, label in enumerate(labels):
                    label_y = label_y_start + (idx * 18)
                    if label_y > 10:
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(marked_img, 
                                    (x, label_y - text_height - 2),
                                    (x + text_width + 4, label_y + 2),
                                    (255, 255, 255), -1)
                        cv2.putText(marked_img, label, (x + 2, label_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                difference_regions.append({
                    'Difference_ID': i + 1,
                    'Equipment_Type': equipment_type,
                    'Safety_Level': safety_level,
                    'Confidence_Score': equipment_info['confidence'],
                    'Position': equipment_info['position'],
                    'Area': area,
                    'Bounding_Box': (x, y, w, h)
                })
        
        return {
            'difference_regions': difference_regions,
            'marked_image': marked_img,
            'difference_mask': thresh,
            'raw_difference': diff,
            'equipment_statistics': equipment_statistics,
            'safety_statistics': safety_statistics
        }
    
    def _compute_missing_additional_equipment(self, equipment_img1: List[Dict[str, Any]], 
                                            equipment_img2: List[Dict[str, Any]], 
                                            distance_threshold: float = 50.0) -> Dict[str, Any]:
        """
        Enhanced function to identify missing and additional equipment with detailed tracking
        """
        def is_position_match(pos1, pos2, threshold):
            """Check if two positions are within threshold distance"""
            return np.linalg.norm(np.array(pos1) - np.array(pos2)) < threshold
        
        def is_equipment_match(item1, item2, distance_threshold):
            """Check if two equipment items match based on position and type"""
            pos1 = item1.get('position', None)
            pos2 = item2.get('position', None)
            type1 = item1.get('equipment_type', 'unknown')
            type2 = item2.get('equipment_type', 'unknown')
            
            if pos1 is None or pos2 is None:
                return False
            
            return (type1 == type2 and 
                    is_position_match(pos1, pos2, distance_threshold))
        
        missing_equipment = []  # Present in img1 but not in img2
        additional_equipment = []  # Present in img2 but not in img1
        missing_pipes = []
        additional_pipes = []
        matched_img2_indices = set()
        
        # Separate equipment and pipes
        equipment_only_img1 = [eq for eq in equipment_img1 if eq.get('equipment_type') != 'pipe']
        pipes_only_img1 = [eq for eq in equipment_img1 if eq.get('equipment_type') == 'pipe']
        equipment_only_img2 = [eq for eq in equipment_img2 if eq.get('equipment_type') != 'pipe']
        pipes_only_img2 = [eq for eq in equipment_img2 if eq.get('equipment_type') == 'pipe']
        
        # Find missing equipment (img1 -> img2)
        for idx1, item1 in enumerate(equipment_only_img1):
            found_match = False
            for idx2, item2 in enumerate(equipment_only_img2):
                if idx2 in matched_img2_indices:
                    continue
                if is_equipment_match(item1, item2, distance_threshold):
                    found_match = True
                    matched_img2_indices.add(idx2)
                    break
            
            if not found_match:
                missing_item = {
                    'equipment_id': item1.get('symbol_id', f"MISSING_{idx1}"),
                    'equipment_type': item1.get('equipment_type', 'unknown'),
                    'position': item1.get('position', (0, 0)),
                    'description': item1.get('description', 'Equipment not found in updated P&ID'),
                    'safety_level': item1.get('safety_level', 'UNKNOWN'),
                    'bounding_box': item1.get('bounding_box', None),
                    'area': item1.get('area', 0),
                    'status': 'MISSING'
                }
                missing_equipment.append(missing_item)
        
        # Find additional equipment (img2 -> img1)
        for idx2, item2 in enumerate(equipment_only_img2):
            if idx2 not in matched_img2_indices:
                additional_item = {
                    'equipment_id': item2.get('symbol_id', f"ADDITIONAL_{idx2}"),
                    'equipment_type': item2.get('equipment_type', 'unknown'),
                    'position': item2.get('position', (0, 0)),
                    'description': item2.get('description', 'New equipment found in updated P&ID'),
                    'safety_level': item2.get('safety_level', 'UNKNOWN'),
                    'bounding_box': item2.get('bounding_box', None),
                    'area': item2.get('area', 0),
                    'status': 'ADDITIONAL'
                }
                additional_equipment.append(additional_item)
        
        # Similar analysis for pipes
        matched_pipe_indices = set()
        
        # Find missing pipes
        for idx1, pipe1 in enumerate(pipes_only_img1):
            found_match = False
            for idx2, pipe2 in enumerate(pipes_only_img2):
                if idx2 in matched_pipe_indices:
                    continue
                if is_equipment_match(pipe1, pipe2, distance_threshold):
                    found_match = True
                    matched_pipe_indices.add(idx2)
                    break
            
            if not found_match:
                missing_pipes.append({
                    'pipe_id': pipe1.get('symbol_id', f"MISSING_PIPE_{idx1}"),
                    'pipe_type': pipe1.get('pipe_type', 'unknown'),
                    'position': pipe1.get('position', (0, 0)),
                    'length': pipe1.get('length', 0),
                    'status': 'MISSING'
                })
        
        # Find additional pipes
        for idx2, pipe2 in enumerate(pipes_only_img2):
            if idx2 not in matched_pipe_indices:
                additional_pipes.append({
                    'pipe_id': pipe2.get('symbol_id', f"ADDITIONAL_PIPE_{idx2}"),
                    'pipe_type': pipe2.get('pipe_type', 'unknown'),
                    'position': pipe2.get('position', (0, 0)),
                    'length': pipe2.get('length', 0),
                    'status': 'ADDITIONAL'
                })
        
        return {
            'missing_equipment': missing_equipment,
            'additional_equipment': additional_equipment,
            'missing_pipes': missing_pipes,
            'additional_pipes': additional_pipes,
            'equipment_missing_count': len(missing_equipment),
            'equipment_additional_count': len(additional_equipment),
            'pipes_missing_count': len(missing_pipes),
            'pipes_additional_count': len(additional_pipes),
            'total_changes': len(missing_equipment) + len(additional_equipment) + len(missing_pipes) + len(additional_pipes)
        }

def generate_enhanced_missing_additional_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive report including missing and additional equipment/pipes
    """
    missing_additional = analysis_results.get('missing_additional_analysis', {})
    metadata = analysis_results.get('metadata', {})
    
    report = f"""
P&ID EQUIPMENT CHANGE ANALYSIS REPORT
{'='*50}

ANALYSIS METADATA:
- Analysis Date: {metadata.get('analysis_timestamp', 'Unknown')}
- Reference P&ID: {metadata.get('image1_path', 'Unknown')}
- Updated P&ID: {metadata.get('image2_path', 'Unknown')}
- Equipment Recognition: Enabled
- Total Equipment Changes: {missing_additional.get('total_changes', 0)}

EQUIPMENT CHANGES SUMMARY:
- Missing Equipment (Removed): {missing_additional.get('equipment_missing_count', 0)}
- Additional Equipment (Added): {missing_additional.get('equipment_additional_count', 0)}
- Missing Pipes (Removed): {missing_additional.get('pipes_missing_count', 0)}
- Additional Pipes (Added): {missing_additional.get('pipes_additional_count', 0)}

DETAILED MISSING EQUIPMENT LIST:
{'='*35}
"""
    
    missing_equipment = missing_additional.get('missing_equipment', [])
    if missing_equipment:
        for i, equipment in enumerate(missing_equipment, 1):
            report += f"""
{i}. MISSING EQUIPMENT:
   ├── ID: {equipment['equipment_id']}
   ├── Type: {equipment['equipment_type'].upper()}
   ├── Position: {equipment['position']}
   ├── Safety Level: {equipment['safety_level']}
   ├── Area: {equipment['area']} pixels
   └── Impact: Equipment removed from updated P&ID
"""
    else:
        report += "\nNo missing equipment detected.\n"
    
    report += f"""

DETAILED ADDITIONAL EQUIPMENT LIST:
{'='*37}
"""
    
    additional_equipment = missing_additional.get('additional_equipment', [])
    if additional_equipment:
        for i, equipment in enumerate(additional_equipment, 1):
            report += f"""
{i}. ADDITIONAL EQUIPMENT:
   ├── ID: {equipment['equipment_id']}
   ├── Type: {equipment['equipment_type'].upper()}
   ├── Position: {equipment['position']}
   ├── Safety Level: {equipment['safety_level']}
   ├── Area: {equipment['area']} pixels
   └── Impact: New equipment added in updated P&ID
"""
    else:
        report += "\nNo additional equipment detected.\n"
    
    report += f"""

DETAILED MISSING PIPES LIST:
{'='*30}
"""
    
    missing_pipes = missing_additional.get('missing_pipes', [])
    if missing_pipes:
        for i, pipe in enumerate(missing_pipes, 1):
            report += f"""
{i}. MISSING PIPE:
   ├── ID: {pipe['pipe_id']}
   ├── Type: {pipe['pipe_type']}
   ├── Position: {pipe['position']}
   ├── Length: {pipe.get('length', 'Unknown')} pixels
   └── Impact: Piping connection removed
"""
    else:
        report += "\nNo missing pipes detected.\n"
    
    report += f"""

DETAILED ADDITIONAL PIPES LIST:
{'='*32}
"""
    
    additional_pipes = missing_additional.get('additional_pipes', [])
    if additional_pipes:
        for i, pipe in enumerate(additional_pipes, 1):
            report += f"""
{i}. ADDITIONAL PIPE:
   ├── ID: {pipe['pipe_id']}
   ├── Type: {pipe['pipe_type']}
   ├── Position: {pipe['position']}
   ├── Length: {pipe.get('length', 'Unknown')} pixels
   └── Impact: New piping connection added
"""
    else:
        report += "\nNo additional pipes detected.\n"
    
    # Safety impact assessment
    missing_critical = len([eq for eq in missing_equipment if eq['safety_level'] == 'CRITICAL'])
    additional_critical = len([eq for eq in additional_equipment if eq['safety_level'] == 'CRITICAL'])
    
    report += f"""

SAFETY IMPACT ASSESSMENT:
{'='*27}
- Critical Equipment Removed: {missing_critical}
- Critical Equipment Added: {additional_critical}
- Total Safety-Critical Changes: {missing_critical + additional_critical}

RECOMMENDATIONS:
{'='*15}
1. MISSING EQUIPMENT REVIEW:
   - Verify removal authorization for all missing equipment
   - Ensure proper isolation and lockout procedures completed
   - Update operating procedures to reflect equipment removal
   - Consider impact on process capacity and safety systems

2. ADDITIONAL EQUIPMENT REVIEW:
   - Verify installation specifications for new equipment
   - Confirm commissioning procedures completed
   - Update equipment database and maintenance schedules
   - Ensure operator training on new equipment functionality

3. PIPING CHANGES REVIEW:
   - Verify piping modifications follow approved designs
   - Confirm pressure testing and leak detection completed
   - Update piping isometrics and stress analysis
   - Ensure material compatibility and code compliance

4. DOCUMENTATION UPDATES:
   - Update P&ID revision numbers and change logs
   - Revise operating procedures and emergency responses
   - Update equipment lists and spare parts inventory
   - Complete Management of Change (MOC) documentation

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Tool: Enhanced P&ID Equipment Difference Analyzer
For Industrial Safety Applications
"""
    
    return report

def create_enhanced_csv_with_missing_additional(analysis_results: Dict[str, Any]) -> bytes:
    """
    Create comprehensive CSV including missing and additional equipment/pipes
    """
    try:
        # Extract missing/additional data
        missing_additional = analysis_results.get('missing_additional_analysis', {})
        
        # Create comprehensive datasets
        all_changes = []
        
        # Add missing equipment
        for equipment in missing_additional.get('missing_equipment', []):
            all_changes.append({
                'Change_Type': 'MISSING_EQUIPMENT',
                'Equipment_ID': equipment['equipment_id'],
                'Equipment_Type': equipment['equipment_type'],
                'Position_X': equipment['position'][0],
                'Position_Y': equipment['position'][1],
                'Safety_Level': equipment['safety_level'],
                'Area_Pixels': equipment['area'],
                'Description': equipment['description'],
                'Impact': 'Equipment removed from updated P&ID'
            })
        
        # Add additional equipment
        for equipment in missing_additional.get('additional_equipment', []):
            all_changes.append({
                'Change_Type': 'ADDITIONAL_EQUIPMENT',
                'Equipment_ID': equipment['equipment_id'],
                'Equipment_Type': equipment['equipment_type'],
                'Position_X': equipment['position'][0],
                'Position_Y': equipment['position'][1],
                'Safety_Level': equipment['safety_level'],
                'Area_Pixels': equipment['area'],
                'Description': equipment['description'],
                'Impact': 'New equipment added in updated P&ID'
            })
        
        # Add missing pipes
        for pipe in missing_additional.get('missing_pipes', []):
            all_changes.append({
                'Change_Type': 'MISSING_PIPE',
                'Equipment_ID': pipe['pipe_id'],
                'Equipment_Type': pipe['pipe_type'],
                'Position_X': pipe['position'][0],
                'Position_Y': pipe['position'][1],
                'Safety_Level': 'MEDIUM',
                'Area_Pixels': pipe.get('length', 0),
                'Description': f"Pipe connection removed: {pipe['pipe_type']}",
                'Impact': 'Piping connection removed from updated P&ID'
            })
        
        # Add additional pipes
        for pipe in missing_additional.get('additional_pipes', []):
            all_changes.append({
                'Change_Type': 'ADDITIONAL_PIPE',
                'Equipment_ID': pipe['pipe_id'],
                'Equipment_Type': pipe['pipe_type'],
                'Position_X': pipe['position'][0],
                'Position_Y': pipe['position'][1],
                'Safety_Level': 'MEDIUM',
                'Area_Pixels': pipe.get('length', 0),
                'Description': f"New pipe connection added: {pipe['pipe_type']}",
                'Impact': 'New piping connection added in updated P&ID'
            })
        
        # Create DataFrame
        df_changes = pd.DataFrame(all_changes)
        
        # Add summary information
        summary_data = [
            ['MISSING_ADDITIONAL_SUMMARY', '', '', '', '', '', '', '', ''],
            ['Total_Missing_Equipment', len(missing_additional.get('missing_equipment', [])), '', '', '', '', '', '', ''],
            ['Total_Additional_Equipment', len(missing_additional.get('additional_equipment', [])), '', '', '', '', '', '', ''],
            ['Total_Missing_Pipes', len(missing_additional.get('missing_pipes', [])), '', '', '', '', '', '', ''],
            ['Total_Additional_Pipes', len(missing_additional.get('additional_pipes', [])), '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', ''],
            ['DETAILED_CHANGES', '', '', '', '', '', '', '', '']
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=df_changes.columns)
        final_df = pd.concat([summary_df, df_changes], ignore_index=True)
        
        # Convert to CSV bytes
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating enhanced CSV: {str(e)}")
        # Return empty CSV on error
        empty_df = pd.DataFrame({'Error': [f'Error creating enhanced CSV: {str(e)}']})
        csv_buffer = BytesIO()
        empty_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

def create_enhanced_visualizations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create enhanced visualizations for missing/additional equipment analysis
    """
    try:
        missing_additional = analysis_results.get('missing_additional_analysis', {})
        visualizations = {}
        
        # Missing vs Additional Equipment Chart
        categories = ['Missing Equipment', 'Additional Equipment', 'Missing Pipes', 'Additional Pipes']
        counts = [
            missing_additional.get('equipment_missing_count', 0),
            missing_additional.get('equipment_additional_count', 0),
            missing_additional.get('pipes_missing_count', 0),
            missing_additional.get('pipes_additional_count', 0)
        ]
        
        if sum(counts) > 0:
            fig = px.bar(
                x=categories, y=counts,
                title="Missing vs Additional Equipment and Pipes",
                color=categories,
                color_discrete_map={
                    'Missing Equipment': '#FF6B6B',
                    'Additional Equipment': '#4ECDC4',
                    'Missing Pipes': '#FFE66D',
                    'Additional Pipes': '#95E1D3'
                }
            )
            visualizations['missing_additional_bar'] = fig
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating enhanced visualizations: {str(e)}")
        return {}

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Enhanced P&ID Equipment Difference Analyzer",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Enhanced P&ID Equipment Difference Analyzer")
    st.markdown("**Industrial Safety Analysis with Equipment Recognition and Missing/Additional Equipment Tracking**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Configuration")
    sensitivity = st.sidebar.slider("Sensitivity", 10.0, 100.0, 30.0, 5.0)
    min_area = st.sidebar.slider("Minimum Area", 50, 1000, 100, 50)
    distance_threshold = st.sidebar.slider("Position Matching Threshold (pixels)", 20, 100, 50, 10)
    
    st.sidebar.markdown("### 🏭 Recognized Equipment Types")
    st.sidebar.markdown("- **Pumps**: Centrifugal and positive displacement")
    st.sidebar.markdown("- **Drums/Tanks**: Storage vessels")  
    st.sidebar.markdown("- **Vessels**: Process reactors and separators")
    st.sidebar.markdown("- **Heat Exchangers**: Shell and tube equipment")
    st.sidebar.markdown("- **Valves**: Control and isolation (CRITICAL)")
    st.sidebar.markdown("- **Instruments**: Monitoring and control devices")
    st.sidebar.markdown("- **Pipes**: Process piping systems")
    
    # File upload
    st.header("📁 Upload P&ID Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reference P&ID (Original)")
        uploaded_img1 = st.file_uploader("Choose reference image", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'], key="img1")
        if uploaded_img1:
            image1 = Image.open(uploaded_img1)
            st.image(image1, caption="Reference P&ID", use_container_width=True)
    
    with col2:
        st.subheader("Updated P&ID (Modified)")
        uploaded_img2 = st.file_uploader("Choose updated image", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'], key="img2")
        if uploaded_img2:
            image2 = Image.open(uploaded_img2)
            st.image(image2, caption="Updated P&ID", use_container_width=True)
    
    # Analysis
    if uploaded_img1 and uploaded_img2:
        st.header("🔬 Enhanced Equipment Analysis")
        
        if st.button("🚀 Analyze Equipment Changes", type="primary"):
            with st.spinner("Analyzing P&ID equipment changes... Detecting missing and additional equipment..."):
                try:
                    # Initialize analyzer
                    analyzer = EnhancedPIDDifferenceAnalyzer(sensitivity=sensitivity, min_area=min_area)
                    
                    # Convert images
                    img1_array = np.array(image1)
                    img2_array = np.array(image2)
                    
                    if len(img1_array.shape) == 3:
                        img1_array = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
                    if len(img2_array.shape) == 3:
                        img2_array = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)
                    
                    # Perform analysis
                    analysis_results = analyzer.analyze_equipment_differences(
                        img1_array, img2_array, uploaded_img1.name, uploaded_img2.name
                    )
                    
                    st.success("✅ Enhanced equipment analysis completed!")
                    
                    # Display summary
                    missing_additional = analysis_results['missing_additional_analysis']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Missing Equipment", missing_additional['equipment_missing_count'])
                    with col2:
                        st.metric("Additional Equipment", missing_additional['equipment_additional_count'])
                    with col3:
                        st.metric("Missing Pipes", missing_additional['pipes_missing_count'])
                    with col4:
                        st.metric("Additional Pipes", missing_additional['pipes_additional_count'])
                    
                    # Safety alerts
                    total_changes = missing_additional['total_changes']
                    if total_changes > 0:
                        st.warning(f"⚠️ {total_changes} equipment/pipe changes detected! Review required.")
                    else:
                        st.info("✅ No equipment or pipe changes detected.")
                    
                    # Display marked image
                    st.subheader("🎯 Marked Differences")
                    marked_img_rgb = cv2.cvtColor(analysis_results['marked_image'], cv2.COLOR_BGR2RGB)
                    st.image(marked_img_rgb, caption="P&ID with Equipment Differences Marked", use_container_width=True)
                    
                    # Visualizations
                    if total_changes > 0:
                        st.subheader("📊 Change Analysis Visualizations")
                        visualizations = create_enhanced_visualizations(analysis_results)
                        
                        if 'missing_additional_bar' in visualizations:
                            st.plotly_chart(visualizations['missing_additional_bar'], use_container_width=True)
                    
                    # Detailed tables
                    st.subheader("📋 Detailed Change Analysis")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Missing Equipment", "Additional Equipment", "Missing Pipes", "Additional Pipes"])
                    
                    with tab1:
                        missing_equipment = missing_additional.get('missing_equipment', [])
                        if missing_equipment:
                            df_missing = pd.DataFrame(missing_equipment)
                            st.dataframe(df_missing, use_container_width=True)
                        else:
                            st.info("No missing equipment detected.")
                    
                    with tab2:
                        additional_equipment = missing_additional.get('additional_equipment', [])
                        if additional_equipment:
                            df_additional = pd.DataFrame(additional_equipment)
                            st.dataframe(df_additional, use_container_width=True)
                        else:
                            st.info("No additional equipment detected.")
                    
                    with tab3:
                        missing_pipes = missing_additional.get('missing_pipes', [])
                        if missing_pipes:
                            df_missing_pipes = pd.DataFrame(missing_pipes)
                            st.dataframe(df_missing_pipes, use_container_width=True)
                        else:
                            st.info("No missing pipes detected.")
                    
                    with tab4:
                        additional_pipes = missing_additional.get('additional_pipes', [])
                        if additional_pipes:
                            df_additional_pipes = pd.DataFrame(additional_pipes)
                            st.dataframe(df_additional_pipes, use_container_width=True)
                        else:
                            st.info("No additional pipes detected.")
                    
                    # Download section
                    st.subheader("💾 Download Enhanced Results")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        # Enhanced CSV download
                        csv_bytes = create_enhanced_csv_with_missing_additional(analysis_results)
                        st.download_button(
                            label="📊 Download Complete Analysis CSV",
                            data=csv_bytes,
                            file_name=f"enhanced_pid_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with download_col2:
                        # Enhanced marked image download
                        img_buffer = BytesIO()
                        marked_pil = Image.fromarray(marked_img_rgb)
                        marked_pil.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        st.download_button(
                            label="🖼️ Download Marked P&ID",
                            data=img_bytes,
                            file_name=f"enhanced_pid_marked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    
                    with download_col3:
                        # Enhanced report download
                        report_text = generate_enhanced_missing_additional_report(analysis_results)
                        st.download_button(
                            label="📄 Download Equipment Change Report",
                            data=report_text,
                            file_name=f"equipment_change_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()
