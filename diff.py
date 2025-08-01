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
    Enhanced P&ID difference analyzer with specific equipment recognition
    """
    
    def __init__(self, sensitivity: float = 30.0, min_area: int = 100):
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.equipment_recognizer = PIDEquipmentRecognizer()
        self.pipe_detector = PIDPipelineDetector()
        
    def analyze_equipment_differences(self, img1: np.ndarray, img2: np.ndarray,
                                    img1_path: str = "Image 1", img2_path: str = "Image 2") -> Dict[str, Any]:
        """
        Enhanced difference analysis with specific P&ID equipment recognition
        """
        try:
            logger.info("Starting enhanced P&ID equipment difference analysis")
            
            # Ensure images are the same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                logger.warning("Images resized to match dimensions")
            
            # Convert to grayscale for analysis
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Apply threshold to create binary difference image
            _, thresh = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)
            
            # Enhanced morphological operations for P&ID equipment
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Dilate to merge nearby differences (equipment parts)
            dilate_kernel = np.ones((7, 7), np.uint8)
            dilated = cv2.dilate(cleaned, dilate_kernel, iterations=2)
            
            # Find contours of differences
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
            
            # Create marked image (copy of img2)
            marked_img = img2.copy()
            
            # Analyze each difference region with equipment recognition
            equipment_differences = []
            pipe_differences = []
            instrument_differences = []
            
            total_diff_area = 0
            equipment_statistics = {}
            safety_statistics = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            # Detect pipes separately using line detection
            detected_pipes = self.pipe_detector.detect_pipes(img2, valid_contours)
            
            for i, contour in enumerate(valid_contours):
                # Basic contour properties
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                total_diff_area += area
                
                # Recognize equipment type
                equipment_info = self.equipment_recognizer.recognize_equipment(contour, img2)
                
                if equipment_info:
                    equipment_type = equipment_info['equipment_type']
                    safety_level = equipment_info['safety_level']
                    confidence = equipment_info['confidence']
                    
                    # Update statistics
                    equipment_statistics[equipment_type] = equipment_statistics.get(equipment_type, 0) + 1
                    safety_statistics[safety_level] += 1
                    
                    # Choose color based on equipment type
                    color = self.equipment_recognizer.equipment_colors.get(equipment_type, 
                                                                          self.equipment_recognizer.equipment_colors['unknown'])
                    
                    # Line thickness based on safety level
                    line_thickness = 5 if safety_level == 'CRITICAL' else 4 if safety_level == 'HIGH' else 3
                    
                    # Draw enhanced marking with equipment-specific styling
                    cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, line_thickness)
                    
                    # Add comprehensive labeling
                    label_lines = [
                        f"D{i+1}: {equipment_type.upper()}",
                        f"SAFETY: {safety_level}",
                        f"CONF: {confidence:.2f}"
                    ]
                    
                    # Draw labels with backgrounds
                    label_y_start = max(y - 60, 10)
                    for idx, label in enumerate(label_lines):
                        label_y = label_y_start + (idx * 18)
                        if label_y > 10:
                            # Background rectangle for text visibility
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            cv2.rectangle(marked_img, 
                                        (x, label_y - text_height - 2),
                                        (x + text_width + 4, label_y + 2),
                                        (255, 255, 255), -1)
                            
                            # Draw text
                            cv2.putText(marked_img, label, (x + 2, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Create comprehensive difference record
                    diff_record = {
                        'Difference_ID': i + 1,
                        'Equipment_Type': equipment_type,
                        'Safety_Level': safety_level,
                        'Confidence_Score': round(confidence, 3),
                        'X_Coordinate': int(x),
                        'Y_Coordinate': int(y),
                        'Width': int(w),
                        'Height': int(h),
                        'Area_Pixels': int(area),
                        'Aspect_Ratio': equipment_info['aspect_ratio'],
                        'Solidity': equipment_info['solidity'],
                        'Circularity': equipment_info['circularity'],
                        'Vertices': equipment_info['vertices'],
                        'Perimeter': equipment_info['perimeter'],
                        'Bounding_Box': f"({x},{y})-({x+w},{y+h})",
                        'Description': equipment_info['description'],
                        'Percentage_of_Image': round((area / (img1.shape[0] * img1.shape[1])) * 100, 4)
                    }
                    
                    # Categorize by equipment type
                    if equipment_type == 'pipe':
                        pipe_differences.append(diff_record)
                    elif equipment_type == 'instrument':
                        instrument_differences.append(diff_record)
                    else:
                        equipment_differences.append(diff_record)
                
                else:
                    # Fallback for unrecognized equipment
                    color = self.equipment_recognizer.equipment_colors['unknown']
                    cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(marked_img, f"D{i+1}: UNKNOWN", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    diff_record = {
                        'Difference_ID': i + 1,
                        'Equipment_Type': 'unknown',
                        'Safety_Level': 'LOW',
                        'Confidence_Score': 0.0,
                        'X_Coordinate': int(x),
                        'Y_Coordinate': int(y),
                        'Width': int(w),
                        'Height': int(h),
                        'Area_Pixels': int(area),
                        'Aspect_Ratio': float(w) / h if h > 0 else 0,
                        'Solidity': 0,
                        'Circularity': 0,
                        'Vertices': 0,
                        'Perimeter': 0,
                        'Bounding_Box': f"({x},{y})-({x+w},{y+h})",
                        'Description': 'Unrecognized equipment',
                        'Percentage_of_Image': round((area / (img1.shape[0] * img1.shape[1])) * 100, 4)
                    }
                    
                    equipment_differences.append(diff_record)
            
            # Add detected pipes to the analysis
            for pipe_info in detected_pipes:
                if 'bounding_box' in pipe_info:
                    x, y, w, h = pipe_info['bounding_box']
                    color = self.equipment_recognizer.equipment_colors['pipe']
                    cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(marked_img, "PIPE", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calculate overall statistics
            total_pixels = img1.shape[0] * img1.shape[1]
            diff_percentage = (total_diff_area / total_pixels) * 100
            
            # Combine all differences
            all_differences = equipment_differences + pipe_differences + instrument_differences
            
            # Create comprehensive analysis results
            analysis_results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image1_path': img1_path,
                    'image2_path': img2_path,
                    'image_dimensions': f"{img1.shape[1]}x{img1.shape[0]}",
                    'sensitivity_threshold': self.sensitivity,
                    'minimum_area_threshold': self.min_area,
                    'equipment_recognition_enabled': True
                },
                'summary_statistics': {
                    'total_differences_found': len(all_differences),
                    'equipment_differences': len(equipment_differences),
                    'pipe_differences': len(pipe_differences),
                    'instrument_differences': len(instrument_differences),
                    'total_difference_area': int(total_diff_area),
                    'percentage_changed': round(diff_percentage, 4),
                    'critical_differences': safety_statistics['CRITICAL'],
                    'high_severity_differences': safety_statistics['HIGH'],
                    'medium_severity_differences': safety_statistics['MEDIUM'],
                    'low_severity_differences': safety_statistics['LOW']
                },
                'equipment_statistics': equipment_statistics,
                'safety_statistics': safety_statistics,
                'equipment_differences': equipment_differences,
                'pipe_differences': pipe_differences,
                'instrument_differences': instrument_differences,
                'all_differences': all_differences,
                'detected_pipes': detected_pipes,
                'marked_image': marked_img,
                'difference_mask': thresh,
                'raw_difference': diff
            }
            
            logger.info(f"Equipment analysis completed: {len(all_differences)} differences found")
            logger.info(f"Equipment distribution: {equipment_statistics}")
            logger.info(f"Safety distribution: {safety_statistics}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in enhanced equipment difference analysis: {str(e)}")
            raise

# Enhanced visualization functions for equipment types
def create_equipment_visualizations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create enhanced visualizations for P&ID equipment analysis
    """
    try:
        equipment_stats = analysis_results.get('equipment_statistics', {})
        safety_stats = analysis_results.get('safety_statistics', {})
        all_differences = analysis_results.get('all_differences', [])
        
        visualizations = {}
        
        if all_differences:
            # Equipment distribution pie chart
            if equipment_stats:
                equipment_fig = px.pie(
                    values=list(equipment_stats.values()),
                    names=list(equipment_stats.keys()),
                    title="P&ID Equipment Types in Differences",
                    color_discrete_map={
                        'pump': '#FF6400',
                        'drum': '#00C800',
                        'vessel': '#0096FF',
                        'heat_exchanger': '#FFC800',
                        'valve': '#FF0000',
                        'instrument': '#FF00FF',
                        'pipe': '#00FFFF'
                    }
                )
                visualizations['equipment_pie'] = equipment_fig
            
            # Safety level analysis
            safety_counts = {k: v for k, v in safety_stats.items() if v > 0}
            if safety_counts:
                safety_fig = px.pie(
                    values=list(safety_counts.values()),
                    names=list(safety_counts.keys()),
                    title="Safety Level Distribution",
                    color_discrete_map={
                        'CRITICAL': '#FF0000',
                        'HIGH': '#FF4500',
                        'MEDIUM': '#FFFF00',
                        'LOW': '#00FF00'
                    }
                )
                visualizations['safety_pie'] = safety_fig
            
            # Equipment vs Area scatter plot
            df_diffs = pd.DataFrame(all_differences)
            if not df_diffs.empty:
                scatter_fig = px.scatter(
                    df_diffs,
                    x='Confidence_Score',
                    y='Area_Pixels',
                    color='Equipment_Type',
                    size='Aspect_Ratio',
                    symbol='Safety_Level',
                    title="P&ID Equipment Analysis: Confidence vs Area",
                    hover_data=['Description', 'Equipment_Type']
                )
                visualizations['equipment_scatter'] = scatter_fig
                
                # Equipment location map
                location_fig = go.Figure()
                
                for eq_type in df_diffs['Equipment_Type'].unique():
                    eq_data = df_diffs[df_diffs['Equipment_Type'] == eq_type]
                    if not eq_data.empty:
                        location_fig.add_trace(go.Scatter(
                            x=eq_data['X_Coordinate'],
                            y=eq_data['Y_Coordinate'],
                            mode='markers',
                            marker=dict(
                                size=eq_data['Area_Pixels'] / 100,
                                opacity=0.8,
                                line=dict(width=2, color='DarkSlateGrey')
                            ),
                            name=f'{eq_type.title()}',
                            text=eq_data['Equipment_Type'],
                            hovertemplate='<b>%{text}</b><br>' +
                                        'Location: (%{x}, %{y})<br>' +
                                        'Safety: %{customdata}<br>' +
                                        '<extra></extra>',
                            customdata=eq_data['Safety_Level']
                        ))
                
                location_fig.update_layout(
                    title="P&ID Equipment Location Map",
                    xaxis_title="X Coordinate (pixels)",
                    yaxis_title="Y Coordinate (pixels)",
                    yaxis=dict(autorange="reversed")
                )
                visualizations['equipment_location_map'] = location_fig
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating equipment visualizations: {str(e)}")
        return {}

# Updated Streamlit Application
def main():
    st.set_page_config(
        page_title="P&ID Equipment Difference Analyzer",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 P&ID Equipment Difference Analyzer")
    st.markdown("**Industrial Safety Analysis with Specific Equipment Recognition: Pumps, Drums, Vessels, Pipes, Instruments**")
    
    # Enhanced sidebar with equipment information
    st.sidebar.header("⚙️ Equipment Analysis Configuration")
    
    sensitivity = st.sidebar.slider(
        "Difference Sensitivity Threshold", 
        min_value=10.0, max_value=100.0, value=30.0, step=5.0,
        help="Lower values detect smaller differences (more sensitive)"
    )
    min_area = st.sidebar.slider(
        "Minimum Equipment Area (pixels)", 
        min_value=50, max_value=1000, value=100, step=50,
        help="Minimum area to consider as equipment"
    )
    
    st.sidebar.markdown("### 🏭 Recognized Equipment Types")
    equipment_legend = {
        "🟠 Pumps": "Centrifugal and positive displacement pumps",
        "🟢 Drums/Tanks": "Storage vessels and process drums",
        "🔵 Vessels": "Process vessels, reactors, separators",
        "🟡 Heat Exchangers": "Shell and tube heat transfer equipment",
        "🔴 Valves": "Control and isolation valves (CRITICAL)",
        "🟣 Instruments": "Process monitoring and control devices",
        "🔷 Pipes": "Process piping and connections"
    }
    
    for equipment, description in equipment_legend.items():
        st.sidebar.markdown(f"**{equipment}**: {description}")
    
    st.sidebar.markdown("### 🚨 Safety Classification")
    st.sidebar.markdown("- **CRITICAL**: Valves, safety systems")
    st.sidebar.markdown("- **HIGH**: Pumps, vessels, instruments")
    st.sidebar.markdown("- **MEDIUM**: Drums, heat exchangers, pipes")
    st.sidebar.markdown("- **LOW**: General equipment")
    
    # File upload section
    st.header("📁 Upload P&ID Images for Equipment Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Reference P&ID (Image 1)")
        uploaded_img1 = st.file_uploader(
            "Choose reference P&ID image", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="img1"
        )
        
        if uploaded_img1:
            image1 = Image.open(uploaded_img1)
            st.image(image1, caption="Reference P&ID", use_container_width=True)
    
    with col2:
        st.subheader("📄 Updated P&ID (Image 2)")
        uploaded_img2 = st.file_uploader(
            "Choose updated P&ID image", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="img2"
        )
        
        if uploaded_img2:
            image2 = Image.open(uploaded_img2)
            st.image(image2, caption="Updated P&ID", use_container_width=True)
    
    # Equipment analysis section
    if uploaded_img1 and uploaded_img2:
        st.header("🔬 P&ID Equipment Difference Analysis")
        
        if st.button("🚀 Analyze Equipment Differences", type="primary"):
            with st.spinner("Analyzing P&ID equipment differences... Recognizing pumps, drums, vessels, pipes, and instruments..."):
                try:
                    # Initialize equipment analyzer
                    analyzer = EnhancedPIDDifferenceAnalyzer(
                        sensitivity=sensitivity, 
                        min_area=min_area
                    )
                    
                    # Convert PIL images to numpy arrays
                    img1_array = np.array(image1)
                    img2_array = np.array(image2)
                    
                    # Ensure images are in BGR format for OpenCV
                    if len(img1_array.shape) == 3:
                        img1_array = cv2.cvtColor(img1_array, cv2.COLOR_RGB2BGR)
                    if len(img2_array.shape) == 3:
                        img2_array = cv2.cvtColor(img2_array, cv2.COLOR_RGB2BGR)
                    
                    # Perform equipment analysis
                    analysis_results = analyzer.analyze_equipment_differences(
                        img1_array, img2_array,
                        uploaded_img1.name, uploaded_img2.name
                    )
                    
                    # Display results
                    st.success("✅ P&ID equipment analysis completed successfully!")
                    
                    # Equipment summary metrics
                    summary = analysis_results['summary_statistics']
                    equipment_stats = analysis_results.get('equipment_statistics', {})
                    safety_stats = analysis_results.get('safety_statistics', {})
                    
                    # Main metrics row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Equipment Differences", summary['total_differences_found'])
                    with col2:
                        st.metric("Equipment Items", summary['equipment_differences'])
                    with col3:
                        st.metric("Pipe Differences", summary['pipe_differences'])
                    with col4:
                        st.metric("Instrument Differences", summary['instrument_differences'])
                    with col5:
                        st.metric("Critical Safety Items", safety_stats.get('CRITICAL', 0))
                    
                    # Equipment type breakdown
                    if equipment_stats:
                        st.subheader("🏭 Equipment Type Distribution")
                        eq_cols = st.columns(min(len(equipment_stats), 5))
                        for i, (eq_type, count) in enumerate(equipment_stats.items()):
                            with eq_cols[i % 5]:
                                st.metric(f"{eq_type.title()}", count)
                    
                    # Enhanced safety alerts with equipment-specific warnings
                    critical_count = safety_stats.get('CRITICAL', 0)
                    high_count = safety_stats.get('HIGH', 0)
                    
                    if critical_count > 0:
                        st.error(f"🚨 {critical_count} CRITICAL equipment differences detected!")
                        st.error("⚠️ **Immediate Review Required**: These may include valve changes, safety system modifications, or critical control equipment.")
                    elif high_count > 0:
                        st.warning(f"⚠️ {high_count} HIGH severity equipment differences detected.")
                        st.warning("These may include pump modifications, vessel changes, or instrument updates.")
                    else:
                        st.info("✅ No critical equipment safety differences detected.")
                    
                    # Display enhanced marked image
                    st.subheader("🎯 P&ID with Equipment Differences Highlighted")
                    marked_img_rgb = cv2.cvtColor(analysis_results['marked_image'], cv2.COLOR_BGR2RGB)
                    st.image(marked_img_rgb, caption="P&ID Equipment Differences with Type Recognition", use_container_width=True)
                    
                    # Equipment-specific visualizations
                    if analysis_results['all_differences']:
                        st.subheader("📊 Equipment Analysis Visualizations")
                        visualizations = create_equipment_visualizations(analysis_results)
                        
                        if visualizations:
                            vis_col1, vis_col2 = st.columns(2)
                            
                            with vis_col1:
                                if 'equipment_pie' in visualizations:
                                    st.plotly_chart(visualizations['equipment_pie'], use_container_width=True)
                                if 'safety_pie' in visualizations:
                                    st.plotly_chart(visualizations['safety_pie'], use_container_width=True)
                            
                            with vis_col2:
                                if 'equipment_scatter' in visualizations:
                                    st.plotly_chart(visualizations['equipment_scatter'], use_container_width=True)
                                if 'equipment_location_map' in visualizations:
                                    st.plotly_chart(visualizations['equipment_location_map'], use_container_width=True)
                    else:
                        st.info("📊 No equipment differences found to visualize.")
                    
                    # Detailed equipment analysis tables
                    st.subheader("📋 Detailed Equipment Analysis")
                    
                    # Tabbed interface for different equipment types
                    tab1, tab2, tab3, tab4 = st.tabs(["All Equipment", "Major Equipment", "Piping Systems", "Instruments"])
                    
                    with tab1:
                        if analysis_results['all_differences']:
                            df_all = pd.DataFrame(analysis_results['all_differences'])
                            st.dataframe(df_all, use_container_width=True)
                        else:
                            st.info("No equipment differences detected.")
                    
                    with tab2:
                        if analysis_results['equipment_differences']:
                            df_equipment = pd.DataFrame(analysis_results['equipment_differences'])
                            st.dataframe(df_equipment, use_container_width=True)
                        else:
                            st.info("No major equipment differences detected.")
                    
                    with tab3:
                        if analysis_results['pipe_differences']:
                            df_pipes = pd.DataFrame(analysis_results['pipe_differences'])
                            st.dataframe(df_pipes, use_container_width=True)
                        else:
                            st.info("No piping differences detected.")
                    
                    with tab4:
                        if analysis_results['instrument_differences']:
                            df_instruments = pd.DataFrame(analysis_results['instrument_differences'])
                            st.dataframe(df_instruments, use_container_width=True)
                        else:
                            st.info("No instrument differences detected.")
                    
                    # Equipment-specific insights
                    if equipment_stats:
                        st.subheader("🔍 Equipment-Specific Safety Insights")
                        
                        for eq_type, count in equipment_stats.items():
                            with st.expander(f"{eq_type.title()} Analysis ({count} differences)"):
                                eq_data = [d for d in analysis_results['all_differences'] if d['Equipment_Type'] == eq_type]
                                critical_items = [d for d in eq_data if d['Safety_Level'] in ['CRITICAL', 'HIGH']]
                                
                                st.write(f"**Safety Analysis**: {len(critical_items)} critical/high severity items")
                                
                                if eq_type == 'pump':
                                    st.write("**Safety Considerations**: Pump changes affect process flow and pressure. Verify impeller compatibility and motor sizing.")
                                elif eq_type == 'valve':
                                    st.write("**Safety Considerations**: Valve modifications are CRITICAL. Verify fail-safe positions and control logic.")
                                elif eq_type == 'vessel':
                                    st.write("**Safety Considerations**: Vessel changes affect pressure containment. Review pressure relief sizing.")
                                elif eq_type == 'pipe':
                                    st.write("**Safety Considerations**: Piping changes affect flow paths. Verify pressure drop and material compatibility.")
                                elif eq_type == 'instrument':
                                    st.write("**Safety Considerations**: Instrument changes affect monitoring and control. Verify safety function integrity.")
                                
                                if eq_data:
                                    eq_df = pd.DataFrame(eq_data)
                                    key_cols = ['Difference_ID', 'Safety_Level', 'Confidence_Score', 'Description']
                                    st.dataframe(eq_df[key_cols], use_container_width=True)
                    
                    # Enhanced download section
                    st.subheader("💾 Download Equipment Analysis Results")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        # Enhanced CSV with equipment data
                        csv_bytes = create_equipment_csv(analysis_results)
                        st.download_button(
                            label="📊 Download Equipment Analysis CSV",
                            data=csv_bytes,
                            file_name=f"pid_equipment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with download_col2:
                        # Enhanced marked image
                        img_bytes = create_downloadable_marked_image(analysis_results['marked_image'])
                        if img_bytes:
                            st.download_button(
                                label="🖼️ Download Marked P&ID",
                                data=img_bytes,
                                file_name=f"pid_equipment_marked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                    
                    with download_col3:
                        # Equipment analysis report
                        report_text = generate_equipment_report(analysis_results)
                        st.download_button(
                            label="📄 Download Equipment Report",
                            data=report_text,
                            file_name=f"pid_equipment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during equipment analysis: {str(e)}")
                    logger.error(f"Equipment analysis error: {str(e)}")
                    st.error("Please check your P&ID images and try again. Ensure both images contain recognizable equipment symbols.")

def create_equipment_csv(analysis_results: Dict[str, Any]) -> bytes:
    """
    Create comprehensive CSV with equipment analysis
    """
    try:
        all_diffs = analysis_results.get('all_differences', [])
        summary = analysis_results['summary_statistics']
        metadata = analysis_results['metadata']
        equipment_stats = analysis_results.get('equipment_statistics', {})
        safety_stats = analysis_results.get('safety_statistics', {})
        
        # Create DataFrame with equipment-specific columns
        if all_diffs:
            df = pd.DataFrame(all_diffs)
        else:
            df = pd.DataFrame(columns=[
                'Difference_ID', 'Equipment_Type', 'Safety_Level', 'Confidence_Score',
                'X_Coordinate', 'Y_Coordinate', 'Width', 'Height', 'Area_Pixels',
                'Aspect_Ratio', 'Solidity', 'Circularity', 'Vertices', 'Perimeter',
                'Description', 'Percentage_of_Image'
            ])
        
        # Enhanced summary with equipment statistics
        summary_data = [
            ['EQUIPMENT_ANALYSIS_SUMMARY', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Analysis_Timestamp', metadata['analysis_timestamp'], '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Equipment_Recognition_Enabled', metadata.get('equipment_recognition_enabled', False), '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Total_Equipment_Differences', summary['total_differences_found'], '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Major_Equipment', summary['equipment_differences'], '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Piping_Systems', summary['pipe_differences'], '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['Instruments', summary['instrument_differences'], '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['EQUIPMENT_TYPE_BREAKDOWN', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
        ]
        
        # Add equipment type statistics
        for eq_type, count in equipment_stats.items():
            summary_data.append([f'{eq_type.title()}_Count', count, '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        summary_data.extend([
            ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['SAFETY_LEVEL_BREAKDOWN', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
        ])
        
        # Add safety statistics
        for level, count in safety_stats.items():
            summary_data.append([f'{level}_Count', count, '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        summary_data.extend([
            ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['DETAILED_EQUIPMENT_ANALYSIS', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
        ])
        
        summary_df = pd.DataFrame(summary_data, columns=df.columns)
        
        # Combine summary and detailed data
        final_df = pd.concat([summary_df, df], ignore_index=True)
        
        # Convert to CSV bytes
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating equipment CSV: {str(e)}")
        empty_df = pd.DataFrame({'Error': [f'Error creating equipment CSV: {str(e)}']})
        csv_buffer = BytesIO()
        empty_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

def generate_equipment_report(analysis_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive equipment analysis report
    """
    try:
        metadata = analysis_results['metadata']
        summary = analysis_results['summary_statistics']
        equipment_stats = analysis_results.get('equipment_statistics', {})
        safety_stats = analysis_results.get('safety_statistics', {})
        all_diffs = analysis_results.get('all_differences', [])
        
        report = f"""
P&ID EQUIPMENT DIFFERENCE ANALYSIS REPORT
{'='*50}

ANALYSIS METADATA:
- Analysis Date: {metadata['analysis_timestamp']}
- Reference P&ID: {metadata['image1_path']}
- Updated P&ID: {metadata['image2_path']}
- Equipment Recognition: {metadata.get('equipment_recognition_enabled', False)}
- Image Dimensions: {metadata['image_dimensions']}

EQUIPMENT SUMMARY:
- Total Equipment Differences: {summary['total_differences_found']}
- Major Equipment Changes: {summary['equipment_differences']}
- Piping System Changes: {summary['pipe_differences']}
- Instrument Changes: {summary['instrument_differences']}
- Percentage of Image Changed: {summary['percentage_changed']}%

EQUIPMENT TYPE BREAKDOWN:
{'='*30}
"""
        
        if equipment_stats:
            for eq_type, count in equipment_stats.items():
                report += f"- {eq_type.title()}: {count} differences\n"
        
        report += f"""

SAFETY CLASSIFICATION:
{'='*25}
- CRITICAL (Immediate Action): {safety_stats.get('CRITICAL', 0)}
- HIGH (Review Required): {safety_stats.get('HIGH', 0)}
- MEDIUM (Monitor): {safety_stats.get('MEDIUM', 0)}
- LOW (Document): {safety_stats.get('LOW', 0)}

DETAILED EQUIPMENT ANALYSIS:
{'='*35}
"""
        
        if all_diffs:
            for diff in all_diffs:
                report += f"""
Equipment ID: {diff['Difference_ID']}
├── Type: {diff['Equipment_Type'].upper()}
├── Safety Level: {diff['Safety_Level']}
├── Confidence: {diff['Confidence_Score']:.3f}
├── Location: ({diff['X_Coordinate']}, {diff['Y_Coordinate']})
├── Dimensions: {diff['Width']} x {diff['Height']} pixels
├── Area: {diff['Area_Pixels']} pixels
└── Description: {diff['Description']}
"""
        
        report += f"""

SAFETY ASSESSMENT BY EQUIPMENT TYPE:
{'='*40}
"""
        
        for eq_type in equipment_stats.keys():
            eq_diffs = [d for d in all_diffs if d['Equipment_Type'] == eq_type]
            critical_count = len([d for d in eq_diffs if d['Safety_Level'] in ['CRITICAL', 'HIGH']])
            
            report += f"""
{eq_type.upper()} EQUIPMENT ({len(eq_diffs)} changes):
- Safety-Critical Items: {critical_count}
"""
            
            if eq_type == 'pump':
                report += "- Safety Impact: Process flow and pressure changes\n"
                report += "- Review: Verify impeller compatibility and motor sizing\n"
            elif eq_type == 'valve':
                report += "- Safety Impact: CRITICAL - Flow control and isolation\n"
                report += "- Review: Verify fail-safe positions and control logic\n"
            elif eq_type == 'vessel':
                report += "- Safety Impact: Pressure containment and process safety\n"
                report += "- Review: Check pressure relief valve sizing\n"
            elif eq_type == 'pipe':
                report += "- Safety Impact: Flow paths and system integrity\n"
                report += "- Review: Verify pressure drop and material compatibility\n"
            elif eq_type == 'instrument':
                report += "- Safety Impact: Monitoring and control capabilities\n"
                report += "- Review: Verify safety function integrity (SIL ratings)\n"
            
            report += "\n"
        
        critical_equipment = [d for d in all_diffs if d['Safety_Level'] == 'CRITICAL']
        if critical_equipment:
            report += f"""
CRITICAL EQUIPMENT REQUIRING IMMEDIATE ATTENTION:
{'='*50}
"""
            for eq in critical_equipment:
                report += f"- {eq['Equipment_Type'].upper()} (ID: {eq['Difference_ID']}): {eq['Description']}\n"
        
        report += f"""

RECOMMENDATIONS:
{'='*15}
1. IMMEDIATE ACTIONS:
   - Review all CRITICAL equipment changes immediately
   - Verify that valve modifications include proper fail-safe design
   - Confirm pressure vessel changes comply with ASME codes
   - Update safety instrumentation function (SIF) documentation

2. ENGINEERING REVIEW:
   - Conduct HAZOP review for modified equipment
   - Update P&ID revision control and numbering
   - Verify equipment specifications and datasheets
   - Check material compatibility and process conditions

3. SAFETY MANAGEMENT:
   - Update safety case documentation
   - Review and update operating procedures
   - Conduct pre-startup safety review (PSSR)
   - Ensure compliance with process safety standards

4. DOCUMENTATION:
   - Update equipment database and maintenance procedures
   - Revise emergency response procedures if needed
   - Document all changes in management of change (MOC) system
   - Update training materials for operators

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Tool: P&ID Equipment Difference Analyzer
For Industrial Safety Applications
"""
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating equipment report: {str(e)}")
        return f"Error generating equipment report: {str(e)}"

def create_downloadable_marked_image(marked_image: np.ndarray) -> bytes:
    """
    Convert marked image to downloadable bytes
    """
    try:
        # Convert BGR to RGB for PIL
        rgb_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes buffer
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating downloadable image: {str(e)}")
        return b''

if __name__ == "__main__":
    main()
