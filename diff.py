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

class IndustrialImageDifferenceAnalyzer:
    """
    Industrial-grade P&ID image difference analyzer for safety-critical applications
    Designed to detect changes in piping and instrumentation diagrams
    """
    
    def __init__(self, sensitivity: float = 30.0, min_area: int = 100):
        self.sensitivity = sensitivity  # Difference threshold (0-255)
        self.min_area = min_area       # Minimum area for difference detection
        self.safety_keywords = [
            'relief', 'psv', 'safety', 'emergency', 'alarm', 'shutdown',
            'pressure', 'temperature', 'flow', 'level', 'trip', 'interlock'
        ]
        
    def analyze_differences(self, img1: np.ndarray, img2: np.ndarray, 
                          img1_path: str = "Image 1", img2_path: str = "Image 2") -> Dict[str, Any]:
        """
        Comprehensive difference analysis between two P&ID images
        """
        try:
            logger.info("Starting industrial image difference analysis")
            
            # Ensure images are the same size
            if img1.shape != img2.shape:
                # Resize img2 to match img1
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                logger.warning("Images resized to match dimensions")
            
            # Convert to grayscale for analysis
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Apply threshold to create binary difference image
            _, thresh = cv2.threshold(diff, self.sensitivity, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up noise and merge nearby differences
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Dilate to merge adjacent differences
            dilate_kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(cleaned, dilate_kernel, iterations=2)
            
            # Find contours of differences
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.min_area]
            
            # Create marked image (copy of img2)
            marked_img = img2.copy()
            
            # Analyze each difference region
            difference_regions = []
            total_diff_area = 0
            
            for i, contour in enumerate(valid_contours):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                total_diff_area += area
                
                # Calculate difference statistics for this region
                roi_diff = diff[y:y+h, x:x+w]
                mean_intensity = np.mean(roi_diff)
                max_intensity = np.max(roi_diff)
                
                # Classify difference severity
                severity = self._classify_difference_severity(mean_intensity, area)
                
                # Mark the difference on the image
                color = self._get_severity_color(severity)
                cv2.rectangle(marked_img, (x, y), (x+w, y+h), color, 3)
                
                # Add label
                label = f"D{i+1}-{severity}"
                cv2.putText(marked_img, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Extract region details
                region_info = {
                    'Difference_ID': i + 1,
                    'X_Coordinate': int(x),
                    'Y_Coordinate': int(y),
                    'Width': int(w),
                    'Height': int(h),
                    'Area_Pixels': int(area),
                    'Bounding_Box': f"({x},{y})-({x+w},{y+h})",
                    'Mean_Intensity_Diff': round(mean_intensity, 2),
                    'Max_Intensity_Diff': int(max_intensity),
                    'Severity': severity,
                    'Safety_Critical': self._assess_safety_criticality(x, y, w, h, img1.shape),
                    'Percentage_of_Image': round((area / (img1.shape[0] * img1.shape[1])) * 100, 4)
                }
                
                difference_regions.append(region_info)
            
            # Calculate overall statistics
            total_pixels = img1.shape[0] * img1.shape[1]
            diff_percentage = (total_diff_area / total_pixels) * 100
            
            # Create comprehensive analysis results
            analysis_results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'image1_path': img1_path,
                    'image2_path': img2_path,
                    'image_dimensions': f"{img1.shape[1]}x{img1.shape[0]}",
                    'sensitivity_threshold': self.sensitivity,
                    'minimum_area_threshold': self.min_area
                },
                'summary_statistics': {
                    'total_differences_found': len(difference_regions),
                    'total_difference_area': int(total_diff_area),
                    'percentage_changed': round(diff_percentage, 4),
                    'critical_differences': len([d for d in difference_regions if d['Safety_Critical']]),
                    'high_severity_differences': len([d for d in difference_regions if d['Severity'] == 'HIGH']),
                    'medium_severity_differences': len([d for d in difference_regions if d['Severity'] == 'MEDIUM']),
                    'low_severity_differences': len([d for d in difference_regions if d['Severity'] == 'LOW'])
                },
                'difference_regions': difference_regions,
                'marked_image': marked_img,
                'difference_mask': thresh,
                'raw_difference': diff
            }
            
            logger.info(f"Analysis completed: {len(difference_regions)} differences found")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in difference analysis: {str(e)}")
            raise
    
    def _classify_difference_severity(self, mean_intensity: float, area: int) -> str:
        """
        Classify difference severity based on intensity and area
        """
        if mean_intensity > 150 or area > 5000:
            return "HIGH"
        elif mean_intensity > 80 or area > 1000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_severity_color(self, severity: str) -> Tuple[int, int, int]:
        """
        Get BGR color for severity level
        """
        colors = {
            "HIGH": (0, 0, 255),    # Red
            "MEDIUM": (0, 165, 255),  # Orange
            "LOW": (0, 255, 255)     # Yellow
        }
        return colors.get(severity, (0, 255, 0))  # Default green
    
    def _assess_safety_criticality(self, x: int, y: int, w: int, h: int, 
                                 image_shape: Tuple[int, int, int]) -> bool:
        """
        Assess if difference region is in a safety-critical area
        Based on location heuristics for P&ID layouts
        """
        # Consider differences near edges or corners as potentially critical
        # (where safety instrumentation is typically located)
        img_height, img_width = image_shape[:2]
        
        # Check if near edges (within 10% of image dimensions)
        edge_threshold_x = img_width * 0.1
        edge_threshold_y = img_height * 0.1
        
        near_edge = (x < edge_threshold_x or 
                    (x + w) > (img_width - edge_threshold_x) or
                    y < edge_threshold_y or 
                    (y + h) > (img_height - edge_threshold_y))
        
        # Large differences are potentially critical
        large_difference = (w * h) > (img_width * img_height * 0.01)  # > 1% of image
        
        return near_edge or large_difference
    
    def generate_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive textual report
        """
        try:
            metadata = analysis_results['metadata']
            summary = analysis_results['summary_statistics']
            regions = analysis_results['difference_regions']
            
            report = f"""
INDUSTRIAL P&ID IMAGE DIFFERENCE ANALYSIS REPORT
{'='*60}

ANALYSIS METADATA:
- Analysis Date: {metadata['analysis_timestamp']}
- Image 1: {metadata['image1_path']}
- Image 2: {metadata['image2_path']}
- Image Dimensions: {metadata['image_dimensions']}
- Sensitivity Threshold: {metadata['sensitivity_threshold']}
- Minimum Area Threshold: {metadata['minimum_area_threshold']} pixels

SUMMARY STATISTICS:
- Total Differences Found: {summary['total_differences_found']}
- Total Changed Area: {summary['total_difference_area']} pixels
- Percentage of Image Changed: {summary['percentage_changed']}%
- Safety-Critical Differences: {summary['critical_differences']}
- High Severity Differences: {summary['high_severity_differences']}
- Medium Severity Differences: {summary['medium_severity_differences']}
- Low Severity Differences: {summary['low_severity_differences']}

DETAILED DIFFERENCE ANALYSIS:
{'='*40}
"""
            
            if regions:
                for region in regions:
                    report += f"""
Difference ID: {region['Difference_ID']}
├── Location: ({region['X_Coordinate']}, {region['Y_Coordinate']})
├── Dimensions: {region['Width']} x {region['Height']} pixels
├── Area: {region['Area_Pixels']} pixels ({region['Percentage_of_Image']}% of image)
├── Severity: {region['Severity']}
├── Mean Intensity Difference: {region['Mean_Intensity_Diff']}
├── Max Intensity Difference: {region['Max_Intensity_Diff']}
└── Safety Critical: {'YES' if region['Safety_Critical'] else 'NO'}
"""
            else:
                report += "\nNo differences detected between the images.\n"
            
            report += f"""

SAFETY ASSESSMENT:
{'='*20}
"""
            
            critical_count = summary['critical_differences']
            if critical_count > 0:
                report += f"⚠️  {critical_count} safety-critical differences detected!\n"
                report += "Recommended Actions:\n"
                report += "- Immediate review of all critical differences\n"
                report += "- Verify changes comply with safety standards\n"
                report += "- Update safety documentation if needed\n"
            else:
                report += "✅ No safety-critical differences detected.\n"
            
            high_severity = summary['high_severity_differences']
            if high_severity > 0:
                report += f"\n⚠️  {high_severity} high-severity differences require attention.\n"
            
            report += f"""

RECOMMENDATIONS:
- Review all HIGH severity differences immediately
- Verify that changes are authorized and documented
- Update P&ID revision control as needed
- Consider impact on safety systems and procedures
- Ensure compliance with industrial safety standards

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"

def create_comparison_visualizations(analysis_results: Dict[str, Any]) -> Dict[str, any]:
    """
    Create visualizations for the difference analysis
    """
    try:
        regions = analysis_results['difference_regions']
        summary = analysis_results['summary_statistics']
        
        visualizations = {}
        
        if regions:
            # Severity distribution pie chart
            severity_counts = {
                'HIGH': summary['high_severity_differences'],
                'MEDIUM': summary['medium_severity_differences'],
                'LOW': summary['low_severity_differences']
            }
            
            # Filter out zero values
            severity_counts = {k: v for k, v in severity_counts.items() if v > 0}
            
            if severity_counts:
                severity_fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Difference Severity Distribution",
                    color_discrete_map={
                        'HIGH': '#FF0000',
                        'MEDIUM': '#FFA500', 
                        'LOW': '#FFFF00'
                    }
                )
                visualizations['severity_pie'] = severity_fig
            
            # Area distribution bar chart
            df_regions = pd.DataFrame(regions)
            area_fig = px.bar(
                df_regions, x='Difference_ID', y='Area_Pixels',
                color='Severity',
                title="Difference Areas by ID",
                color_discrete_map={
                    'HIGH': '#FF0000',
                    'MEDIUM': '#FFA500',
                    'LOW': '#FFFF00'
                }
            )
            visualizations['area_bar'] = area_fig
            
            # Scatter plot of difference locations
            scatter_fig = go.Figure()
            
            for severity in ['HIGH', 'MEDIUM', 'LOW']:
                severity_data = df_regions[df_regions['Severity'] == severity]
                if not severity_data.empty:
                    scatter_fig.add_trace(go.Scatter(
                        x=severity_data['X_Coordinate'],
                        y=severity_data['Y_Coordinate'],
                        mode='markers',
                        marker=dict(
                            size=severity_data['Area_Pixels'] / 100,
                            color={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}[severity],
                            opacity=0.7
                        ),
                        name=f'{severity} Severity',
                        text=severity_data['Difference_ID'],
                        hovertemplate='<b>Difference %{text}</b><br>' +
                                    'Location: (%{x}, %{y})<br>' +
                                    'Area: %{marker.size:.0f} pixels<extra></extra>'
                    ))
            
            scatter_fig.update_layout(
                title="Difference Locations on Image",
                xaxis_title="X Coordinate",
                yaxis_title="Y Coordinate",
                yaxis=dict(autorange="reversed")  # Invert Y-axis to match image coordinates
            )
            visualizations['location_scatter'] = scatter_fig
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        return {}

def create_downloadable_csv(analysis_results: Dict[str, Any]) -> bytes:
    """
    Create downloadable CSV file with difference analysis results
    """
    try:
        regions = analysis_results['difference_regions']
        summary = analysis_results['summary_statistics']
        metadata = analysis_results['metadata']
        
        # Create comprehensive DataFrame
        if regions:
            df = pd.DataFrame(regions)
        else:
            # Create empty DataFrame with expected columns if no differences
            df = pd.DataFrame(columns=[
                'Difference_ID', 'X_Coordinate', 'Y_Coordinate', 'Width', 'Height',
                'Area_Pixels', 'Bounding_Box', 'Mean_Intensity_Diff', 'Max_Intensity_Diff',
                'Severity', 'Safety_Critical', 'Percentage_of_Image'
            ])
        
        # Add summary information as header rows
        summary_data = [
            ['ANALYSIS_SUMMARY', '', '', '', '', '', '', '', '', '', '', ''],
            ['Analysis_Timestamp', metadata['analysis_timestamp'], '', '', '', '', '', '', '', '', '', ''],
            ['Image_1', metadata['image1_path'], '', '', '', '', '', '', '', '', '', ''],
            ['Image_2', metadata['image2_path'], '', '', '', '', '', '', '', '', '', ''],
            ['Image_Dimensions', metadata['image_dimensions'], '', '', '', '', '', '', '', '', '', ''],
            ['Total_Differences', summary['total_differences_found'], '', '', '', '', '', '', '', '', '', ''],
            ['Total_Changed_Area', summary['total_difference_area'], '', '', '', '', '', '', '', '', '', ''],
            ['Percentage_Changed', f"{summary['percentage_changed']}%", '', '', '', '', '', '', '', '', '', ''],
            ['Critical_Differences', summary['critical_differences'], '', '', '', '', '', '', '', '', '', ''],
            ['High_Severity', summary['high_severity_differences'], '', '', '', '', '', '', '', '', '', ''],
            ['Medium_Severity', summary['medium_severity_differences'], '', '', '', '', '', '', '', '', '', ''],
            ['Low_Severity', summary['low_severity_differences'], '', '', '', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', '', '', '', '', ''],  # Empty row
            ['DETAILED_DIFFERENCES', '', '', '', '', '', '', '', '', '', '', '']
        ]
        
        summary_df = pd.DataFrame(summary_data, columns=df.columns)
        
        # Combine summary and detailed data
        final_df = pd.concat([summary_df, df], ignore_index=True)
        
        # Convert to CSV bytes
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        
        return csv_bytes
        
    except Exception as e:
        logger.error(f"Error creating CSV: {str(e)}")
        # Return empty CSV on error
        empty_df = pd.DataFrame({'Error': [f'Error creating CSV: {str(e)}']})
        csv_buffer = BytesIO()
        empty_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

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
        img_bytes = img_buffer.getvalue()
        
        return img_bytes
        
    except Exception as e:
        logger.error(f"Error creating downloadable image: {str(e)}")
        return b''

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Industrial P&ID Image Difference Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Industrial P&ID Image Difference Analyzer")
    st.markdown("**Safety-Critical Image Comparison for Process Industry Applications**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Analysis Configuration")
    sensitivity = st.sidebar.slider(
        "Difference Sensitivity Threshold", 
        min_value=10.0, max_value=100.0, value=30.0, step=5.0,
        help="Lower values detect smaller differences (more sensitive)"
    )
    min_area = st.sidebar.slider(
        "Minimum Difference Area (pixels)", 
        min_value=50, max_value=1000, value=100, step=50,
        help="Minimum area to consider as a meaningful difference"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Analysis Features")
    st.sidebar.markdown("- **Safety-critical difference detection**")
    st.sidebar.markdown("- **Severity classification (HIGH/MEDIUM/LOW)**")
    st.sidebar.markdown("- **Comprehensive reporting**")
    st.sidebar.markdown("- **Downloadable CSV and marked images**")
    st.sidebar.markdown("- **Interactive visualizations**")
    
    # File upload section
    st.header("📁 Upload P&ID Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Reference Image (Image 1)")
        uploaded_img1 = st.file_uploader(
            "Choose reference P&ID image", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="img1"
        )
        
        if uploaded_img1:
            image1 = Image.open(uploaded_img1)
            st.image(image1, caption="Reference Image", use_container_width=True)
    
    with col2:
        st.subheader("📄 Comparison Image (Image 2)")
        uploaded_img2 = st.file_uploader(
            "Choose comparison P&ID image", 
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="img2"
        )
        
        if uploaded_img2:
            image2 = Image.open(uploaded_img2)
            st.image(image2, caption="Comparison Image", use_container_width=True)
    
    # Analysis section
    if uploaded_img1 and uploaded_img2:
        st.header("🔬 Difference Analysis")
        
        if st.button("🚀 Analyze Differences", type="primary"):
            with st.spinner("Analyzing differences... This may take a moment for large images."):
                try:
                    # Initialize analyzer
                    analyzer = IndustrialImageDifferenceAnalyzer(
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
                    
                    # Perform analysis
                    analysis_results = analyzer.analyze_differences(
                        img1_array, img2_array,
                        uploaded_img1.name, uploaded_img2.name
                    )
                    
                    # Display results
                    st.success("✅ Analysis completed successfully!")
                    
                    # Summary metrics
                    summary = analysis_results['summary_statistics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Differences", summary['total_differences_found'])
                    with col2:
                        st.metric("% Changed", f"{summary['percentage_changed']:.2f}%")
                    with col3:
                        st.metric("Critical Differences", summary['critical_differences'])
                    with col4:
                        st.metric("High Severity", summary['high_severity_differences'])
                    
                    # Safety alerts
                    if summary['critical_differences'] > 0:
                        st.error(f"⚠️ {summary['critical_differences']} safety-critical differences detected! Immediate review required.")
                    elif summary['high_severity_differences'] > 0:
                        st.warning(f"⚠️ {summary['high_severity_differences']} high-severity differences require attention.")
                    else:
                        st.info("✅ No critical differences detected.")
                    
                    # Display marked image
                    st.subheader("🎯 Marked Differences")
                    marked_img_rgb = cv2.cvtColor(analysis_results['marked_image'], cv2.COLOR_BGR2RGB)
                    st.image(marked_img_rgb, caption="Differences Marked on Comparison Image", use_container_width=True)
                    
                    # Visualizations
                    if analysis_results['difference_regions']:
                        st.subheader("📊 Analysis Visualizations")
                        visualizations = create_comparison_visualizations(analysis_results)
                        
                        if visualizations:
                            vis_col1, vis_col2 = st.columns(2)
                            
                            with vis_col1:
                                if 'severity_pie' in visualizations:
                                    st.plotly_chart(visualizations['severity_pie'], use_container_width=True)
                                if 'area_bar' in visualizations:
                                    st.plotly_chart(visualizations['area_bar'], use_container_width=True)
                            
                            with vis_col2:
                                if 'location_scatter' in visualizations:
                                    st.plotly_chart(visualizations['location_scatter'], use_container_width=True)
                    else:
                        st.info("📊 No differences found to visualize.")
                    
                    # Detailed results table
                    if analysis_results['difference_regions']:
                        st.subheader("📋 Detailed Difference Analysis")
                        df_results = pd.DataFrame(analysis_results['difference_regions'])
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.info("📋 No differences detected between the images.")
                    
                    # Generate report
                    st.subheader("📄 Detailed Report")
                    report_text = analyzer.generate_detailed_report(analysis_results)
                    st.text_area("Analysis Report", report_text, height=400)
                    
                    # Download section
                    st.subheader("💾 Download Results")
                    
                    download_col1, download_col2, download_col3 = st.columns(3)
                    
                    with download_col1:
                        # CSV download
                        csv_bytes = create_downloadable_csv(analysis_results)
                        st.download_button(
                            label="📊 Download CSV Report",
                            data=csv_bytes,
                            file_name=f"pid_difference_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with download_col2:
                        # Marked image download
                        img_bytes = create_downloadable_marked_image(analysis_results['marked_image'])
                        if img_bytes:
                            st.download_button(
                                label="🖼️ Download Marked Image",
                                data=img_bytes,
                                file_name=f"marked_differences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                    
                    with download_col3:
                        # Text report download
                        st.download_button(
                            label="📄 Download Text Report",
                            data=report_text,
                            file_name=f"pid_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
                    st.error("Please check your images and try again. Ensure both images are valid and readable.")
    
    # Usage instructions
    with st.expander("📖 Usage Instructions"):
        st.markdown("""
        ### How to Use This Tool:
        
        1. **Upload Images**: Select two P&ID images to compare
           - Image 1: Reference/original P&ID
           - Image 2: Updated/modified P&ID
        
        2. **Configure Settings**: Adjust sensitivity and minimum area thresholds
           - Lower sensitivity detects smaller changes
           - Higher minimum area filters out noise
        
        3. **Run Analysis**: Click 'Analyze Differences' to start processing
        
        4. **Review Results**: 
           - Check summary metrics and safety alerts
           - Examine marked differences on the image
           - Review detailed analysis table
        
        5. **Download Results**:
           - CSV file for further analysis
           - Marked image for documentation
           - Text report for records
        
        ### Safety Features:
        - Automatically identifies safety-critical differences
        - Classifies severity levels (HIGH/MEDIUM/LOW)
        - Provides safety recommendations
        - Generates audit-ready reports
        
        ### Supported Formats:
        - PNG, JPG, JPEG, TIFF, BMP
        - Recommended: High-resolution images for better accuracy
        """)
    
    # Technical information
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        ### Analysis Algorithm:
        1. **Image Preprocessing**: Resize and convert to grayscale
        2. **Difference Detection**: Calculate absolute pixel differences
        3. **Thresholding**: Apply configurable sensitivity threshold
        4. **Morphological Operations**: Clean noise and merge nearby changes
        5. **Contour Analysis**: Identify and classify difference regions
        6. **Safety Assessment**: Evaluate criticality based on location and size
        
        ### Severity Classification:
        - **HIGH**: Mean intensity > 150 OR area > 5000 pixels
        - **MEDIUM**: Mean intensity > 80 OR area > 1000 pixels  
        - **LOW**: Below medium thresholds
        
        ### Safety Criticality:
        - Differences near image edges (typical safety instrument locations)
        - Large differences (> 1% of total image area)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Industrial P&ID Image Difference Analyzer** | "
        "Developed for Safety-Critical Industrial Applications | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
