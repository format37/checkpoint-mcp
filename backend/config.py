"""
Configuration module for the Knowledge Evaluation MCP Server.

Centralizes all configuration settings for paths, plotting, and analysis.
"""

import os
from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Base data directory
DATA_DIR = Path("/app/data")

# Plots directory
PLOTS_DIR = DATA_DIR / "plots"

# Ensure plots directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Plotting Configuration
# ============================================================================

# Plot quality settings
PLOT_DPI = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Color scheme for visualizations
COLORS = {
    "width": "#3498db",  # Blue for width scores
    "depth": "#e74c3c",  # Red/orange for depth scores
    "primary": "#2c3e50",  # Dark gray for text
    "secondary": "#95a5a6",  # Light gray for gridlines
    "success": "#27ae60",  # Green for positive trends
    "warning": "#f39c12",  # Orange for warnings
    "danger": "#e74c3c",  # Red for issues
}

# Figure sizes (width, height in inches)
FIGURE_SIZES = {
    "single_topic": (12, 6),  # For plot_topic()
    "all_topics": (14, 10),  # For plot_all()
}

# Image export settings
IMAGE_SETTINGS = {
    "format": "png",
    "dpi": PLOT_DPI,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
}

# Preview image settings (for MCP image display)
PREVIEW_MAX_WIDTH = 1200
PREVIEW_MAX_HEIGHT = 900
PREVIEW_FORMAT = "jpeg"
PREVIEW_QUALITY = 85

# ============================================================================
# Analysis Configuration
# ============================================================================

# Threshold for determining if there's a significant trend (improvement/decline)
TREND_THRESHOLD = 0.05  # 5% change

# Score thresholds for gap analysis
SCORE_THRESHOLDS = {
    "low": 0.5,  # Scores below this indicate knowledge gaps
    "medium": 0.7,  # Scores between low and medium need attention
    "high": 0.85,  # Scores above this indicate strong knowledge
}

# Minimum number of evaluations for trend analysis
MIN_EVALUATIONS_FOR_TREND = 2

# ============================================================================
# CSV Column Names
# ============================================================================

CSV_COLUMNS = {
    "date": "date",
    "answer": "answer",
    "width_score": "width_score",
    "width_explanation": "width_explanation",
    "depth_score": "depth_score",
    "depth_explanation": "depth_explanation",
}
