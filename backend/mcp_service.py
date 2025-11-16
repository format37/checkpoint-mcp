import logging
import os
import json
from datetime import datetime
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pydantic import Field
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
from mcp.server.fastmcp import Image as MCPImage

from sentry_utils import with_sentry_tracing
from topics import TOPICS
from mcp_image_utils import to_mcp_image
from config import (
    DATA_DIR, PLOTS_DIR, PLOT_DPI, PLOT_STYLE, COLORS, FIGURE_SIZES,
    IMAGE_SETTINGS, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT,
    PREVIEW_FORMAT, PREVIEW_QUALITY, TREND_THRESHOLD, SCORE_THRESHOLDS,
    MIN_EVALUATIONS_FOR_TREND
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_checkpoint_tools(local_mcp_instance):
    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("get_progress")
    def get_progress() -> str:
        """
        Get overall learning progress for all topics in a single call.

        This function provides a comprehensive dashboard view of learning analytics
        across all available topics, making it ideal for:
        - Quick progress overview in voice interactions
        - Identifying which topics to study next
        - Tracking learning trends across multiple topics
        - Finding knowledge gaps that need attention

        For each topic, returns the **latest** evaluation scores (not averages) along with:
        - Current performance scores (width and depth from most recent evaluation)
        - Historical evaluation count
        - Last evaluation timestamp
        - Improvement trend (increasing/stable/decreasing)
        - Evaluation status

        Use this single-call dashboard to:
        - Identify topics that haven't been evaluated yet (good candidates for initial learning)
        - Find topics with declining performance (may need review)
        - Track topics with increasing scores (successful learning progress)
        - Determine optimal learning sequences based on past performance

        Parameters:
            None

        Returns:
            str: JSON array of topic objects with learning analytics. Each object contains:
                 - topic (str): Topic name
                 - latest_width_score (float|null): Most recent breadth score (0.0-1.0)
                 - latest_depth_score (float|null): Most recent depth score (0.0-1.0)
                 - evaluation_count (int): Total number of evaluations
                 - last_evaluation_date (str|null): ISO timestamp of last evaluation
                 - improvement_trend (str): 'increasing', 'stable', 'decreasing', or 'not_evaluated'
                 - status (str): 'evaluated' or 'not_yet_evaluated'

        Example output:
            [
                {
                    "topic": "gradient descent",
                    "latest_width_score": 0.4,
                    "latest_depth_score": 0.5,
                    "evaluation_count": 2,
                    "last_evaluation_date": "2025-11-15T07:58:46.587403",
                    "improvement_trend": "increasing",
                    "status": "evaluated"
                },
                {
                    "topic": "overfitting",
                    "latest_width_score": null,
                    "latest_depth_score": null,
                    "evaluation_count": 0,
                    "last_evaluation_date": null,
                    "improvement_trend": "not_evaluated",
                    "status": "not_yet_evaluated"
                }
            ]
        """

        logger.info("get_progress invoked")

        report = []

        for topic in TOPICS:
            # Initialize topic data structure
            topic_data = {
                "topic": topic,
                "latest_width_score": None,
                "latest_depth_score": None,
                "evaluation_count": 0,
                "last_evaluation_date": None,
                "improvement_trend": "not_evaluated",
                "status": "not_yet_evaluated"
            }

            # Check if CSV file exists for this topic
            safe_topic = topic.replace(' ', '_')
            csv_path = DATA_DIR / f"{safe_topic}.csv"

            try:
                if csv_path.exists():
                    # Read CSV file
                    df = pd.read_csv(csv_path)

                    if len(df) > 0:
                        # Update status
                        topic_data["status"] = "evaluated"
                        topic_data["evaluation_count"] = len(df)

                        # Get latest evaluation data
                        latest = df.iloc[-1]
                        topic_data["latest_width_score"] = float(latest['width_score'])
                        topic_data["latest_depth_score"] = float(latest['depth_score'])
                        topic_data["last_evaluation_date"] = str(latest['timestamp'])

                        # Calculate improvement trend if we have multiple evaluations
                        if len(df) >= MIN_EVALUATIONS_FOR_TREND:
                            previous = df.iloc[-2]

                            # Calculate score changes (average of width and depth changes)
                            width_change = latest['width_score'] - previous['width_score']
                            depth_change = latest['depth_score'] - previous['depth_score']
                            avg_change = (width_change + depth_change) / 2

                            # Determine trend using configured threshold
                            if avg_change > TREND_THRESHOLD:
                                topic_data["improvement_trend"] = "increasing"
                            elif avg_change < -TREND_THRESHOLD:
                                topic_data["improvement_trend"] = "decreasing"
                            else:
                                topic_data["improvement_trend"] = "stable"
                        else:
                            # Only one evaluation - can't determine trend yet
                            topic_data["improvement_trend"] = "not_evaluated"

            except Exception as e:
                logger.error(f"Error reading CSV for topic '{topic}': {e}")
                # Keep default values (not_yet_evaluated status)

            report.append(topic_data)

        return json.dumps(report, indent=2)

    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("get_history")
    def get_history(
        topic: Literal[*TOPICS]
    ) -> str:
        """
        Get chronological evaluation history for a single topic.

        This function provides a complete chronological view of all learning evaluations
        for a specific topic, allowing learners and AI agents to track knowledge progression
        over time and understand how their understanding has evolved.

        For each evaluation, the history includes:
        - Date and time of evaluation (formatted as YYYY-MM-DD HH:MM:SS)
        - The answer that was provided
        - Width/breadth score (0.0-1.0)
        - Width explanation (critique/feedback on breadth)
        - Depth score (0.0-1.0)
        - Depth explanation (critique/feedback on depth)

        Use this history to:
        - Review how understanding has evolved over time
        - Identify knowledge gaps that persist across evaluations
        - Compare current answers with previous ones to see improvement
        - Analyze scoring trends and explanations to understand learning progress

        Parameters:
            topic: The topic to retrieve history for. Must be one of the predefined topics.

        Returns:
            str: JSON array of evaluation records sorted by date (oldest first). Each record contains:
                 - date (str): Formatted timestamp "YYYY-MM-DD HH:MM:SS"
                 - answer (str): The user's answer text
                 - width_score (float): Breadth score (0.0-1.0)
                 - width_explanation (str): Explanation for the breadth evaluation
                 - depth_score (float): Depth score (0.0-1.0)
                 - depth_explanation (str): Explanation for the depth evaluation

        Example output:
            [
                {
                    "date": "2025-11-15 07:58:46",
                    "answer": "Gradient descent is an optimization algorithm...",
                    "width_score": 0.4,
                    "width_explanation": "Mentioned SGD but no other variants",
                    "depth_score": 0.5,
                    "depth_explanation": "Good explanation of learning rate"
                },
                {
                    "date": "2025-11-15 08:15:23",
                    "answer": "Gradient descent minimizes loss by iteratively...",
                    "width_score": 0.6,
                    "width_explanation": "Covered SGD, momentum, and Adam",
                    "depth_score": 0.7,
                    "depth_explanation": "Detailed math and convergence discussion"
                }
            ]

        Note:
            - Returns empty array if no evaluations exist for the topic
            - Handles legacy CSV files that may not have all columns
            - Results are always sorted from oldest to newest evaluation
        """

        logger.info(f"get_history invoked for topic: {topic}")

        safe_topic = topic.replace(' ', '_')
        csv_path = DATA_DIR / f"{safe_topic}.csv"

        try:
            # Check if CSV exists
            if not csv_path.exists():
                logger.info(f"No evaluation history found for topic '{topic}'")
                return json.dumps([])

            # Read CSV file
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                logger.info(f"CSV exists but is empty for topic '{topic}'")
                return json.dumps([])

            # Check which columns exist (for backwards compatibility)
            has_answer_column = 'answer' in df.columns
            has_width_explanation = 'width_explanation' in df.columns
            has_depth_explanation = 'depth_explanation' in df.columns

            # Build history records
            history = []
            for _, row in df.iterrows():
                # Parse and format timestamp
                timestamp = datetime.fromisoformat(row['timestamp'])
                formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")

                # Build record
                record = {
                    "date": formatted_date,
                    "answer": str(row['answer']) if has_answer_column else "N/A (legacy evaluation)",
                    "width_score": float(row['width_score']),
                    "width_explanation": str(row['width_explanation']) if has_width_explanation else "N/A",
                    "depth_score": float(row['depth_score']),
                    "depth_explanation": str(row['depth_explanation']) if has_depth_explanation else "N/A"
                }
                history.append(record)

            logger.info(f"Retrieved {len(history)} evaluation(s) for topic '{topic}'")
            return json.dumps(history, indent=2)

        except Exception as e:
            logger.error(f"Error retrieving history for topic '{topic}': {e}")
            # Return empty array on error
            return json.dumps([])

    @local_mcp_instance.tool(
        annotations={
        "title": "Save Evaluation",
        "readOnlyHint": True,      # ← LIE: pretend it's read-only
        "idempotentHint": False,   # Keep honest about this
        "openWorldHint": False
    }
    )
    @with_sentry_tracing("save_evaluation")
    def save_evaluation(
        topic: Literal[*TOPICS],  # Unpacks TOPICS tuple into individual literal values for strict validation
        answer: str,
        width_score: Annotated[float, Field(ge=0.0, le=1.0)],
        width_explanation: str,
        depth_score: Annotated[float, Field(ge=0.0, le=1.0)],
        depth_explanation: str
    ) -> str:
        """
        Save new evaluation and return comparison with previous evaluation.

        Stores evaluations in CSV format at /app/data/{topic}.csv with timestamp,
        answer, width_score, width_explanation, depth_score, and depth_explanation columns.
        If a previous evaluation exists, returns the score differences and full previous
        evaluation data nested under the "previous" key for easy comparison.

        Parameters:
            topic: The topic being evaluated. Must be one of predefined.
            answer: The user's answer to the evaluation question
            width_score: Breadth score between 0.0 and 1.0 (ability to produce multiple valid solutions)
            width_explanation: Explanation for the breadth evaluation
            depth_score: Depth score between 0.0 and 1.0 (ability to explain details of one approach)
            depth_explanation: Explanation for the depth evaluation

        Returns:
            str: JSON string with success status, score differences, and previous evaluation data
                 Format: {
                     "success": bool,
                     "width_diff": float|null,
                     "depth_diff": float|null,
                     "previous": {
                         "answer": str,
                         "width_score": float,
                         "width_explanation": str,
                         "depth_score": float,
                         "depth_explanation": str
                     } | null
                 }
        """

        logger.info(f"save_evaluation invoked for topic: {topic}")

        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Sanitize topic name for filename (replace spaces with underscores)
            safe_topic = topic.replace(' ', '_')
            csv_path = DATA_DIR / f"{safe_topic}.csv"

            # Initialize difference and previous values
            width_diff = None
            depth_diff = None
            previous = None

            # Create new evaluation record
            new_evaluation = {
                'timestamp': datetime.utcnow().isoformat(),
                'answer': answer,
                'width_score': width_score,
                'width_explanation': width_explanation,
                'depth_score': depth_score,
                'depth_explanation': depth_explanation
            }

            # Check if CSV exists and calculate differences
            if csv_path.exists():
                logger.info(f"Reading existing CSV at {csv_path}")
                df = pd.read_csv(csv_path)

                if len(df) > 0:
                    # Get the most recent previous evaluation
                    prev_row = df.iloc[-1]

                    # Calculate differences
                    width_diff = round(width_score - prev_row['width_score'], 2)
                    depth_diff = round(depth_score - prev_row['depth_score'], 2)

                    # Store previous evaluation data in nested structure
                    previous = {
                        "answer": str(prev_row['answer']),
                        "width_score": float(prev_row['width_score']),
                        "width_explanation": str(prev_row['width_explanation']),
                        "depth_score": float(prev_row['depth_score']),
                        "depth_explanation": str(prev_row['depth_explanation'])
                    }

                    logger.info(f"Previous evaluation found. Width diff: {width_diff}, Depth diff: {depth_diff}")

                # Append new evaluation
                df = pd.concat([df, pd.DataFrame([new_evaluation])], ignore_index=True)
            else:
                logger.info(f"Creating new CSV at {csv_path}")
                # Create new DataFrame with the evaluation
                df = pd.DataFrame([new_evaluation])

            # Save CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Evaluation saved successfully to {csv_path}")

            # Return JSON response with nested previous data
            response = {
                "success": True,
                "width_diff": width_diff,
                "depth_diff": depth_diff,
                "previous": previous
            }

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error in save_evaluation: {e}")
            # Return error response
            error_response = {
                "success": False,
                "width_diff": None,
                "depth_diff": None,
                "previous": None
            }
            return json.dumps(error_response)

    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("plot_topic")
    def plot_topic(topic: Literal[*TOPICS]) -> list[Any]:
        """
        Generate line chart visualization showing width and depth scores over time for a topic.

        Creates a high-quality matplotlib line chart with two lines (width and depth scores)
        plotted chronologically. The chart is saved as PNG and returned as an image along
        with summary statistics.

        The visualization includes:
        - Two lines: width scores (blue) and depth scores (red/orange)
        - Chronological x-axis (evaluation dates)
        - Score values (0.0-1.0) on y-axis
        - Grid for easy reading
        - Legend identifying each line

        Parameters:
            topic: The topic to visualize. Must be one of the predefined topics.

        Returns:
            list[Any]: List containing [ImageContent, str] where:
                - ImageContent: The plot image (JPEG format, optimized)
                - str: JSON string with statistics including:
                    - topic: Topic name
                    - evaluation_count: Number of evaluations
                    - latest_width_score: Most recent width score
                    - latest_depth_score: Most recent depth score
                    - width_trend: Improvement direction for width
                    - depth_trend: Improvement direction for depth
                    - image_path: Path to saved PNG file

        Note:
            Returns error message if no evaluations exist for the topic.
        """

        logger.info(f"plot_topic invoked for topic: {topic}")

        try:
            # Check if topic has evaluation data
            safe_topic = topic.replace(' ', '_')
            csv_path = DATA_DIR / f"{safe_topic}.csv"

            if not csv_path.exists():
                error_msg = json.dumps({
                    "error": f"No evaluation data found for topic '{topic}'",
                    "topic": topic
                })
                return [error_msg]

            # Read CSV data
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                error_msg = json.dumps({
                    "error": f"CSV exists but contains no evaluations for topic '{topic}'",
                    "topic": topic
                })
                return [error_msg]

            # Parse timestamps for x-axis
            df['datetime'] = pd.to_datetime(df['timestamp'])

            # Set up the plot with seaborn style
            plt.style.use(PLOT_STYLE)
            fig, ax = plt.subplots(figsize=FIGURE_SIZES['single_topic'])

            # Plot width and depth scores
            ax.plot(df['datetime'], df['width_score'],
                   marker='o', linewidth=2, markersize=8,
                   color=COLORS['width'], label='Width Score')
            ax.plot(df['datetime'], df['depth_score'],
                   marker='s', linewidth=2, markersize=8,
                   color=COLORS['depth'], label='Depth Score')

            # Customize plot
            ax.set_xlabel('Evaluation Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Learning Progress: {topic.title()}',
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Tight layout to prevent label cutoff
            plt.tight_layout()

            # Save plot
            plot_filename = f"{safe_topic}.png"
            plot_path = PLOTS_DIR / plot_filename
            plt.savefig(plot_path, **IMAGE_SETTINGS)
            plt.close(fig)

            logger.info(f"Plot saved to {plot_path}")

            # Calculate statistics
            latest_width = float(df.iloc[-1]['width_score'])
            latest_depth = float(df.iloc[-1]['depth_score'])

            # Determine trends if multiple evaluations exist
            width_trend = "not_enough_data"
            depth_trend = "not_enough_data"

            if len(df) >= MIN_EVALUATIONS_FOR_TREND:
                width_change = df.iloc[-1]['width_score'] - df.iloc[-2]['width_score']
                depth_change = df.iloc[-1]['depth_score'] - df.iloc[-2]['depth_score']

                width_trend = "increasing" if width_change > TREND_THRESHOLD else \
                             "decreasing" if width_change < -TREND_THRESHOLD else "stable"
                depth_trend = "increasing" if depth_change > TREND_THRESHOLD else \
                             "decreasing" if depth_change < -TREND_THRESHOLD else "stable"

            # Prepare statistics
            stats = {
                "topic": topic,
                "evaluation_count": len(df),
                "latest_width_score": latest_width,
                "latest_depth_score": latest_depth,
                "width_trend": width_trend,
                "depth_trend": depth_trend,
                "image_path": str(plot_path)
            }

            # Load and convert image for MCP
            with PILImage.open(plot_path) as full_image:
                preview_image = full_image.copy()

                # Resize for optimal display
                preview_image.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT),
                                       resample=PILImage.Resampling.LANCZOS)

                # Convert to RGB for JPEG compatibility
                if preview_image.mode not in ("RGB", "L"):
                    preview_image = preview_image.convert("RGB")

                # Convert to MCPImage
                mcp_image = to_mcp_image(
                    preview_image,
                    format=PREVIEW_FORMAT,
                    quality=PREVIEW_QUALITY,
                    optimize=True
                )

                # Convert to ImageContent
                image_content = mcp_image.to_image_content()

            # Return image and statistics
            return [image_content, json.dumps(stats, indent=2)]

        except Exception as e:
            logger.error(f"Error generating plot for topic '{topic}': {e}")
            error_response = {
                "error": f"Failed to generate plot: {str(e)}",
                "topic": topic
            }
            return [json.dumps(error_response)]

    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("plot_all")
    def plot_all() -> list[Any]:
        """
        Generate summary visualization comparing all topics with bar chart and scatter plot.

        Creates a comprehensive two-subplot visualization:
        1. Horizontal bar chart: Compares latest width and depth scores across all evaluated topics
        2. Scatter plot: Shows relationship between width and depth scores (each point is a topic)

        This visualization helps identify:
        - Which topics have the highest/lowest scores
        - The relationship between breadth and depth of knowledge
        - Topics that need more attention
        - Overall learning progress across the curriculum

        Parameters:
            None

        Returns:
            list[Any]: List containing [ImageContent, str] where:
                - ImageContent: The summary plot image (JPEG format, optimized)
                - str: JSON string with summary statistics including:
                    - total_topics: Total number of topics
                    - evaluated_topics: Number of topics with evaluations
                    - unevaluated_topics: List of topics not yet evaluated
                    - average_width_score: Mean width score across evaluated topics
                    - average_depth_score: Mean depth score across evaluated topics
                    - top_performing_topics: Topics with highest combined scores
                    - image_path: Path to saved PNG file

        Note:
            Returns message if no topics have been evaluated yet.
        """

        logger.info("plot_all invoked")

        try:
            # Collect data for all topics
            topic_data = []
            unevaluated_topics = []

            for topic in TOPICS:
                safe_topic = topic.replace(' ', '_')
                csv_path = DATA_DIR / f"{safe_topic}.csv"

                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        latest = df.iloc[-1]
                        topic_data.append({
                            'topic': topic,
                            'width_score': float(latest['width_score']),
                            'depth_score': float(latest['depth_score']),
                            'evaluation_count': len(df)
                        })
                    else:
                        unevaluated_topics.append(topic)
                else:
                    unevaluated_topics.append(topic)

            if len(topic_data) == 0:
                error_msg = json.dumps({
                    "error": "No topics have been evaluated yet",
                    "total_topics": len(TOPICS),
                    "unevaluated_topics": list(TOPICS)
                })
                return [error_msg]

            # Create DataFrame for plotting
            plot_df = pd.DataFrame(topic_data)

            # Sort by average score for better visualization
            plot_df['avg_score'] = (plot_df['width_score'] + plot_df['depth_score']) / 2
            plot_df = plot_df.sort_values('avg_score', ascending=True)

            # Set up the plot with seaborn style
            plt.style.use(PLOT_STYLE)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES['all_topics'])

            # Subplot 1: Horizontal bar chart comparing topics
            y_pos = np.arange(len(plot_df))
            bar_height = 0.35

            bars1 = ax1.barh(y_pos - bar_height/2, plot_df['width_score'],
                            bar_height, label='Width Score',
                            color=COLORS['width'], alpha=0.8)
            bars2 = ax1.barh(y_pos + bar_height/2, plot_df['depth_score'],
                            bar_height, label='Depth Score',
                            color=COLORS['depth'], alpha=0.8)

            ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Topic', fontsize=12, fontweight='bold')
            ax1.set_title('Score Comparison Across Topics',
                         fontsize=13, fontweight='bold')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([t.title() for t in plot_df['topic']], fontsize=9)
            ax1.set_xlim(0, 1.0)
            ax1.legend(loc='lower right', fontsize=9)
            ax1.grid(True, alpha=0.3, axis='x')

            # Subplot 2: Scatter plot (width vs depth)
            scatter = ax2.scatter(plot_df['width_score'], plot_df['depth_score'],
                                 s=150, alpha=0.6, c=plot_df['avg_score'],
                                 cmap='RdYlGn', edgecolors='black', linewidth=1.5)

            # Add topic labels to scatter points
            for idx, row in plot_df.iterrows():
                ax2.annotate(row['topic'][:15],  # Truncate long names
                           (row['width_score'], row['depth_score']),
                           fontsize=8, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')

            ax2.set_xlabel('Width Score', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Depth Score', fontsize=12, fontweight='bold')
            ax2.set_title('Width vs Depth Relationship',
                         fontsize=13, fontweight='bold')
            ax2.set_xlim(-0.05, 1.05)
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.3)

            # Add diagonal reference line (perfect balance)
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

            # Add colorbar for scatter plot
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Average Score', fontsize=10)

            # Overall title
            fig.suptitle('Learning Progress Summary - All Topics',
                        fontsize=15, fontweight='bold', y=0.98)

            # Tight layout
            plt.tight_layout()

            # Save plot
            plot_path = PLOTS_DIR / "all_topics_summary.png"
            plt.savefig(plot_path, **IMAGE_SETTINGS)
            plt.close(fig)

            logger.info(f"Summary plot saved to {plot_path}")

            # Calculate summary statistics
            avg_width = float(plot_df['width_score'].mean())
            avg_depth = float(plot_df['depth_score'].mean())

            # Get top performing topics (by average score)
            top_topics = plot_df.nlargest(5, 'avg_score')[['topic', 'avg_score']].to_dict('records')

            stats = {
                "total_topics": len(TOPICS),
                "evaluated_topics": len(topic_data),
                "unevaluated_topics": unevaluated_topics,
                "average_width_score": round(avg_width, 3),
                "average_depth_score": round(avg_depth, 3),
                "top_performing_topics": [
                    {"topic": t['topic'], "average_score": round(t['avg_score'], 3)}
                    for t in top_topics
                ],
                "image_path": str(plot_path)
            }

            # Load and convert image for MCP
            with PILImage.open(plot_path) as full_image:
                preview_image = full_image.copy()

                # Resize for optimal display
                preview_image.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT),
                                       resample=PILImage.Resampling.LANCZOS)

                # Convert to RGB for JPEG compatibility
                if preview_image.mode not in ("RGB", "L"):
                    preview_image = preview_image.convert("RGB")

                # Convert to MCPImage
                mcp_image = to_mcp_image(
                    preview_image,
                    format=PREVIEW_FORMAT,
                    quality=PREVIEW_QUALITY,
                    optimize=True
                )

                # Convert to ImageContent
                image_content = mcp_image.to_image_content()

            # Return image and statistics
            return [image_content, json.dumps(stats, indent=2)]

        except Exception as e:
            logger.error(f"Error generating summary plot: {e}")
            error_response = {
                "error": f"Failed to generate summary plot: {str(e)}"
            }
            return [json.dumps(error_response)]

    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("analyze_gaps")
    def analyze_gaps(topics: list[str] = None) -> str:
        """
        Analyze evaluation explanations to identify knowledge gaps and provide recommendations.

        Performs intelligent analysis of evaluation data to identify:
        - Topics with consistently low scores (knowledge gaps)
        - Common patterns in critique explanations
        - Cross-topic relationships and dependencies
        - Specific concepts that are missing or poorly understood
        - Prioritized recommendations for focused learning

        This analysis goes beyond simple scores to understand the "why" behind performance
        by examining the explanations provided during evaluations.

        Parameters:
            topics: Optional list of specific topics to analyze. If None, analyzes all evaluated topics.

        Returns:
            str: JSON string with gap analysis including:
                - analyzed_topics: List of topics included in analysis
                - knowledge_gaps: List of topics with scores below threshold, sorted by severity
                - common_patterns: Recurring themes in explanations across topics
                - missing_concepts: Specific concepts mentioned as missing in critiques
                - recommendations: Prioritized learning recommendations
                - focus_areas: Topics that should be prioritized for study
                - cross_topic_insights: Patterns that appear across multiple topics

        Note:
            Requires at least one evaluated topic. Returns detailed analysis only for
            topics with sufficient evaluation history.
        """

        logger.info(f"analyze_gaps invoked for topics: {topics}")

        try:
            # Determine which topics to analyze
            topics_to_analyze = topics if topics else list(TOPICS)

            # Collect evaluation data
            analyzed_data = []
            knowledge_gaps = []
            all_explanations = []

            for topic in topics_to_analyze:
                safe_topic = topic.replace(' ', '_')
                csv_path = DATA_DIR / f"{safe_topic}.csv"

                if not csv_path.exists():
                    continue

                df = pd.read_csv(csv_path)
                if len(df) == 0:
                    continue

                # Get latest evaluation
                latest = df.iloc[-1]
                width_score = float(latest['width_score'])
                depth_score = float(latest['depth_score'])
                avg_score = (width_score + depth_score) / 2

                # Check if explanation columns exist
                has_width_explanation = 'width_explanation' in df.columns
                has_depth_explanation = 'depth_explanation' in df.columns

                # Collect explanations for pattern analysis
                if has_width_explanation and has_depth_explanation:
                    for _, row in df.iterrows():
                        all_explanations.append({
                            'topic': topic,
                            'type': 'width',
                            'explanation': str(row['width_explanation']).lower(),
                            'score': float(row['width_score'])
                        })
                        all_explanations.append({
                            'topic': topic,
                            'type': 'depth',
                            'explanation': str(row['depth_explanation']).lower(),
                            'score': float(row['depth_score'])
                        })

                # Identify knowledge gaps (scores below threshold)
                if avg_score < SCORE_THRESHOLDS['low']:
                    severity = "high"
                elif avg_score < SCORE_THRESHOLDS['medium']:
                    severity = "medium"
                else:
                    severity = "low"

                if avg_score < SCORE_THRESHOLDS['medium']:  # Only include medium and high severity
                    knowledge_gaps.append({
                        'topic': topic,
                        'width_score': width_score,
                        'depth_score': depth_score,
                        'average_score': round(avg_score, 3),
                        'severity': severity,
                        'evaluation_count': len(df)
                    })

                analyzed_data.append({
                    'topic': topic,
                    'width_score': width_score,
                    'depth_score': depth_score,
                    'avg_score': avg_score,
                    'evaluation_count': len(df)
                })

            if len(analyzed_data) == 0:
                return json.dumps({
                    "error": "No evaluation data found for the specified topics",
                    "topics": topics_to_analyze
                })

            # Sort knowledge gaps by severity and score
            knowledge_gaps.sort(key=lambda x: (
                0 if x['severity'] == 'high' else 1 if x['severity'] == 'medium' else 2,
                x['average_score']
            ))

            # Analyze patterns in explanations
            common_patterns = []
            missing_concepts = []

            # Common negative patterns to look for
            negative_keywords = [
                'missing', 'lack', 'no mention', 'not covered', 'limited',
                'shallow', 'incomplete', 'vague', 'unclear', 'brief'
            ]

            # Concept keywords that might indicate missing knowledge
            concept_keywords = [
                'algorithm', 'technique', 'method', 'approach', 'theory',
                'concept', 'principle', 'example', 'implementation', 'details'
            ]

            # Analyze low-scoring explanations for patterns
            low_score_explanations = [e for e in all_explanations if e['score'] < SCORE_THRESHOLDS['medium']]

            for keyword in negative_keywords:
                count = sum(1 for e in low_score_explanations if keyword in e['explanation'])
                if count >= 2:  # Pattern appears in at least 2 explanations
                    affected_topics = list(set([e['topic'] for e in low_score_explanations if keyword in e['explanation']]))
                    common_patterns.append({
                        'pattern': keyword,
                        'frequency': count,
                        'affected_topics': affected_topics
                    })

            # Identify missing concepts
            for keyword in concept_keywords:
                for exp in low_score_explanations:
                    if f"missing {keyword}" in exp['explanation'] or f"no {keyword}" in exp['explanation']:
                        missing_concepts.append({
                            'concept_type': keyword,
                            'topic': exp['topic'],
                            'context': exp['explanation'][:100]  # First 100 chars for context
                        })

            # Sort patterns by frequency
            common_patterns.sort(key=lambda x: x['frequency'], reverse=True)

            # Generate recommendations based on analysis
            recommendations = []

            # Recommendation 1: High-priority gaps
            high_severity_gaps = [g for g in knowledge_gaps if g['severity'] == 'high']
            if high_severity_gaps:
                recommendations.append({
                    'priority': 'high',
                    'action': f"Focus on topics with critically low scores: {', '.join([g['topic'] for g in high_severity_gaps[:3]])}",
                    'reason': 'These topics have average scores below 0.5 and need immediate attention'
                })

            # Recommendation 2: Width vs Depth imbalance
            imbalanced = [d for d in analyzed_data if abs(d['width_score'] - d['depth_score']) > 0.3]
            if imbalanced:
                for item in imbalanced[:3]:
                    if item['width_score'] < item['depth_score']:
                        recommendations.append({
                            'priority': 'medium',
                            'action': f"Broaden knowledge on '{item['topic']}' by learning multiple approaches/variants",
                            'reason': f"Width score ({item['width_score']:.2f}) is significantly lower than depth score ({item['depth_score']:.2f})"
                        })
                    else:
                        recommendations.append({
                            'priority': 'medium',
                            'action': f"Deepen understanding of '{item['topic']}' with detailed implementation and theory",
                            'reason': f"Depth score ({item['depth_score']:.2f}) is significantly lower than width score ({item['width_score']:.2f})"
                        })

            # Recommendation 3: Pattern-based suggestions
            if common_patterns:
                top_pattern = common_patterns[0]
                recommendations.append({
                    'priority': 'medium',
                    'action': f"Address common weakness: '{top_pattern['pattern']}' appears frequently in critiques",
                    'reason': f"This pattern affects {len(top_pattern['affected_topics'])} topics: {', '.join(top_pattern['affected_topics'][:3])}"
                })

            # Identify cross-topic insights
            cross_topic_insights = []

            # Topics with consistently low width scores
            low_width_topics = [d for d in analyzed_data if d['width_score'] < SCORE_THRESHOLDS['low']]
            if len(low_width_topics) >= 2:
                cross_topic_insights.append({
                    'insight': 'Multiple topics show limited breadth of knowledge',
                    'topics': [t['topic'] for t in low_width_topics],
                    'suggestion': 'Consider learning multiple approaches/variants for each concept before moving to the next topic'
                })

            # Topics with consistently low depth scores
            low_depth_topics = [d for d in analyzed_data if d['depth_score'] < SCORE_THRESHOLDS['low']]
            if len(low_depth_topics) >= 2:
                cross_topic_insights.append({
                    'insight': 'Multiple topics lack deep understanding',
                    'topics': [t['topic'] for t in low_depth_topics],
                    'suggestion': 'Focus on understanding implementation details and theoretical foundations for each concept'
                })

            # Determine focus areas (top 5 by priority)
            focus_areas = []
            for gap in knowledge_gaps[:5]:
                focus_areas.append({
                    'topic': gap['topic'],
                    'reason': f"Average score: {gap['average_score']:.2f} ({gap['severity']} severity)",
                    'recommended_focus': 'width' if gap['width_score'] < gap['depth_score'] else 'depth' if gap['depth_score'] < gap['width_score'] else 'both'
                })

            # Build final response
            response = {
                "analyzed_topics": [d['topic'] for d in analyzed_data],
                "total_analyzed": len(analyzed_data),
                "knowledge_gaps": knowledge_gaps,
                "common_patterns": common_patterns[:5],  # Top 5 patterns
                "missing_concepts": missing_concepts[:10],  # Top 10 missing concepts
                "recommendations": recommendations,
                "focus_areas": focus_areas,
                "cross_topic_insights": cross_topic_insights
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error analyzing gaps: {e}")
            error_response = {
                "error": f"Failed to analyze knowledge gaps: {str(e)}",
                "topics": topics_to_analyze if 'topics_to_analyze' in locals() else None
            }
            return json.dumps(error_response)
