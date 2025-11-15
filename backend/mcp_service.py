import logging
import os
import json
from datetime import datetime
from typing import Annotated, Literal
import pandas as pd
from pydantic import Field
from sentry_utils import with_sentry_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Centralized topics list - single source of truth for all topic-related functions
TOPICS = (
    "gradient descent",
    "overfitting",
    "hyperparameters",
    "logistical regression",
    "supervised learning",
    "unsupervised learning",
    "regularization"
)

def register_checkpoint_tools(local_mcp_instance):
    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("topics_learning_report")
    def topics_learning_report() -> str:
        """
        Generate a comprehensive learning progress report for all available topics.

        This function provides detailed analytics to help AI agents and learners identify
        which topics to study next based on evaluation history, performance trends, and
        knowledge gaps.

        For each topic, the report includes:
        - Current performance scores (width/breadth and depth)
        - Historical evaluation count
        - Last evaluation timestamp
        - Improvement trend (increasing/stable/decreasing)
        - Evaluation status

        Use this report to:
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

        logger.info("topics_learning_report invoked")

        data_dir = "/app/data"
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
            csv_path = os.path.join(data_dir, f"{safe_topic}.csv")

            try:
                if os.path.exists(csv_path):
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
                        if len(df) >= 2:
                            previous = df.iloc[-2]

                            # Calculate score changes (average of width and depth changes)
                            width_change = latest['width_score'] - previous['width_score']
                            depth_change = latest['depth_score'] - previous['depth_score']
                            avg_change = (width_change + depth_change) / 2

                            # Determine trend (using 0.05 as threshold for "stable")
                            if avg_change > 0.05:
                                topic_data["improvement_trend"] = "increasing"
                            elif avg_change < -0.05:
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
            "readOnlyHint": False,
            "destructiveHint": False
        }
    )
    @with_sentry_tracing("evaluate_knowledge")
    def evaluate_knowledge(
        topic: Literal[*TOPICS],  # Unpacks TOPICS tuple into individual literal values for strict validation
        answer: str,
        width_score: Annotated[float, Field(ge=0.0, le=1.0)],
        width_explanation: str,
        depth_score: Annotated[float, Field(ge=0.0, le=1.0)],
        depth_explanation: str
    ) -> str:
        """
        Evaluate knowledge on a specific topic and store the evaluation.

        Stores evaluations in CSV format at /app/data/{topic}.csv with timestamp,
        answer, width_score, width_explanation, depth_score, and depth_explanation columns.
        Returns the difference from the previous evaluation if one exists, along with
        previous evaluation data for comparison.

        Parameters:
            topic: The topic being evaluated. Must be one of predefined.
            answer: The user's answer to the evaluation question
            width_score: Breadth score between 0.0 and 1.0 (ability to produce multiple valid solutions)
            width_explanation: Explanation for the breadth evaluation
            depth_score: Depth score between 0.0 and 1.0 (ability to explain details of one approach)
            depth_explanation: Explanation for the depth evaluation

        Returns:
            str: JSON string with success status, differences, and previous evaluation data
                 Format: {
                     "success": bool,
                     "width_difference": float|null,
                     "depth_difference": float|null,
                     "previous_answer": str|null,
                     "previous_width_score": float|null,
                     "previous_width_explanation": str|null,
                     "previous_depth_score": float|null,
                     "previous_depth_explanation": str|null
                 }
        """

        logger.info(f"evaluate_knowledge invoked for topic: {topic}")

        try:
            # Ensure data directory exists
            data_dir = "/app/data"
            os.makedirs(data_dir, exist_ok=True)

            # Sanitize topic name for filename (replace spaces with underscores)
            safe_topic = topic.replace(' ', '_')
            csv_path = os.path.join(data_dir, f"{safe_topic}.csv")

            # Initialize difference and previous values
            width_difference = None
            depth_difference = None
            previous_answer = None
            previous_width_score = None
            previous_width_explanation = None
            previous_depth_score = None
            previous_depth_explanation = None

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
            if os.path.exists(csv_path):
                logger.info(f"Reading existing CSV at {csv_path}")
                df = pd.read_csv(csv_path)

                if len(df) > 0:
                    # Get the most recent previous evaluation
                    previous = df.iloc[-1]

                    # Calculate differences
                    width_difference = round(width_score - previous['width_score'], 2)
                    depth_difference = round(depth_score - previous['depth_score'], 2)

                    # Store previous values
                    previous_answer = str(previous['answer'])
                    previous_width_score = float(previous['width_score'])
                    previous_width_explanation = str(previous['width_explanation'])
                    previous_depth_score = float(previous['depth_score'])
                    previous_depth_explanation = str(previous['depth_explanation'])

                    logger.info(f"Previous evaluation found. Width diff: {width_difference}, Depth diff: {depth_difference}")

                # Append new evaluation
                df = pd.concat([df, pd.DataFrame([new_evaluation])], ignore_index=True)
            else:
                logger.info(f"Creating new CSV at {csv_path}")
                # Create new DataFrame with the evaluation
                df = pd.DataFrame([new_evaluation])

            # Save CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Evaluation saved successfully to {csv_path}")

            # Return JSON response
            response = {
                "success": True,
                "width_difference": width_difference,
                "depth_difference": depth_difference,
                "previous_answer": previous_answer,
                "previous_width_score": previous_width_score,
                "previous_width_explanation": previous_width_explanation,
                "previous_depth_score": previous_depth_score,
                "previous_depth_explanation": previous_depth_explanation
            }

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error in evaluate_knowledge: {e}")
            # Return error response
            error_response = {
                "success": False,
                "width_difference": None,
                "depth_difference": None,
                "previous_answer": None,
                "previous_width_score": None,
                "previous_width_explanation": None,
                "previous_depth_score": None,
                "previous_depth_explanation": None
            }
            return json.dumps(error_response)
