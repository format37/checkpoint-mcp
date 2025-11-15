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
    
def register_checkpoint_tools(local_mcp_instance):
    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True
        }
    )
    @with_sentry_tracing("topics_list")
    def topics_list() -> str:
        """
        Return a list of topics to learn

        Parameters:
            None

        Returns:
            str: Comma separated list of topics
        """

        logger.info("topics_list invoked")

        try:
            with open("/app/data/topics.txt", "r") as f:
                topics = [line.strip() for line in f.readlines() if line.strip()]
            return ", ".join(topics)
        except FileNotFoundError:
            logger.error("topics.txt not found at /app/data/topics.txt")
            return ""
        except Exception as e:
            logger.error(f"Error reading topics.txt: {e}")
            return ""

    @local_mcp_instance.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False
        }
    )
    @with_sentry_tracing("evaluate_knowledge")
    def evaluate_knowledge(
        topic: Literal[
            "gradient descent", 
            "overfitting", 
            "hyperparameters", 
            "logistical regression", 
            "supervised learning", 
            "unsupervised learning", 
            "regularization"
            ],
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
            topic: The topic being evaluated (e.g., "gradient descent")
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
