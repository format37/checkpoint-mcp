import logging
from sentry_utils import with_sentry_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
def register_checkpoint_tools(local_mcp_instance):
    @local_mcp_instance.tool()
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
    