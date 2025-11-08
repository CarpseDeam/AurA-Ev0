
import httpx
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalSummarizerService:
    """
    A service to summarize conversation history using a local LLM endpoint (e.g., Ollama).
    """

    def __init__(self, endpoint: str, model: str = "llama3:latest"):
        """
        Initializes the LocalSummarizerService.

        Args:
            endpoint (str): The API endpoint for the local model (e.g., "http://localhost:11434/api/generate").
            model (str): The name of the model to use for summarization.
        """
        self.endpoint = endpoint
        self.model = model

    async def summarize_conversation(self, conversation_history: List[Tuple[str, str]]) -> str:
        """
        Summarizes a conversation using the local model.

        Args:
            conversation_history (List[Tuple[str, str]]): A list of tuples representing the conversation.

        Returns:
            str: The summarized conversation, or an empty string if an error occurs.
        """
        if not conversation_history:
            return ""

        # Format the conversation history into a single string for the prompt
        history_str = "\n".join([f"{sender}: {content}" for sender, content in conversation_history])
        
        prompt = (
            "Please summarize the following conversation. The summary should be concise and capture the key points, "
            "decisions, and action items. It will be used as context for a large language model, so it should be "
            "informative and dense.\n\n"
            f"Conversation:\n{history_str}\n\nSummary:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.endpoint, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                
                response_data = response.json()
                summary = response_data.get("response", "").strip()
                logger.info("Successfully summarized conversation.")
                return summary

        except httpx.RequestError as e:
            logger.error(f"Error connecting to local model at {self.endpoint}: {e}")
            return "Error: Could not connect to the local summarization model."
        except httpx.ResponseNotRead as e:
            logger.error(f"Error reading response from local model: {e}")
            return "Error: Incomplete response from the local summarization model."
        except Exception as e:
            logger.error(f"An unexpected error occurred during summarization: {e}")
            return "Error: An unexpected error occurred during summarization."

