from typing import Any, Dict

import requests

from app.config import config
from app.logger import logger
from app.tool.base import BaseTool


class Summary(BaseTool):
    """A tool for fetching real time information summary of the query"""

    name: str = "summary"
    description: str = (
        "This is to fetch the real time information summary of the query. Most of the times when approximate results are enough, use this tool to fetch search results"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to fetch the summary of",
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, num_results: int = None, search_params: Dict[str, Any] = None
    ) -> Dict:
        """
        fetch the real time information summary of the query

        Args:
            query (str): The query to fetch the summary of

        Returns:
            Dict: list of search items
        """

        """Get search results from Perplexity API."""
        try:
            headers = {
                "Authorization": f"Bearer {config.llm['default'].perplexity_api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who excels at providing up to date and accurate data",
                    },
                    {"role": "user", "content": query},
                ],
                "web_search_options": {
                    "search_context_size": "medium",
                },
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                res = response.json()
                return res["choices"][0]["message"]["content"]
            else:
                logger.warning(
                    f"Perplexity API returned status code {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error with Perplexity search: {str(e)}")

        return []
