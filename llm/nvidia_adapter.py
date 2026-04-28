"""
NVIDIA NIM / Llama 3 API Adapter
---------------------------------
Works with NVIDIA's OpenAI-compatible endpoints (integrate.api.nvidia.com)
Requires: pip install openai
Env vars: NVIDIA_API_KEY
"""

import os
from typing import Optional

from llm.openai_adapter import OpenAIAdapter

class NvidiaAdapter(OpenAIAdapter):
    """
    Extends the OpenAIAdapter since NVIDIA NIM uses an identical API signature.
    Just overrides the base URL and API key env var.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        nvidia_api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        nvidia_base_url = base_url or os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        
        super().__init__(
            api_key=nvidia_api_key,
            base_url=nvidia_base_url,
            timeout=timeout,
        )

    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            models = await client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return [
                "meta/llama-3.3-70b-instruct",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-8b-instruct",
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "nvidia/nemotron-4-340b-instruct",
                "deepseek-ai/deepseek-r1",
            ]
