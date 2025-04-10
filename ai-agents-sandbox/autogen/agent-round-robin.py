import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv

# Core imports for semantic kernel
from autogen_agentchat.agents import AssistantAgent
from openai import AsyncOpenAI
from semantic_kernel import Kernel

# Agent imports
from semantic_kernel.agents import ChatCompletionAgent, agent
from semantic_kernel.agents.open_ai import OpenAIAssistantAgent

# Content imports
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, AuthorRole, ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent

# Function imports
from semantic_kernel.functions import KernelArguments, kernel_function

# Connector imports
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)

# Load environment variables from .env file
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

agent_retrieve = AssistantAgent(
    name="data_retrieval"
    model_client=client,
    