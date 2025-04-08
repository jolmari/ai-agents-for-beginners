import asyncio
import os
import random

from typing import Annotated
from openai import AsyncOpenAI
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

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
)  # for OpenAI chat completion


class BookTravelPlugin:
    """Plugin to handle travel booking tasks."""

    @kernel_function(
        name="book_flight", description="Books a flight on a given date and location."
    )
    def book_flight(
        self,
        date: Annotated[str, "The date of the flight."],
        location: Annotated[str, "The destination location."],
    ) -> str:
        return f"Flight booked to {location} on {date}."


class DestinationPlugin:
    """Plugin to handle destination-related tasks."""

    # Constructor
    def __init__(self):

        # Vacation destination list
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia",
        ]

        # Track latest destination to avoid duplicate suggestions
        self.latest_destination = None

    """ The @kernel_function decorator registers the function as a kernel function for the agent to recognize. """

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(
        self,
    ) -> Annotated[str, "Returns a random vacation destination."]:
        # Copy the list of destinations to avoid modifying the original list
        available_destinations = self.destinations.copy()

        # Remove the latest destination from the list if it exists
        if self.latest_destination in available_destinations:
            available_destinations.remove(self.latest_destination)

        # Select a random destination from the available options
        next_destination = random.choice(available_destinations)

        # Print the selected destination
        print(f"Selected destination: {next_destination}")

        # Update the latest destination
        self.latest_destination = next_destination
        return next_destination


# Load environment variables from .env file
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

service_id = "agent"

chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
    service_id=service_id,
)

print("Initialized OpenAI kernel...")

""""
A kernel is a collection of the services and plugins that will be used by your Agents. 
In this snippet, we are creating the kernel and adding the chat_completion_service to it.
"""
kernel = Kernel()
kernel.add_plugin(BookTravelPlugin(), plugin_name="book_travel")
kernel.add_plugin(DestinationPlugin(), plugin_name="select_destination")
kernel.add_service(chat_completion_service)

print("Kernel initialized with chat completion service.")

"""
The following code sets up the chat completion agent using the OpenAI chat completion service.
"""
print("Initializing TravelAgent...")

AGENT_NAME = "TravelAgent"
AGENT_INSTRUCTIONS = "You are a helpful AI Agent that can help plan vacations for customers at random destinations"

settings = kernel.get_prompt_execution_settings_from_service_id(
    service_id=service_id,  # The service ID of the chat completion service
)

# Define the request settings to configure the model with auto-function calling
settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

agent = ChatCompletionAgent(
    service_id=service_id,
    kernel=kernel,
    name=AGENT_NAME,
    instructions=AGENT_INSTRUCTIONS,
    arguments=KernelArguments(settings=settings),
)

print("TravelAgent initialized.")

async def main():

    # Create a new chat history
    chat_history = ChatHistory()

    # Responsd to user input
    user_inputs = [
        "Plan me a day trip.",
        "I don't like that destination. Plan me another vacation.",
        "I want to book a flight to the destination you suggested, on the next Friday.",
        "Actually, I am busy then. Book me a flight to the destination you suggested, on the next Saturday.",
    ]

    for user_input in user_inputs:

        # Add the user input to the chat history
        chat_history.add_user_message(user_input)

        # Print user input
        print("\n" + "=" * 50)
        print(f"User: {user_input}")
        print("=" * 50)

        agent_name: str | None = None
        full_response = ""
        function_calls = []
        function_results = {}

        # Collect the agent's response with function call tracking
        async for content in agent.invoke_stream(chat_history):
            if not agent_name and hasattr(content, "name"):
                agent_name = content.name

            # Track function calls and results
            for item in content.items:
                if isinstance(item, FunctionCallContent):
                    call_info = f"Calling: {item.function_name}({item.arguments})"
                    function_calls.append(call_info)
                elif isinstance(item, FunctionResultContent):
                    result_info = f"Result: {item.result}"
                    function_calls.append(result_info)
                    # Store function results
                    function_results[item.function_name] = item.result

            # Add content to response if it's not a function-related message
            if (
                hasattr(content, "content")
                and content.content
                and content.content.strip()
                and not any(
                    isinstance(item, (FunctionCallContent, FunctionResultContent))
                    for item in content.items
                )
            ):
                full_response += content.content

        # Print function calls if any occurred
        if function_calls:
            print("\nFunction Calls:")
            print("-" * 20)
            for call in function_calls:
                print(call)
            print("-" * 20)

        # Print agent response
        print(f"\n{agent_name or 'Assistant'}:")
        print(full_response)
        print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
