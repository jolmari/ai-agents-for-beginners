import os

from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool
from azure.identity import DefaultAzureCredential
from typing import Any
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
load_dotenv(verbose=True, override=True)

# Initialize the AIProjectClient with the connection string from environment variables
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"),
)

async def run_agent_with_visualization():
    with project_client as client:
        # Create the Azure AI Agent Service Code Interpreter tool
        code_interpreter = CodeInterpreterTool()

        agents = client.agents.list_agents()
        print(f"Total agents found: {len(agents)}")
        
        code_intepreter_agents = [
            agent for agent in agents.data if agent.name == "code-interpreter-agent"
        ]

        if code_intepreter_agents:
            print(
                f"Code interpreter agent already exists. Agent ID: {code_intepreter_agents[0].id}"
            )
            agent = client.agents.get_agent(agent_id=code_intepreter_agents[0].id)
        else:
            print(
                "No existing code interpreter agent found. Proceeding to create a new one."
            )
            
            agent = client.agents.create_agent(
                model="gpt-4o-mini",
                name="code-interpreter-agent",
                instructions="You are a helpful assistant, that can run code and visualize the results.",
                tools=code_interpreter.definitions,
                tool_resources=code_interpreter.resources,
            )

            print(f"Agent created successfully. Agent ID: {agent.id}")

        # Create a thread
        thread = client.agents.create_thread()

        # Create a message to the thread with a hard-coded user query
        user_query = "Could you please create a bar chart for the operating profit using the following data and provide the file to me? Bali: 100 Travelers, Paris: 356 Travelers, London: 900 Travelers, Tokyo: 850 Travellers"
        message = client.agents.create_message(
            thread_id=thread.id, role="user", content=user_query
        )

        print(f"Executing the code interpreter agent with the user query: {user_query}")

        # Run the processing
        run = client.agents.create_and_process_run(
            agent_id=agent.id,
            thread_id=thread.id,
        )

        # Check the status of the run
        if run.status == "failed":
            print("Run failed.")
            return
        elif run.status == "completed":
            print("Run completed successfully.")

        # Get messages from the run
        messages = client.agents.list_messages(
            thread_id=thread.id,
        )
        
        saved_images = []
        
        # Print all {type: text} messages in the thread. Outputs a simple JSON dump.
        for text_message in messages.text_messages:
            print(text_message.as_dict())
            
        for file_path_annotation in messages.file_path_annotations:
            file_id = file_path_annotation.file_path.file_id
            file_name = Path(file_path_annotation.text).name
            
            print(f"Downloading file: {file_name}")
            client.agents.save_file(
                file_id=file_id,
                file_name=file_name,
            )
            saved_images.append(file_name)
            print(f"File saved as: {file_name}")
            
        # Download the image files from the messages
        print("Downloading image files from the messages...")               
        
        for image_content in messages.image_contents:
            file_id = image_content.image_file.file_id
            file_name = f"{file_id}_image_file.png"
            
            print(f"Downloading file: {file_name}")
            client.agents.save_file(
                file_id=file_id,
                file_name=Path(file_name),
            )
            saved_images.append(file_name)
            print(f"File saved as: {file_name}")
            
        print("All image files downloaded successfully.")
        
        for image in saved_images:
            print(f"Image file: {image}")