from crewai import Agent, Task, Crew, Process
from crewai_tools import *
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import sys

# Load environment variables
load_dotenv()

def create_llm(model="gpt-4o-mini", temperature=0.7):
    """Create LLM instance"""
    return ChatOpenAI(
        model_name=model,
        temperature=temperature
    )

def create_tool(tool_config):
    """Create tool instance based on configuration"""
    tool_map = {
        "CustomApiTool": CustomApiTool,
        "CustomCodeInterpreterTool": CustomCodeInterpreterTool,
        "ScrapeWebsiteToolEnhanced": ScrapeWebsiteToolEnhanced,
        "CSVSearchToolEnhanced": CSVSearchToolEnhanced
    }
    
    tool_class = tool_map.get(tool_config["name"])
    if tool_class:
        return tool_class()
    return None

def create_agents_from_json(json_data):
    """Create CrewAI agents from JSON configuration"""
    agents = []
    tools_map = {tool["tool_id"]: create_tool(tool) for tool in json_data["Tools"]}
    
    for agent_config in json_data["Agents"]:
        # Get tools for this agent
        agent_tools = [
            tools_map[tool["tool_id"]] 
            for tool in json_data["Tools"] 
            if tool["name"] in agent_config["SelectTools"]
        ]
        
        # Create agent
        agent = Agent(
            role=agent_config["CreateAgent"],
            goal=agent_config["Goal"],
            backstory=agent_config["Backstory"],
            allow_delegation=agent_config["AllowDelegation"] == "yes",
            verbose=agent_config["Verbose"],
            tools=agent_tools,
            llm=create_llm(
                model=agent_config["LLM"],
                temperature=agent_config["Temperature"]
            )
        )
        agents.append(agent)
    
    return agents

def create_tasks_from_json(json_data, agents):
    """Create CrewAI tasks from JSON configuration"""
    tasks = []
    agent_map = {agent.role: agent for agent in agents}
    
    for task_config in json_data["Tasks"]:
        task = Task(
            description=task_config["Description"],
            expected_output=task_config["ExpectedOutput"],
            agent=agent_map[task_config["Agent"]],
            async_execution=task_config["AsyncExecution"]
        )
        tasks.append(task)
    
    return tasks

def create_crew_from_json_data(json_data):
    """Create complete CrewAI setup from JSON data"""
    # Create agents
    agents = create_agents_from_json(json_data)
    
    # Create tasks
    tasks = create_tasks_from_json(json_data, agents)
    
    # Create crew
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical if json_data["CrewCreationDetails"]["Process"] == "horizontal" else Process.sequential,
        verbose=json_data["CrewCreationDetails"]["Verbose"],
        manager_llm=create_llm(json_data["CrewCreationDetails"]["ManagerLLM"]),
        memory=json_data["CrewCreationDetails"]["Memory"] == "enabled",
        cache=json_data["CrewCreationDetails"]["Cache"] == "enabled",
        max_rpm=json_data["CrewCreationDetails"]["MaxReqMin"]
    )
    
    return crew

def create_crew_from_json(json_path):
    """Create complete CrewAI setup from JSON file"""
    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        return create_crew_from_json_data(json_data)
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_path}'")
        print("Please make sure the JSON configuration file exists in the current directory.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{json_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():
    # Example JSON data
    example_json = {
        "CrewCreationDetails": {
            "Name": "AI_Agent_Team_01",
            "Process": "horizontal",
            "Agents": ["DataEntryAgent", "PredictiveModelAgent"],
            "Tasks": ["DataIngestionTask", "ModelTrainingTask", "OutputDeliveryTask"],
            "ManagerLLM": "GPT-4",
            "ManagerAgent": "AI_Manager_Agent",
            "Verbose": True,
            "Memory": "enabled",
            "Cache": "enabled",
            "Planning": "automated",
            "MaxReqMin": 1000
        },
        "Tools": [
            {
                "tool_id": "tool_01",
                "name": "CustomApiTool",
                "type": "API Integration",
                "availability": True
            },
            {
                "tool_id": "tool_02",
                "name": "CustomCodeInterpreterTool",
                "type": "Code Execution",
                "availability": True
            }
        ],
        "Agents": [
            {
                "CreateAgent": "DataEntryAgent",
                "Role": "Automate data entry tasks",
                "Backstory": "Experienced in data processing",
                "Goal": "To streamline data entry processes",
                "AllowDelegation": "yes",
                "Verbose": True,
                "Cache": "enabled",
                "LLM": "gpt-4o-mini",
                "Temperature": 0.5,
                "MaxIteration": 5,
                "SelectTools": ["CustomApiTool", "CustomCodeInterpreterTool"]
            },
            {
                "CreateAgent": "PredictiveModelAgent",
                "Role": "Generate predictive models",
                "Backstory": "Specializes in machine learning",
                "Goal": "To create accurate predictive models",
                "AllowDelegation": "yes",
                "Verbose": True,
                "Cache": "enabled",
                "LLM": "gpt-4o-mini",
                "Temperature": 0.7,
                "MaxIteration": 10,
                "SelectTools": ["CustomApiTool", "CustomCodeInterpreterTool"]
            }
        ],
        "Tasks": [
            {
                "CreateTask": "DataIngestionTask",
                "Description": "Ingest and validate data",
                "ExpectedOutput": "Cleaned and structured data",
                "Agent": "DataEntryAgent",
                "AsyncExecution": True
            },
            {
                "CreateTask": "ModelTrainingTask",
                "Description": "Train predictive models",
                "ExpectedOutput": "Trained models with metrics",
                "Agent": "PredictiveModelAgent",
                "AsyncExecution": True
            }
        ]
    }

    if len(sys.argv) > 1:
        # If a file path is provided as argument, use it
        json_path = sys.argv[1]
        crew = create_crew_from_json(json_path)
    else:
        # Otherwise use the example JSON data
        print("No JSON file provided, using example configuration...")
        crew = create_crew_from_json_data(example_json)
    
    # Run the crew
    result = crew.kickoff()
    print("Crew Execution Result:", result)

if __name__ == "__main__":
    main() 