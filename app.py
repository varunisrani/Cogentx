import streamlit as st
import json
import os
from datetime import datetime
from ai_crew_generator import AICrewGenerator

def create_default_config():
    return {
        "CrewCreationDetails": {
            "Name": "AI_Agent_Team_01",
            "Process": "sequential",
            "Agents": [],
            "Tasks": [],
            "ManagerLLM": "gpt-4o-mini",
            "ManagerAgent": "AI_Manager_Agent",
            "Verbose": True,
            "Memory": "enabled",
            "Cache": "enabled",
            "Planning": "automated",
            "MaxReqMin": 1000
        },
        "Tools": [],
        "Agents": [],
        "Tasks": []
    }

def add_tool():
    tool = {
        "tool_id": f"tool_{len(st.session_state.config['Tools']) + 1:02d}",
        "name": st.session_state.tool_name,
        "type": st.session_state.tool_type,
        "description": st.session_state.tool_description,
        "availability": st.session_state.tool_availability
    }
    st.session_state.config["Tools"].append(tool)
    st.session_state.config["CrewCreationDetails"]["Tools"] = [t["name"] for t in st.session_state.config["Tools"]]

def add_agent():
    agent = {
        "CreateAgent": st.session_state.agent_name,
        "Role": st.session_state.agent_role,
        "Backstory": st.session_state.agent_backstory,
        "Goal": st.session_state.agent_goal,
        "AllowDelegation": "yes" if st.session_state.agent_delegation else "no",
        "Verbose": st.session_state.agent_verbose,
        "Cache": "enabled" if st.session_state.agent_cache else "disabled",
        "LLM": st.session_state.agent_llm,
        "Temperature": st.session_state.agent_temperature,
        "MaxIteration": st.session_state.agent_max_iteration,
        "SelectTools": st.session_state.agent_tools
    }
    st.session_state.config["Agents"].append(agent)
    st.session_state.config["CrewCreationDetails"]["Agents"] = [a["CreateAgent"] for a in st.session_state.config["Agents"]]

def add_task():
    task = {
        "CreateTask": st.session_state.task_name,
        "Description": st.session_state.task_description,
        "ExpectedOutput": st.session_state.task_output,
        "Agent": st.session_state.task_agent,
        "AsyncExecution": st.session_state.task_async,
        "ContextFromAsyncTasks": st.session_state.task_async_context,
        "ContextFromSyncTasks": st.session_state.task_sync_context
    }
    st.session_state.config["Tasks"].append(task)
    st.session_state.config["CrewCreationDetails"]["Tasks"] = [t["CreateTask"] for t in st.session_state.config["Tasks"]]

def main():
    st.set_page_config(page_title="CrewAI Configuration Generator", layout="wide")
    st.title("CrewAI Configuration Generator")

    # Initialize session state
    if "config" not in st.session_state:
        st.session_state.config = create_default_config()

    # Sidebar for configuration options
    with st.sidebar:
        st.header("Configuration Options")
        
        # Crew Creation Details
        st.subheader("Crew Details")
        st.session_state.config["CrewCreationDetails"]["Name"] = st.text_input(
            "Crew Name", 
            st.session_state.config["CrewCreationDetails"]["Name"]
        )
        st.session_state.config["CrewCreationDetails"]["Process"] = st.selectbox(
            "Process Type",
            ["sequential", "horizontal"],
            index=0 if st.session_state.config["CrewCreationDetails"]["Process"] == "sequential" else 1
        )
        st.session_state.config["CrewCreationDetails"]["ManagerLLM"] = st.selectbox(
            "Manager LLM",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state.config["CrewCreationDetails"]["Verbose"] = st.checkbox(
            "Verbose",
            st.session_state.config["CrewCreationDetails"]["Verbose"]
        )
        st.session_state.config["CrewCreationDetails"]["Memory"] = "enabled" if st.checkbox(
            "Enable Memory",
            st.session_state.config["CrewCreationDetails"]["Memory"] == "enabled"
        ) else "disabled"
        st.session_state.config["CrewCreationDetails"]["Cache"] = "enabled" if st.checkbox(
            "Enable Cache",
            st.session_state.config["CrewCreationDetails"]["Cache"] == "enabled"
        ) else "disabled"

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tools", "Agents", "Tasks", "Final JSON"])

    # Tools Tab
    with tab1:
        st.header("Tools Configuration")
        
        # Tool input form
        with st.form("tool_form"):
            st.session_state.tool_name = st.selectbox(
                "Tool Name",
                ["SerperDevTool", "ScrapeWebsiteTool", "YahooFinanceNewsTool", 
                 "CustomApiTool", "CustomCodeInterpreterTool"]
            )
            st.session_state.tool_type = st.text_input("Tool Type", "Search")
            st.session_state.tool_description = st.text_area("Tool Description", "")
            st.session_state.tool_availability = st.checkbox("Tool Available", True)
            
            if st.form_submit_button("Add Tool"):
                add_tool()
                st.success("Tool added successfully!")

        # Display current tools
        if st.session_state.config["Tools"]:
            st.subheader("Current Tools")
            for tool in st.session_state.config["Tools"]:
                st.json(tool)

    # Agents Tab
    with tab2:
        st.header("Agents Configuration")
        
        # Agent input form
        with st.form("agent_form"):
            st.session_state.agent_name = st.text_input("Agent Name", "")
            st.session_state.agent_role = st.text_input("Role", "")
            st.session_state.agent_backstory = st.text_area("Backstory", "")
            st.session_state.agent_goal = st.text_area("Goal", "")
            st.session_state.agent_delegation = st.checkbox("Allow Delegation", True)
            st.session_state.agent_verbose = st.checkbox("Verbose", True)
            st.session_state.agent_cache = st.checkbox("Enable Cache", True)
            st.session_state.agent_llm = st.selectbox(
                "LLM Model",
                ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
            )
            st.session_state.agent_temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            st.session_state.agent_max_iteration = st.number_input("Max Iterations", 1, 100, 5)
            st.session_state.agent_tools = st.multiselect(
                "Select Tools",
                [tool["name"] for tool in st.session_state.config["Tools"]]
            )
            
            if st.form_submit_button("Add Agent"):
                add_agent()
                st.success("Agent added successfully!")

        # Display current agents
        if st.session_state.config["Agents"]:
            st.subheader("Current Agents")
            for agent in st.session_state.config["Agents"]:
                st.json(agent)

    # Tasks Tab
    with tab3:
        st.header("Tasks Configuration")
        
        # Task input form
        with st.form("task_form"):
            st.session_state.task_name = st.text_input("Task Name", "")
            st.session_state.task_description = st.text_area("Description", "")
            st.session_state.task_output = st.text_area("Expected Output", "")
            st.session_state.task_agent = st.selectbox(
                "Assign Agent",
                [agent["CreateAgent"] for agent in st.session_state.config["Agents"]]
            )
            st.session_state.task_async = st.checkbox("Async Execution", False)
            st.session_state.task_async_context = st.text_input("Context From Async Tasks", "")
            st.session_state.task_sync_context = st.text_input("Context From Sync Tasks", "")
            
            if st.form_submit_button("Add Task"):
                add_task()
                st.success("Task added successfully!")

        # Display current tasks
        if st.session_state.config["Tasks"]:
            st.subheader("Current Tasks")
            for task in st.session_state.config["Tasks"]:
                st.json(task)

    # Final JSON Tab
    with tab4:
        st.header("Final Configuration")
        st.json(st.session_state.config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download JSON"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_str = json.dumps(st.session_state.config, indent=2)
                st.download_button(
                    label="Download JSON File",
                    data=json_str,
                    file_name=f"crew_config_{timestamp}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Generate Crew App"):
                try:
                    with st.spinner("Generating CrewAI application..."):
                        # Create unique output directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_dir = f"crew_app_{timestamp}"
                        
                        # Generate the app and get zip file path
                        zip_path = AICrewGenerator.create_from_json(
                            st.session_state.config,
                            output_dir
                        )
                        
                        # Read the zip file
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        # Offer download
                        st.download_button(
                            label="Download Crew App",
                            data=zip_data,
                            file_name=f"crew_app_{timestamp}.zip",
                            mime="application/zip"
                        )
                        
                        # Clean up zip file
                        os.remove(zip_path)
                        
                        st.success("CrewAI application generated successfully!")
                except Exception as e:
                    st.error(f"Error generating CrewAI application: {str(e)}")

if __name__ == "__main__":
    main() 