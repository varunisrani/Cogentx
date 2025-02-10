import streamlit as st
import json
from json_to_crew import create_crew_from_json
import tempfile
import os

def create_crew_config():
    st.title("CrewAI Configuration Creator")
    
    # Crew Creation Details
    st.header("Crew Creation Details")
    crew_details = {}
    crew_details["Name"] = st.text_input("Crew Name", "AI_Agent_Team_01")
    crew_details["Process"] = st.selectbox("Process Type", ["sequential", "horizontal"])
    
    # Agents List
    st.subheader("Agents List")
    num_agents = st.number_input("Number of Agents", min_value=1, value=2)
    agents_list = []
    for i in range(int(num_agents)):
        st.write(f"Agent {i+1}")
        agent = {}
        agent["CreateAgent"] = st.text_input(f"Agent {i+1} Name", f"Agent_{i+1}")
        agent["Role"] = st.text_area(f"Agent {i+1} Role", "Describe the role...")
        agent["Backstory"] = st.text_area(f"Agent {i+1} Backstory", "Agent's background...")
        agent["Goal"] = st.text_area(f"Agent {i+1} Goal", "Agent's objective...")
        agent["AllowDelegation"] = st.selectbox(f"Agent {i+1} Allow Delegation", ["yes", "no"])
        agent["Verbose"] = st.checkbox(f"Agent {i+1} Verbose", True)
        agent["Cache"] = st.selectbox(f"Agent {i+1} Cache", ["enabled", "disabled"])
        agent["LLM"] = st.selectbox(f"Agent {i+1} LLM", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
        agent["Temperature"] = st.slider(f"Agent {i+1} Temperature", 0.0, 1.0, 0.7)
        agent["MaxIteration"] = st.number_input(f"Agent {i+1} Max Iterations", 1, 100, 5)
        
        # Tool Selection
        available_tools = ["CustomApiTool", "CustomCodeInterpreterTool"]
        agent["SelectTools"] = st.multiselect(f"Agent {i+1} Tools", available_tools)
        
        agents_list.append(agent)
    
    # Tasks
    st.header("Tasks")
    num_tasks = st.number_input("Number of Tasks", min_value=1, value=2)
    tasks_list = []
    for i in range(int(num_tasks)):
        st.write(f"Task {i+1}")
        task = {}
        task["CreateTask"] = st.text_input(f"Task {i+1} Name", f"Task_{i+1}")
        task["Description"] = st.text_area(f"Task {i+1} Description", "Describe the task...")
        task["ExpectedOutput"] = st.text_area(f"Task {i+1} Expected Output", "Expected output...")
        task["Agent"] = st.selectbox(f"Task {i+1} Agent", [agent["CreateAgent"] for agent in agents_list])
        task["AsyncExecution"] = st.checkbox(f"Task {i+1} Async Execution", False)
        task["ContextFromAsyncTasks"] = st.text_input(f"Task {i+1} Context From Async Tasks", "N/A")
        task["ContextFromSyncTasks"] = st.text_input(f"Task {i+1} Context From Sync Tasks", "N/A")
        tasks_list.append(task)
    
    # Tools Configuration
    st.header("Tools Configuration")
    tools_list = []
    for i, tool_name in enumerate(["CustomApiTool", "CustomCodeInterpreterTool"]):
        tool = {
            "tool_id": f"tool_{i+1:02d}",
            "name": tool_name,
            "type": "API Integration" if tool_name == "CustomApiTool" else "Code Execution",
            "availability": True
        }
        tools_list.append(tool)
    
    # Additional Crew Details
    crew_details["Agents"] = [agent["CreateAgent"] for agent in agents_list]
    crew_details["Tasks"] = [task["CreateTask"] for task in tasks_list]
    crew_details["ManagerLLM"] = st.selectbox("Manager LLM", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"])
    crew_details["ManagerAgent"] = st.selectbox("Manager Agent", [agent["CreateAgent"] for agent in agents_list])
    crew_details["Verbose"] = st.checkbox("Verbose", True)
    crew_details["Memory"] = st.selectbox("Memory", ["enabled", "disabled"])
    crew_details["Cache"] = st.selectbox("Cache", ["enabled", "disabled"])
    crew_details["Planning"] = st.selectbox("Planning", ["automated", "manual"])
    crew_details["MaxReqMin"] = st.number_input("Max Requests per Minute", 1, 10000, 1000)
    
    # Create final configuration
    config = {
        "CrewCreationDetails": crew_details,
        "Tools": tools_list,
        "Agents": agents_list,
        "Tasks": tasks_list
    }
    
    return config

def main():
    st.set_page_config(page_title="CrewAI Creator", layout="wide")
    
    config = create_crew_config()
    
    # Create and Run Crew
    if st.button("Create and Run Crew"):
        # Save configuration to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config, f, indent=2)
            temp_file_path = f.name
        
        try:
            # Create and run crew
            crew = create_crew_from_json(temp_file_path)
            result = crew.kickoff()
            
            # Display results
            st.success("Crew created and executed successfully!")
            st.json(result)
            
        except Exception as e:
            st.error(f"Error creating/running crew: {str(e)}")
        
        finally:
            # Cleanup temporary file
            os.unlink(temp_file_path)
    
    # Download Configuration
    if st.button("Download Configuration"):
        st.download_button(
            label="Download JSON Configuration",
            data=json.dumps(config, indent=2),
            file_name="crew_config.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main() 