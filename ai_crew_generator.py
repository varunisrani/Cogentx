from crewai import Agent, Task, Crew, Process
from crewai_tools import *
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import sys
from typing import Dict, List
import shutil
import logging
from datetime import datetime

# Configure logging
def setup_logging(output_dir: str = None):
    """Setup logging configuration"""
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s'
    )
    
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create debug log file (all messages)
        debug_log_file = os.path.join(output_dir, f'crew_generation_debug_{timestamp}.log')
        debug_handler = logging.FileHandler(debug_log_file)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(debug_handler)
        
        # Create info log file (info and above)
        info_log_file = os.path.join(output_dir, f'crew_generation_{timestamp}.log')
        file_handler = logging.FileHandler(info_log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to files:")
        logging.info(f"  Debug log: {debug_log_file}")
        logging.info(f"  Info log: {info_log_file}")

# Load environment variables
load_dotenv()

class AICrewGenerator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("="*50)
        self.logger.info("Initializing AICrewGenerator")
        self.logger.info("="*50)
        try:
            self.logger.debug("Setting up OpenAI LLM with model: gpt-4o-mini")
            self.llm = ChatOpenAI(
                model_name="gpt-4o-mini", 
                temperature=0.7
            )
            self.logger.info("Successfully initialized OpenAI LLM")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
            self.logger.debug(f"OpenAI LLM initialization error details:", exc_info=True)
            raise

    def analyze_requirements(self, json_data: Dict) -> Dict:
        """Use OpenAI to analyze the JSON input and identify required components"""
        self.logger.info("Analyzing requirements from JSON configuration")
        
        analysis_prompt = f"""
        Analyze this JSON configuration and identify the key components needed for a CrewAI implementation:
        {json.dumps(json_data, indent=2)}
        
        Please identify:
        1. Required tools and their configurations
        2. Agent roles and their responsibilities
        3. Task dependencies and flow
        4. Any specific LLM requirements
        5. Process type recommendation (sequential vs horizontal)
        
        Format your response as JSON with these sections.
        """
        
        try:
            self.logger.debug("Sending analysis prompt to OpenAI")
            response = self.llm.invoke(analysis_prompt)
            analysis = json.loads(response.content)
            self.logger.info("Successfully analyzed requirements")
            self.logger.debug(f"Analysis result: {json.dumps(analysis, indent=2)}")
            return analysis
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse OpenAI response as JSON: {str(e)}")
            self.logger.info("Falling back to original JSON configuration")
            return json_data
        except Exception as e:
            self.logger.error(f"Error during requirements analysis: {str(e)}")
            self.logger.info("Falling back to original JSON configuration")
            return json_data

    def validate_tools(self, tools_config: List[Dict]) -> List[Dict]:
        """Use OpenAI to validate and configure tools"""
        self.logger.info(f"Validating {len(tools_config)} tools")
        
        # Normalize tool configurations
        normalized_tools = []
        for tool in tools_config:
            normalized_tool = tool.copy()
            # Handle case-insensitive key matching
            if "Name" in tool and "name" not in tool:
                normalized_tool["name"] = tool["Name"]
            if "Purpose" in tool:
                normalized_tool["description"] = tool["Purpose"]
            normalized_tools.append(normalized_tool)
        
        tools_prompt = f"""
        Analyze these tool configurations and provide proper initialization parameters:
        {json.dumps(normalized_tools, indent=2)}
        
        For each tool, specify:
        1. Required parameters
        2. Default values
        3. Any special configuration needed
        
        Format your response as a list of tool configurations.
        """
        
        try:
            self.logger.debug("Sending tools validation prompt to OpenAI")
            response = self.llm.invoke(tools_prompt)
            validated_tools = json.loads(response.content)
            self.logger.info("Successfully validated tools configuration")
            self.logger.debug(f"Validated tools: {json.dumps(validated_tools, indent=2)}")
            return validated_tools
        except Exception as e:
            self.logger.warning(f"Tool validation failed: {str(e)}")
            self.logger.info("Using normalized tool configuration")
            return normalized_tools

    def create_tool(self, tool_config: Dict):
        """Create tool instance based on configuration"""
        self.logger.info(f"Creating tool from config: {json.dumps(tool_config, indent=2)}")
        
        tool_map = {
            "CustomApiTool": CustomApiTool,
            "CustomCodeInterpreterTool": CustomCodeInterpreterTool,
            "ScrapeWebsiteToolEnhanced": ScrapeWebsiteToolEnhanced,
            "CSVSearchToolEnhanced": CSVSearchToolEnhanced,
            "SerperDevTool": SerperDevTool,
            "ScrapeWebsiteTool": ScrapeWebsiteTool,
            "YahooFinanceNewsTool": YahooFinanceNewsTool
        }
        
        try:
            # Handle case-insensitive tool name matching
            tool_name = tool_config.get("name") or tool_config.get("Name")
            if not tool_name:
                self.logger.error("Tool configuration missing 'name' field")
                raise ValueError("Tool configuration must include 'name' field")
                
            self.logger.debug(f"Looking up tool class for: {tool_name}")
            tool_class = tool_map.get(tool_name)
            
            if tool_class:
                self.logger.info(f"Found tool class for: {tool_name}")
                # Use OpenAI to get proper tool initialization
                tool_prompt = f"""
                For this tool configuration:
                {json.dumps(tool_config, indent=2)}
                
                Provide the exact initialization parameters needed for {tool_name}.
                Format as JSON.
                """
                
                try:
                    self.logger.debug("Sending tool initialization prompt to OpenAI")
                    response = self.llm.invoke(tool_prompt)
                    tool_params = json.loads(response.content)
                    self.logger.info(f"Successfully got tool parameters for {tool_name}")
                    return tool_class(**tool_params)
                except Exception as e:
                    self.logger.warning(f"Failed to get custom parameters for {tool_name}, using default initialization: {str(e)}")
                    return tool_class()
            else:
                self.logger.warning(f"No tool class found for: {tool_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating tool: {str(e)}")
            return None

    def create_agents_from_json(self, json_data: Dict, analysis: Dict) -> List[Agent]:
        """Create CrewAI agents with OpenAI-enhanced configuration"""
        agents = []
        tools_map = {tool["tool_id"]: self.create_tool(tool) for tool in json_data["Tools"]}
        
        for agent_config in json_data["Agents"]:
            # Get OpenAI recommendations for agent configuration
            agent_prompt = f"""
            For this agent configuration:
            {json.dumps(agent_config, indent=2)}
            
            Provide optimal:
            1. Goal refinement
            2. Backstory enhancement
            3. Tool selection justification
            4. LLM parameters
            
            Format as JSON.
            """
            
            response = self.llm.invoke(agent_prompt)
            try:
                enhanced_config = json.loads(response.content)
                agent_config.update(enhanced_config)
            except:
                pass
            
            # Get tools for this agent
            agent_tools = [
                tools_map[tool["tool_id"]] 
                for tool in json_data["Tools"] 
                if tool["name"] in agent_config["SelectTools"]
            ]
            
            # Create agent with enhanced configuration
            agent = Agent(
                role=agent_config["CreateAgent"],
                goal=agent_config["Goal"],
                backstory=agent_config["Backstory"],
                allow_delegation=agent_config["AllowDelegation"] == "yes",
                verbose=agent_config["Verbose"],
                tools=agent_tools,
                llm=ChatOpenAI(
                    model_name=agent_config["LLM"],
                    temperature=agent_config["Temperature"]
                )
            )
            agents.append(agent)
        
        return agents

    def create_tasks_from_json(self, json_data: Dict, agents: List[Agent], analysis: Dict) -> List[Task]:
        """Create CrewAI tasks with OpenAI-enhanced configuration"""
        tasks = []
        agent_map = {agent.role: agent for agent in agents}
        
        for task_config in json_data["Tasks"]:
            # Get OpenAI recommendations for task configuration
            task_prompt = f"""
            For this task configuration:
            {json.dumps(task_config, indent=2)}
            
            Provide optimal:
            1. Description enhancement
            2. Expected output refinement
            3. Execution strategy
            4. Dependencies handling
            
            Format as JSON.
            """
            
            response = self.llm.invoke(task_prompt)
            try:
                enhanced_config = json.loads(response.content)
                task_config.update(enhanced_config)
            except:
                pass
            
            task = Task(
                description=task_config["Description"],
                expected_output=task_config["ExpectedOutput"],
                agent=agent_map[task_config["Agent"]],
                async_execution=task_config["AsyncExecution"]
            )
            tasks.append(task)
        
        return tasks

    def create_crew(self, json_data: Dict) -> Crew:
        """Create complete CrewAI setup with OpenAI analysis"""
        
        # Get OpenAI analysis of requirements
        analysis = self.analyze_requirements(json_data)
        
        # Validate and enhance tools configuration
        validated_tools = self.validate_tools(json_data["Tools"])
        json_data["Tools"] = validated_tools
        
        # Create agents with enhanced configuration
        agents = self.create_agents_from_json(json_data, analysis)
        
        # Create tasks with enhanced configuration
        tasks = self.create_tasks_from_json(json_data, agents, analysis)
        
        # Create crew with optimal settings
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if json_data["CrewCreationDetails"]["Process"] == "horizontal" else Process.sequential,
            verbose=json_data["CrewCreationDetails"]["Verbose"],
            manager_llm=ChatOpenAI(
                model_name=json_data["CrewCreationDetails"]["ManagerLLM"],
                temperature=0.7
            ),
            memory=json_data["CrewCreationDetails"]["Memory"] == "enabled",
            cache=json_data["CrewCreationDetails"]["Cache"] == "enabled",
            max_rpm=json_data["CrewCreationDetails"]["MaxReqMin"]
        )
        
        return crew

    def generate_app(self, json_data: Dict, output_dir: str):
        """Generate a Streamlit app for the crew"""
        self.logger.info(f"Generating Streamlit app in directory: {output_dir}")
        
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                self.logger.debug(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)

            # Validate JSON structure
            self.logger.info("Validating JSON structure")
            required_keys = ["CrewCreationDetails", "Tools", "Agents", "Tasks"]
            for key in required_keys:
                if key not in json_data:
                    error_msg = f"Missing required key '{key}' in JSON configuration"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

            crew_details = json_data["CrewCreationDetails"]
            required_crew_details = ["Process", "Verbose", "Memory", "Cache", "MaxReqMin", "ManagerLLM"]
            for key in required_crew_details:
                if key not in crew_details:
                    error_msg = f"Missing required key '{key}' in CrewCreationDetails"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Set default name if not provided
            crew_name = crew_details.get("Name", "AI_Crew")
            self.logger.info(f"Using crew name: {crew_name}")
            
            # Check if any custom tools are used
            custom_tools_used = any(
                tool["name"] in ["CustomApiTool", "CustomCodeInterpreterTool", "ScrapeWebsiteToolEnhanced", "CSVSearchToolEnhanced"]
                for tool in json_data["Tools"]
            )
            if custom_tools_used:
                self.logger.info("Custom tools detected in configuration")

            # Generate files
            self.logger.info("Generating application files")
            
            # Generate app.py
            self.logger.debug("Generating app.py")
            app_content = self._generate_app_content(json_data, crew_details, crew_name, custom_tools_used)
            app_path = os.path.join(output_dir, 'app.py')
            with open(app_path, 'w') as f:
                f.write(app_content)
            self.logger.info(f"Created app.py: {app_path}")

            # Create .env file
            self.logger.debug("Creating .env file")
            env_path = os.path.join(output_dir, '.env')
            with open(env_path, 'w') as f:
                f.write(self._get_env_content())
            self.logger.info(f"Created .env file: {env_path}")

            # Create requirements.txt
            self.logger.debug("Creating requirements.txt")
            req_path = os.path.join(output_dir, 'requirements.txt')
            with open(req_path, 'w') as f:
                f.write(self._get_requirements_content())
            self.logger.info(f"Created requirements.txt: {req_path}")

            # Create run scripts
            self.logger.debug("Creating run scripts")
            self._create_run_scripts(output_dir)
            self.logger.info("Created run scripts")

            # Copy custom tools if needed
            if custom_tools_used:
                self.logger.info("Setting up custom tools")
                tools_dir = os.path.join(output_dir, 'tools')
                os.makedirs(tools_dir, exist_ok=True)
                self.logger.info(f"Created tools directory: {tools_dir}")
                # TODO: Implement copying of custom tool files

            self.logger.info("Successfully generated all application files")
            
        except Exception as e:
            self.logger.error(f"Error generating app: {str(e)}")
            raise

    def _generate_app_content(self, json_data, crew_details, crew_name, custom_tools_used):
        """Generate the content for app.py"""
        self.logger.debug("Generating app.py content")
        
        def json_dumps_python(obj):
            if isinstance(obj, bool):
                return str(obj).lower()
            return json.dumps(obj)

        # Generate agent definitions
        agent_definitions = ",\n        ".join([
            f"""Agent(
            role={json_dumps_python(agent["CreateAgent"])},
            backstory={json_dumps_python(agent.get("Backstory", "An AI agent"))},
            goal={json_dumps_python(agent.get("Goal", "To complete assigned tasks"))},
            allow_delegation={json_dumps_python(agent.get("AllowDelegation", "yes") == "yes")},
            verbose={json_dumps_python(agent.get("Verbose", True))},
            tools=[{', '.join([f'{tool}()' for tool in agent.get("SelectTools", [])])}],
            llm=create_llm({json_dumps_python(f"OpenAI: {agent.get('LLM', 'gpt-4o-mini')}")}, {json_dumps_python(agent.get("Temperature", 0.7))})
        )"""
            for agent in json_data["Agents"]
        ])

        # Generate task definitions
        task_definitions = ",\n        ".join([
            f"""Task(
            description={json_dumps_python(task["Description"])},
            expected_output={json_dumps_python(task.get("ExpectedOutput", "Task completed successfully"))},
            agent=next(agent for agent in agents if agent.role == {json_dumps_python(task["Agent"])}),
            async_execution={json_dumps_python(task.get("AsyncExecution", False))}
        )"""
            for task in json_data["Tasks"]
        ])

        # Prepare custom tool imports
        custom_tool_imports = []
        if custom_tools_used:
            custom_tool_imports.extend([
                "from tools.CustomApiTool import CustomApiTool",
                "from tools.CustomCodeInterpreterTool import CustomCodeInterpreterTool",
                "from tools.ScrapeWebsiteToolEnhanced import ScrapeWebsiteToolEnhanced",
                "from tools.CSVSearchToolEnhanced import CSVSearchToolEnhanced"
            ])

        # Generate app.py content
        app_content = f'''import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *
{chr(10).join(custom_tool_imports) if custom_tools_used else ""}

# Load environment variables
load_dotenv()

def create_lmstudio_llm(model, temperature):
    """Create an LLM instance using LM Studio."""
    api_base = os.getenv('LMSTUDIO_API_BASE')
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_API_BASE"] = api_base
    if api_base:
        return ChatOpenAI(openai_api_key='lm-studio', openai_api_base=api_base, temperature=temperature)
    else:
        raise ValueError("LM Studio API base not set in .env file")

def create_openai_llm(model, temperature):
    """Create an LLM instance using OpenAI."""
    safe_pop_env_var('OPENAI_API_KEY')
    safe_pop_env_var('OPENAI_API_BASE')
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1/')
    if api_key:
        return ChatOpenAI(openai_api_key=api_key, openai_api_base=api_base, model_name=model, temperature=temperature)
    else:
        raise ValueError("OpenAI API key not set in .env file")

def create_groq_llm(model, temperature):
    """Create an LLM instance using Groq."""
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Groq API key not set in .env file")

def create_anthropic_llm(model, temperature):
    """Create an LLM instance using Anthropic."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return ChatAnthropic(anthropic_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Anthropic API key not set in .env file")

def safe_pop_env_var(key):
    """Safely remove an environment variable."""
    try:
        os.environ.pop(key)
    except KeyError:
        pass
        
LLM_CONFIG = {{
    "OpenAI": {{
        "create_llm": create_openai_llm
    }},
    "Groq": {{
        "create_llm": create_groq_llm
    }},
    "LM Studio": {{
        "create_llm": create_lmstudio_llm
    }},
    "Anthropic": {{
        "create_llm": create_anthropic_llm
    }}
}}

def create_llm(provider_and_model="OpenAI: gpt-4o-mini", temperature=0.7):
    """Create an LLM instance based on provider and model."""
    if isinstance(provider_and_model, str) and ":" in provider_and_model:
        provider, model = provider_and_model.split(": ")
        create_llm_func = LLM_CONFIG.get(provider, {{}}).get("create_llm")
        if create_llm_func:
            return create_llm_func(model, temperature)
    
    # Default to OpenAI
    return create_openai_llm("gpt-4o-mini", temperature)

def load_agents():
    """Load and create agent instances."""
    agents = [
        {agent_definitions}
    ]
    return agents

def load_tasks(agents):
    """Load and create task instances."""
    tasks = [
        {task_definitions}
    ]
    return tasks

def main():
    st.title({json_dumps_python(crew_name)})

    agents = load_agents()
    tasks = load_tasks(agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process={json_dumps_python("Process.hierarchical" if crew_details["Process"] == "horizontal" else "Process.sequential")},
        verbose={json_dumps_python(crew_details["Verbose"])},
        memory={json_dumps_python(crew_details["Memory"] == "enabled")},
        cache={json_dumps_python(crew_details["Cache"] == "enabled")},
        max_rpm={json_dumps_python(crew_details["MaxReqMin"])},
        manager_llm=create_llm({json_dumps_python(f"OpenAI: {crew_details['ManagerLLM']}")})
    )

    if st.button("Run Crew"):
        with st.spinner("Running crew..."):
            try:
                result = crew.kickoff()
                with st.expander("Final output", expanded=True):
                    if hasattr(result, 'raw'):
                        st.write(result.raw)
                with st.expander("Full output", expanded=False):
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {{str(e)}}")

if __name__ == '__main__':
    main()
'''
        return app_content

    def _get_env_content(self):
        """Get content for .env file"""
        return """
# OPENAI_API_KEY="FILL-IN-YOUR-OPENAI-API-KEY"
# OPENAI_API_BASE="OPTIONAL-FILL-IN-YOUR-OPENAI-API-BASE"
# GROQ_API_KEY="FILL-IN-YOUR-GROQ-API-KEY"
# ANTHROPIC_API_KEY="FILL-IN-YOUR-ANTHROPIC-API-KEY"
# LMSTUDIO_API_BASE="http://localhost:1234/v1"
"""

    def _get_requirements_content(self):
        """Get content for requirements.txt"""
        return """
streamlit
crewai
langchain-openai
langchain-groq
langchain-anthropic
python-dotenv
"""

    def _create_run_scripts(self, output_dir):
        """Create run scripts for different platforms"""
        # Create run.sh
        run_sh_path = os.path.join(output_dir, 'run.sh')
        with open(run_sh_path, 'w') as f:
            f.write("""#!/bin/bash
source venv/bin/activate
streamlit run app.py --server.headless true
""")
        os.chmod(run_sh_path, 0o755)
        self.logger.debug(f"Created run.sh: {run_sh_path}")

        # Create run.bat
        run_bat_path = os.path.join(output_dir, 'run.bat')
        with open(run_bat_path, 'w') as f:
            f.write("""@echo off
call venv\\Scripts\\activate
streamlit run app.py --server.headless true
""")
        self.logger.debug(f"Created run.bat: {run_bat_path}")

    def normalize_json_input(self, json_data: Dict) -> Dict:
        """Use OpenAI to normalize the JSON input to match expected format"""
        self.logger.info("Normalizing JSON input format")
        
        normalize_prompt = f"""
        Transform this JSON configuration into the standard CrewAI format.
        Input JSON:
        {json.dumps(json_data, indent=2)}
        
        Convert it to match this exact structure:
        {{
            "CrewCreationDetails": {{
                "Name": "string",
                "Process": "sequence/horizontal",
                "Agents": ["list of agent names"],
                "Tasks": ["list of task names"],
                "ManagerLLM": "model name",
                "ManagerAgent": "agent name",
                "Verbose": boolean,
                "Memory": "enabled/disabled",
                "Cache": "enabled/disabled",
                "Planning": "string",
                "MaxReqMin": number
            }},
            "Tools": [
                {{
                    "tool_id": "string",
                    "name": "string",
                    "type": "string",
                    "description": "string",
                    "availability": boolean
                }}
            ],
            "Agents": [
                {{
                    "CreateAgent": "string",
                    "Role": "string",
                    "Backstory": "string",
                    "Goal": "string",
                    "AllowDelegation": "yes/no",
                    "Verbose": boolean,
                    "Cache": "enabled/disabled",
                    "LLM": "model name",
                    "Temperature": number,
                    "MaxIteration": number,
                    "SelectTools": ["list of tool names"]
                }}
            ],
            "Tasks": [
                {{
                    "CreateTask": "string",
                    "Description": "string",
                    "ExpectedOutput": "string",
                    "Agent": "string",
                    "AsyncExecution": boolean,
                    "ContextFromAsyncTasks": "string",
                    "ContextFromSyncTasks": "string"
                }}
            ]
        }}
        
        Rules:
        1. Map similar fields to their standard names (e.g., "Crew creation details" -> "CrewCreationDetails")
        2. Convert values to appropriate types (boolean, number, string)
        3. Ensure all required fields are present
        4. Keep the original data and intent while normalizing the format
        5. If a field doesn't exist in input, use reasonable defaults
        
        Return ONLY the normalized JSON without any additional text or explanation.
        The response should be a valid JSON object that can be parsed directly.
        """
        
        try:
            self.logger.debug("Sending normalization prompt to OpenAI")
            response = self.llm.invoke(normalize_prompt)
            
            # Log the raw response for debugging
            self.logger.debug(f"Raw OpenAI response: {response}")
            
            # Extract JSON from the response
            try:
                # Try to parse the response content directly
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Remove any markdown code block indicators if present
                content = content.replace('```json', '').replace('```', '').strip()
                
                # Try to parse the JSON
                self.logger.debug(f"Attempting to parse JSON content: {content}")
                normalized_data = json.loads(content)
                
                # Validate the normalized data has required structure
                required_keys = ["CrewCreationDetails", "Tools", "Agents", "Tasks"]
                missing_keys = [key for key in required_keys if key not in normalized_data]
                
                if missing_keys:
                    self.logger.warning(f"Normalized data missing required keys: {missing_keys}")
                    # Add missing sections with defaults
                    for key in missing_keys:
                        if key == "CrewCreationDetails":
                            normalized_data[key] = {
                                "Name": "Default_Crew",
                                "Process": "sequential",
                                "Agents": [],
                                "Tasks": [],
                                "ManagerLLM": "gpt-4o-mini",
                                "ManagerAgent": "Default_Manager",
                                "Verbose": True,
                                "Memory": "enabled",
                                "Cache": "enabled",
                                "Planning": "automated",
                                "MaxReqMin": 1000
                            }
                        elif key == "Tools":
                            normalized_data[key] = []
                        elif key == "Agents":
                            normalized_data[key] = []
                        elif key == "Tasks":
                            normalized_data[key] = []
                
                self.logger.info("Successfully normalized JSON format")
                self.logger.debug(f"Final normalized data: {json.dumps(normalized_data, indent=2)}")
                return normalized_data
                
            except json.JSONDecodeError as je:
                self.logger.error(f"Failed to parse normalized JSON: {str(je)}")
                self.logger.debug(f"JSON parse error details:", exc_info=True)
                # Fall back to original JSON with basic normalization
                return self._basic_normalize(json_data)
                
        except Exception as e:
            self.logger.error(f"Error during normalization process: {str(e)}")
            self.logger.debug("Normalization error details:", exc_info=True)
            # Fall back to original JSON with basic normalization
            return self._basic_normalize(json_data)

    def _basic_normalize(self, json_data: Dict) -> Dict:
        """Perform basic normalization of JSON data without using OpenAI"""
        self.logger.info("Performing basic JSON normalization")
        
        normalized = {
            "CrewCreationDetails": {
                "Name": json_data.get("CrewCreationDetails", {}).get("Name", "Default_Crew"),
                "Process": json_data.get("CrewCreationDetails", {}).get("Process", "sequential"),
                "Agents": json_data.get("CrewCreationDetails", {}).get("Agents", []),
                "Tasks": json_data.get("CrewCreationDetails", {}).get("Tasks", []),
                "ManagerLLM": json_data.get("CrewCreationDetails", {}).get("ManagerLLM", "gpt-4o-mini"),
                "ManagerAgent": json_data.get("CrewCreationDetails", {}).get("ManagerAgent", "Default_Manager"),
                "Verbose": json_data.get("CrewCreationDetails", {}).get("Verbose", True),
                "Memory": json_data.get("CrewCreationDetails", {}).get("Memory", "enabled"),
                "Cache": json_data.get("CrewCreationDetails", {}).get("Cache", "enabled"),
                "Planning": json_data.get("CrewCreationDetails", {}).get("Planning", "automated"),
                "MaxReqMin": json_data.get("CrewCreationDetails", {}).get("MaxReqMin", 1000)
            },
            "Tools": json_data.get("Tools", []),
            "Agents": json_data.get("Agents", []),
            "Tasks": json_data.get("Tasks", [])
        }
        
        self.logger.debug(f"Basic normalized data: {json.dumps(normalized, indent=2)}")
        return normalized

    def generate_and_zip(self, json_data: Dict, output_dir: str) -> str:
        """Generate app files and create a zip archive"""
        try:
            self.logger.info("Starting file generation and zipping process")
            
            # Generate the app files
            self.generate_app(json_data, output_dir)
            
            # Create zip file
            zip_path = f"{output_dir}.zip"
            self.logger.info(f"Creating zip archive: {zip_path}")
            
            shutil.make_archive(output_dir, 'zip', output_dir)
            
            # Clean up the output directory
            self.logger.debug(f"Cleaning up directory: {output_dir}")
            shutil.rmtree(output_dir)
            
            self.logger.info("Successfully created zip archive")
            return zip_path
            
        except Exception as e:
            self.logger.error(f"Error in generate_and_zip: {str(e)}")
            self.logger.debug("Generate and zip error details:", exc_info=True)
            raise

    @staticmethod
    def create_from_json(json_data: Dict, output_dir: str) -> str:
        """Static method to create crew app from JSON data"""
        generator = AICrewGenerator()
        normalized_json = generator.normalize_json_input(json_data)
        return generator.generate_and_zip(normalized_json, output_dir)

def main():
    print("CrewAI App Generator with OpenAI Analysis")
    print("="*40)
    
    while True:
        try:
            json_path = input("Enter path to your JSON configuration file (or press Enter for example): ").strip()
            output_dir = input("Enter output directory name (default: crew_app): ").strip() or "crew_app"
            
            # Setup logging
            setup_logging(output_dir)
            logger = logging.getLogger("main")
            
            logger.info("="*50)
            logger.info("Starting CrewAI App Generation Process")
            logger.info("="*50)
            logger.info(f"Output Directory: {output_dir}")
            logger.info(f"JSON Configuration Path: {json_path or 'Using example configuration'}")
            
            if not json_path:
                logger.info("Creating example configuration...")
                # Use example configuration
                json_path = "example_config.json"
                with open(json_path, "w") as f:
                    example_config = {
                        "CrewCreationDetails": {
                            "Name": "AI_Agent_Team_01",
                            "Process": "horizontal",
                            "Agents": ["DataEntryAgent", "PredictiveModelAgent"],
                            "Tasks": ["DataIngestionTask", "ModelTrainingTask"],
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
                    json.dump(example_config, f, indent=2)
                logger.info(f"Created example configuration file: {json_path}")
                logger.debug(f"Example configuration content: {json.dumps(example_config, indent=2)}")
            
            # Load and parse JSON
            logger.info("Loading JSON configuration...")
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            logger.debug(f"Loaded JSON content: {json.dumps(json_data, indent=2)}")
            
            # Create generator and normalize JSON
            logger.info("Creating AICrewGenerator instance...")
            generator = AICrewGenerator()
            
            # Normalize JSON format
            logger.info("Normalizing JSON format...")
            logger.debug("Starting JSON normalization process")
            normalized_json = generator.normalize_json_input(json_data)
            logger.debug(f"Normalized JSON: {json.dumps(normalized_json, indent=2)}")
            
            logger.info("Generating application...")
            generator.generate_app(normalized_json, output_dir)
            
            logger.info("="*50)
            logger.info(f"Successfully created Streamlit app in directory: {output_dir}")
            logger.info("="*50)
            
            print(f"\nCreated Streamlit app in directory: {output_dir}")
            print("\nTo run the app:")
            print(f"1. cd {output_dir}")
            print("2. pip install -r requirements.txt")
            print("3. streamlit run app.py")
            
            logger.info("Generation process completed successfully")
            break
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            logger.debug("File not found error details:", exc_info=True)
            print(f"Error: Could not find file '{json_path}'. Please try again.")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.debug("JSON decode error details:", exc_info=True)
            print(f"Error: Invalid JSON format in file '{json_path}'. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug("Unexpected error details:", exc_info=True)
            print(f"Error: {str(e)}")
            if input("Would you like to try again? (y/n): ").lower() != 'y':
                logger.info("User chose to exit after error")
                break

if __name__ == "__main__":
    main() 