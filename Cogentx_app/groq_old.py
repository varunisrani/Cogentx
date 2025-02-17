import streamlit as st
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import pydantic_ai.models.groq as groq_module
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.tools import Tool
from pydantic_graph import Graph, BaseNode, End, GraphRunContext
from dotenv import load_dotenv
import os
from dataclasses import dataclass
import asyncio
import nest_asyncio
import logging
import json
from datetime import datetime
import sys
import logging.handlers

# Initialize basic logging configuration first
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging with both file and console handlers
logger = logging.getLogger("AIAgentSystem")
logger.setLevel(logging.INFO)

# Console Handler (CLI output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File Handler (txt log file)
log_file_path = 'logs/ai_agent_system.txt'
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def initialize_streamlit_state():
    """Initialize Streamlit session state for logging"""
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'log_container' not in st.session_state:
        st.session_state.log_container = st.empty()
    # Read existing logs from file if they exist
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            existing_logs = f.readlines()
            for log in existing_logs:
                try:
                    # Split the log line and handle potential missing parts
                    parts = log.strip().split(' - ')
                    log_entry = {
                        'timestamp': parts[0] if len(parts) > 0 else '',
                        'agent': parts[1] if len(parts) > 1 else 'System',
                        'level': parts[2] if len(parts) > 2 else 'INFO',
                        'message': ' - '.join(parts[3:]) if len(parts) > 3 else log.strip()
                    }
                    st.session_state.logs.append(log_entry)
                except Exception as e:
                    # Skip malformed log entries
                    continue

def log_message(message: str, level: str = "info", agent_name: str = None, to_agent: str = None):
    """Log messages to CLI, file, and Streamlit interface with enhanced communication tracking"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a formatted message with agent communication details
    if agent_name and to_agent:
        formatted_message = f"[{agent_name} → {to_agent}] {message}"
    elif agent_name:
        formatted_message = f"[{agent_name}] {message}"
    else:
        formatted_message = f"[System] {message}"
    
    # Create the full log entry with clear separation for parsing
    log_entry = f"{timestamp} | {formatted_message} | {level.upper()}"
    
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Write to txt file first to ensure no logs are lost
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
            f.flush()  # Ensure immediate write to disk
            os.fsync(f.fileno())  # Force system to write to disk
    except Exception as e:
        # If we can't write to the file, at least print to console
        print(f"Error writing to log file: {str(e)}\nAttempted to log: {log_entry}")
    
    # Log to CLI with different colors based on level - only log once
    if level.lower() == "error":
        logger.error(formatted_message)
    elif level.lower() == "warning":
        logger.warning(formatted_message)
    else:
        logger.info(formatted_message)
    
    # Only update Streamlit if we're in a Streamlit context - only add log entry once
    try:
        if st._is_running_with_streamlit:
            if 'logs' not in st.session_state:
                initialize_streamlit_state()
            
            # Create a unique identifier for the log entry to prevent duplicates
            log_id = f"{timestamp}_{agent_name}_{message[:50]}"
            
            # Only add if this exact log entry hasn't been added before
            if not any(log.get('id') == log_id for log in st.session_state.logs):
                log_entry_dict = {
                    'id': log_id,
                    'timestamp': timestamp,
                    'level': level.upper(),
                    'agent': agent_name,
                    'to_agent': to_agent,
                    'message': message,
                    'is_communication': bool(to_agent),
                    'raw_output': message if len(message) > 500 else None  # Store full output separately if large
                }
                
                st.session_state.logs.append(log_entry_dict)
    except:
        pass  # Not in Streamlit context, skip UI updates

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Create a Streamlit container for logs
if 'log_container' not in st.session_state:
    st.session_state.log_container = st.empty()

# Define base models for structured data
class AgentConfig(BaseModel):
    role: str = Field(..., description="The role of the agent")
    backstory: str = Field(..., description="The agent's backstory")
    goal: str = Field(..., description="The agent's primary goal")
    allow_delegation: bool = Field(default=True)
    verbose: bool = Field(default=True)
    model: Optional[Any] = None

class TaskConfig(BaseModel):
    name: str = Field(..., description="The name of the task")
    description: str = Field(..., description="Detailed description of what the task does")
    expected_output: str = Field(..., description="Expected output format from the task")
    agent_role: str = Field(..., description="Role of the agent responsible for this task")
    async_execution: bool = Field(default=False, description="Whether this task should be executed asynchronously")
    context_requirements: Dict[str, List[str]] = Field(
        default_factory=lambda: {"async": [], "sync": []},
        description="Context required from other tasks"
    )

# Model Provider configurations
def create_groq_model_with_fallback(primary_model: str = "deepseek-r1-distill-qwen-32b", temperature: float = 0.7) -> Any:
    api_key = "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"
    try:
        model = groq_module.GroqModel(
            model_name=primary_model,
            api_key=api_key
        )
        # Test the model with a simple prompt
        test_prompt = "Test connection"
        asyncio.get_event_loop().run_until_complete(model.request(test_prompt, max_tokens=10))
        log_message(f"Successfully initialized {primary_model}", agent_name="System")
        return model
    except Exception as e:
        log_message(
            f"Error initializing {primary_model}, falling back to llama-3.3-70b-versatile: {str(e)}",
            level="warning",
            agent_name="System"
        )
        return groq_module.GroqModel(
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )

def create_anthropic_model(model_name: str = "claude-3-opus-20240229", temperature: float = 0.7) -> AnthropicModel:
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("Anthropic API key not set in .env file")
    return AnthropicModel(
        api_key=api_key,
        model_name=model_name
    )

MODEL_CONFIG = {
    "Groq": create_groq_model_with_fallback,
    "Anthropic": create_anthropic_model
}

def create_model(provider_and_model: str = "Groq:deepseek-r1-distill-qwen-32b", temperature: float = 0.7) -> Any:
    if ":" in provider_and_model:
        provider, model = provider_and_model.split(":")
        if provider in MODEL_CONFIG:
            return create_groq_model_with_fallback(model, temperature)
    return create_groq_model_with_fallback("deepseek-r1-distill-qwen-32b", temperature)

# Define node classes for the graph
@dataclass
class AnalysisNode(BaseNode):
    agent: Agent
    requirements: str

    async def run(self, ctx: GraphRunContext) -> 'ArchitectureNode':
        # Initialize state if it doesn't exist
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {}
        
        # Log start of CEO analysis
        log_message(
            f"Starting CEO analysis of requirements:\n{self.requirements}",
            agent_name="CEO Agent"
        )
        
        ceo_prompt = (
            "As the CEO, analyze these requirements and create a strategic plan:\n"
            f"User Requirements: {self.requirements}\n\n"
            "Think deeply about:\n"
            "1. Strategic Vision:\n"
            "   - Core business objectives\n"
            "   - Key success metrics\n"
            "   - Critical constraints\n\n"
            "2. Team Structure:\n"
            "   - Required agent roles and responsibilities\n"
            "   - Team collaboration patterns\n"
            "   - Communication protocols\n\n"
            "3. Resource Allocation:\n"
            "   - Tool requirements and justification\n"
            "   - Process optimization opportunities\n"
            "   - Performance considerations\n\n"
            "4. Implementation Strategy:\n"
            "   - Development phases\n"
            "   - Quality assurance measures\n"
            "   - Risk mitigation plans\n\n"
            "Provide a detailed strategic plan that can be executed by the team.\n"
            "Focus on web functionality and ensure all components work together seamlessly."
        )
        
        # Log the CEO's strategic thinking process
        log_message(
            f"CEO strategic analysis in progress:\n{ceo_prompt}",
            agent_name="CEO Agent"
        )
        
        try:
            result = await self.agent.run(ceo_prompt)
            
            # Initialize state storage if needed
            if "outputs" not in ctx.state:
                ctx.state["outputs"] = {}
            
            # Store CEO's output
            if "ceo_output" not in ctx.state["outputs"]:
                ctx.state["outputs"]["ceo_output"] = []
            
            ctx.state["outputs"]["ceo_output"].append({
                "timestamp": datetime.now().isoformat(),
                "output": result.data
            })
            
            # Log the CEO's strategic decisions
            log_message(
                f"CEO strategic analysis completed. Strategic plan:\n\n{result.data}",
                agent_name="CEO Agent"
            )
            
            # Log delegation to Architecture team
            log_message(
                f"CEO delegating architecture planning:\n\n{result.data}",
                agent_name="CEO Agent",
                to_agent="Architecture Specialist"
            )
            
            return ArchitectureNode(self.agent, self.requirements, result.data)
            
        except Exception as e:
            error_msg = f"CEO strategic planning error: {str(e)}"
            log_message(error_msg, level="error", agent_name="CEO Agent")
            raise

@dataclass
class ArchitectureNode(BaseNode):
    agent: Agent
    requirements: str
    analysis: str

    async def run(self, ctx: GraphRunContext) -> 'ImplementationNode':
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}
            
        # Log receipt of analysis
        log_message(
            f"Received analysis from Requirements Analyst:\n\n{self.analysis}",
            agent_name="Architecture Specialist"
        )
        
        # Log start of architecture design
        log_message(
            f"Starting architecture design\nRequirements: {self.requirements}",
            agent_name="Architecture Specialist"
        )
        
        arch_prompt = (
            "Based on the analysis and requirements:\n"
            f"Analysis: {self.analysis}\n"
            f"Requirements: {self.requirements}\n\n"
            "Design a technical architecture using the CrewAI framework. "
            "Structure your response as follows:\n\n"
            "1. Framework Setup:\n"
            "   - Required CrewAI imports\n"
            "   - Environment configuration\n"
            "   - Core dependencies\n\n"
            "2. Agent Class Definitions:\n"
            "   - Agent initialization parameters\n"
            "   - Tool configurations\n"
            "   - LLM configurations\n\n"
            "3. Task Structures:\n"
            "   - Task class definitions\n"
            "   - Task dependencies\n"
            "   - Task execution flow\n\n"
            "4. Crew Configuration:\n"
            "   - Crew initialization\n"
            "   - Process management\n"
            "   - Resource handling\n\n"
            "Remember: Every component must map to CrewAI framework features."
        )
        
        # Log the prompt being sent to the agent
        log_message(
            f"Sending architecture prompt to agent:\n{arch_prompt}",
            agent_name="Architecture Specialist"
        )
        
        try:
            result = await self.agent.run(arch_prompt)
            
            # Store Architecture Specialist's output
            if "architecture_output" not in ctx.state["outputs"]:
                ctx.state["outputs"]["architecture_output"] = []
            
            ctx.state["outputs"]["architecture_output"].append({
                "timestamp": datetime.now().isoformat(),
                "output": result.data
            })
            
            # Log the complete architecture results
            log_message(
                f"Architecture design completed successfully. Full results:\n\n{result.data}",
                agent_name="Architecture Specialist"
            )
            
            # Log the handoff to Implementation Agent
            log_message(
                f"Sending architecture to Implementation Agent:\n\n{result.data}",
                agent_name="Architecture Specialist",
                to_agent="Implementation Agent"
            )
            
            return ImplementationNode(self.agent, self.requirements, result.data)
            
        except Exception as e:
            error_msg = f"Error during architecture design: {str(e)}"
            log_message(error_msg, level="error", agent_name="Architecture Specialist")
            raise

@dataclass
class ImplementationNode(BaseNode):
    agent: Agent
    requirements: str
    architecture: str

    async def run(self, ctx: GraphRunContext) -> 'FinalNode':
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}
            
        # Log receipt of architecture
        log_message(
            f"Received architecture from Architecture Specialist:\n\n{self.architecture}",
            agent_name="Implementation Agent"
        )
        
        # Log start of implementation planning
        log_message(
            f"Starting implementation planning\nRequirements: {self.requirements}",
            agent_name="Implementation Agent"
        )
        
        impl_prompt = (
            "Based on the CrewAI architecture and requirements:\n"
            f"Architecture: {self.architecture}\n"
            f"Requirements: {self.requirements}\n\n"
            "Create a detailed implementation plan for the CrewAI system. "
            "Structure your response as follows:\n\n"
            "1. Project Structure:\n"
            "   - File organization\n"
            "   - Module dependencies\n"
            "   - Configuration files\n\n"
            "2. Agent Implementations:\n"
            "   - Agent class code\n"
            "   - Tool integrations\n"
            "   - Role definitions\n\n"
            "3. Task Implementations:\n"
            "   - Task class code\n"
            "   - Task chaining\n"
            "   - Error handling\n\n"
            "4. Crew Setup:\n"
            "   - Crew initialization\n"
            "   - Process configuration\n"
            "   - Execution flow\n\n"
            "Remember: Follow CrewAI framework best practices and patterns."
        )
        
        # Log the prompt being sent to the agent
        log_message(
            f"Sending implementation prompt to agent:\n{impl_prompt}",
            agent_name="Implementation Agent"
        )
        
        try:
            result = await self.agent.run(impl_prompt)
            
            # Store Implementation Agent's output
            if "implementation_output" not in ctx.state["outputs"]:
                ctx.state["outputs"]["implementation_output"] = []
            
            ctx.state["outputs"]["implementation_output"].append({
                "timestamp": datetime.now().isoformat(),
                "output": result.data
            })
            
            # Log the complete implementation results
            log_message(
                f"Implementation planning completed successfully. Full results:\n\n{result.data}",
                agent_name="Implementation Agent"
            )
            
            # Log the handoff to Code Generator
            log_message(
                f"Sending implementation plan to Code Generator:\n\n{result.data}",
                agent_name="Implementation Agent",
                to_agent="Code Generator"
            )
            
            return FinalNode(self.agent, result.data)
            
        except Exception as e:
            error_msg = f"Error during implementation planning: {str(e)}"
            log_message(error_msg, level="error", agent_name="Implementation Agent")
            raise

@dataclass
class FinalNode(BaseNode):
    agent: Agent
    implementation: str

    async def run(self, ctx: GraphRunContext) -> 'QualityControlNode':
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}
        
        # Store outputs in a dictionary
        agent_outputs = {
            "CEO Agent": ctx.state["outputs"].get("ceo_output", []),
            "Architecture Specialist": ctx.state["outputs"].get("architecture_output", []),
            "Implementation Agent": ctx.state["outputs"].get("implementation_output", []),
            "Code Generator": []
        }
        
        # Log start of code generation
        log_message("Starting code generation", agent_name="Code Generator")
        
        # Parse the implementation details
        try:
            implementation_details = json.loads(self.implementation) if isinstance(self.implementation, str) else self.implementation
            log_message(
                f"Successfully parsed implementation details:\n{json.dumps(implementation_details, indent=2)}",
                agent_name="Code Generator"
            )
        except:
            implementation_details = {"raw": self.implementation}
            log_message(
                "Failed to parse implementation as JSON, using raw format",
                level="warning",
                agent_name="Code Generator"
            )
        
        final_prompt = (
            "You are a CrewAI framework expert. Generate a complete, executable CrewAI Python script based on this implementation plan:\n"
            f"{json.dumps(implementation_details, indent=2)}\n\n"
            "Follow this exact structure for the CrewAI implementation:\n\n"
            "```python\n"
            "import streamlit as st\n"
            "from crewai import Agent, Task, Crew, Process\n"
            "from langchain_groq import ChatGroq\n"
            "from dotenv import load_dotenv\n"
            "import os\n"
            "import logging\n\n"
            "# Configure logging\n"
            "logging.basicConfig(\n"
            "    level=logging.INFO,\n"
            "    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n"
            ")\n"
            "logger = logging.getLogger(__name__)\n\n"
            "def create_llm(model_name: str = 'deepseek-r1-distill-qwen-32b', temperature: float = 0.7):\n"
            "    api_key = os.getenv('GROQ_API_KEY')\n"
            "    if not api_key:\n"
            "        raise ValueError('Groq API key not found in environment')\n"
            "    return ChatGroq(\n"
            "        groq_api_key=api_key,\n"
            "        model_name=model_name,\n"
            "        temperature=temperature\n"
            "    )\n\n"
            "def load_agents():\n"
            "    logger.info('Creating agents with specific roles and capabilities')\n"
            "    agents = [\n"
            "        Agent(\n"
            "            role='Role Name',\n"
            "            goal='Specific goal',\n"
            "            backstory='Detailed backstory',\n"
            "            verbose=True,\n"
            "            allow_delegation=True,\n"
            "            llm=create_llm()\n"
            "        ),\n"
            "        # More agents...\n"
            "    ]\n"
            "    return agents\n\n"
            "def load_tasks(agents):\n"
            "    logger.info('Creating tasks with specific assignments to agents')\n"
            "    tasks = [\n"
            "        Task(\n"
            "            description='Task description',\n"
            "            expected_output='Expected output format',\n"
            "            agent=next(agent for agent in agents if agent.role == 'Role Name'),\n"
            "            async_execution=False\n"
            "        ),\n"
            "        # More tasks...\n"
            "    ]\n"
            "    return tasks\n\n"
            "def main():\n"
            "    st.title('CrewAI System')\n"
            "    \n"
            "    try:\n"
            "        logger.info('Initializing CrewAI system')\n"
            "        agents = load_agents()\n"
            "        tasks = load_tasks(agents)\n"
            "        \n"
            "        crew = Crew(\n"
            "            agents=agents,\n"
            "            tasks=tasks,\n"
            "            process='sequential',\n"
            "            verbose=True,\n"
            "            memory=False,\n"
            "            cache=True,\n"
            "            max_rpm=1000\n"
            "        )\n"
            "        \n"
            "        with st.spinner('Running crew...'):\n"
            "            logger.info('Starting crew execution')\n"
            "            result = crew.kickoff()\n"
            "            \n"
            "            with st.expander('Final output', expanded=True):\n"
            "                if hasattr(result, 'raw'):\n"
            "                    st.write(result.raw)\n"
            "                else:\n"
            "                    st.write(result)\n"
            "            \n"
            "            logger.info('Crew execution completed successfully')\n"
            "            \n"
            "    except Exception as e:\n"
            "        error_msg = f'An error occurred: {str(e)}'\n"
            "        logger.error(error_msg)\n"
            "        st.error(error_msg)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "```\n\n"
            "Generate a complete CrewAI implementation following this exact structure, but with your specific:\n"
            "1. Agent roles, goals, and backstories based on the implementation plan\n"
            "2. Task descriptions and assignments\n"
            "3. Proper tool integrations\n"
            "4. Appropriate process type (sequential/hierarchical)\n"
            "5. Detailed logging throughout the execution\n\n"
            "The code must be fully functional and include proper error handling and logging.\n"
            "Output only the Python code, no explanations or markdown."
        )
        
        # Log the prompt being sent to the agent
        log_message(
            f"Sending code generation prompt to agent:\n{final_prompt}",
            agent_name="Code Generator"
        )
        
        try:
            result = await self.agent.run(final_prompt)
            
            # Store Code Generator's output
            if "code_generator_output" not in ctx.state["outputs"]:
                ctx.state["outputs"]["code_generator_output"] = []
            
            ctx.state["outputs"]["code_generator_output"].append({
                "timestamp": datetime.now().isoformat(),
                "output": result.data
            })
            
            agent_outputs["Code Generator"] = ctx.state["outputs"]["code_generator_output"]
            
            # Update context state
            ctx.state["code_generator_output"] = agent_outputs["Code Generator"]
            
            # Log the initial code generation result
            log_message(
                f"Initial code generation completed. Raw output:\n\n{result.data}",
                agent_name="Code Generator"
            )
            
            try:
                # Get the code string
                code_str = result.data if isinstance(result.data, str) else str(result.data)
                
                # Clean up the code string
                if "```python" in code_str:
                    code_str = code_str.split("```python")[1].split("```")[0].strip()
                elif "```" in code_str:
                    code_str = code_str.split("```")[1].strip()
                
                # Validate the code syntax
                compile(code_str, '<string>', 'exec')
                
                # Verify essential imports
                required_imports = [
                    "from crewai import Agent, Task, Crew",
                    "from langchain_groq import ChatGroq",
                    "import logging"
                ]
                
                for imp in required_imports:
                    if imp not in code_str:
                        code_str = imp + "\n" + code_str
                
                # Verify essential components
                required_components = [
                    "load_agents()",
                    "load_tasks(agents)",
                    "Crew(",
                    "crew.kickoff()",
                    "logging.basicConfig"
                ]
                
                missing_components = [comp for comp in required_components if comp not in code_str]
                if missing_components:
                    raise ValueError(f"Missing essential components: {', '.join(missing_components)}")
                
                # Log successful validation
                log_message(
                    f"Code validation successful. Final code:\n\n```python\n{code_str}\n```",
                    agent_name="Code Generator"
                )
                
                # Pass to QualityControlNode with collected outputs
                return QualityControlNode(
                    self.agent,
                    code_str,
                    implementation_details,
                    agent_outputs
                )
                
            except Exception as e:
                # Log validation error
                error_msg = f"Error in generated code: {str(e)}"
                log_message(
                    f"{error_msg}\nAttempting to clean up code...",
                    level="error",
                    agent_name="Code Generator"
                )
                
                try:
                    if "invalid syntax" in str(e):
                        # Clean up the code and try again
                        clean_code = "\n".join(line for line in code_str.split("\n") 
                                             if line.strip() and not line.strip().startswith(("#", "//", "/*")))
                        compile(clean_code, '<string>', 'exec')
                        log_message(
                            f"Code cleanup successful. Cleaned code:\n\n```python\n{clean_code}\n```",
                            agent_name="Code Generator"
                        )
                        return QualityControlNode(
                            self.agent,
                            clean_code,
                            implementation_details,
                            agent_outputs
                        )
                    else:
                        error_code = f"# Error in generated code: {str(e)}\n# Please review the implementation plan and try again.\n\n{code_str}"
                        log_message(
                            f"Unable to fix code. Error code:\n\n```python\n{error_code}\n```",
                            level="error",
                            agent_name="Code Generator"
                        )
                        return QualityControlNode(
                            self.agent,
                            error_code,
                            implementation_details,
                            agent_outputs
                        )
                except:
                    error_code = f"# Error in generated code: {str(e)}\n# Original code with issues:\n{code_str}"
                    log_message(
                        f"Failed to clean up code. Error code:\n\n```python\n{error_code}\n```",
                        level="error",
                        agent_name="Code Generator"
                    )
                    return QualityControlNode(
                        self.agent,
                        error_code,
                        implementation_details,
                        agent_outputs
                    )
                    
        except Exception as e:
            error_msg = f"Error during code generation: {str(e)}"
            log_message(error_msg, level="error", agent_name="Code Generator")
            raise

@dataclass
class QualityControlNode(BaseNode):
    agent: Agent
    generated_code: str
    implementation_details: Any
    agent_outputs: Dict[str, Any]

    async def run(self, ctx: GraphRunContext) -> End[Dict[str, Any]]:
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}

        log_message(
            "Starting CEO quality control review",
            agent_name="CEO Agent"
        )

        # Format agent outputs for review
        formatted_outputs = {}
        for agent_name, outputs in self.agent_outputs.items():
            formatted_outputs[agent_name] = [
                {
                    "timestamp": output.get("timestamp", ""),
                    "content": str(output.get("output", ""))
                } for output in outputs
            ]

        # Format implementation details
        if isinstance(self.implementation_details, dict):
            impl_details = self.implementation_details
        else:
            impl_details = {"raw": str(self.implementation_details)}

        review_prompt = f"""As the CEO, perform a comprehensive review of the entire AI system development:

1. Review all agent outputs and interactions:
{json.dumps(formatted_outputs, indent=2)}

2. Review the implementation details:
{json.dumps(impl_details, indent=2)}

3. Review the generated code:
```python
{self.generated_code}
```

Evaluate the following aspects:
1. Requirements Alignment:
   - Does the solution fully address user requirements?
   - Are there any missing features?
   - Is the web functionality properly implemented?

2. Code Quality:
   - Is the code well-structured and maintainable?
   - Are there proper error handlers?
   - Is logging comprehensive?
   - Is the web interface user-friendly?

3. Agent Collaboration:
   - Did all agents perform their roles effectively?
   - Was communication between agents clear?
   - Were tasks properly delegated and executed?

4. Technical Implementation:
   - Is the CrewAI framework used correctly?
   - Are tools integrated properly?
   - Is the web functionality robust?

Provide your evaluation in this JSON format:
{{
    "meets_requirements": true/false,
    "code_quality": true/false,
    "agent_performance": true/false,
    "technical_implementation": true/false,
    "issues": ["list of issues if any"],
    "recommendations": ["list of recommendations if any"],
    "needs_iteration": true/false,
    "final_approval": true/false
}}

If any aspect fails, provide specific details about what needs improvement."""

        try:
            review_result = await self.agent.run(review_prompt)
            
            # Store CEO's review output
            if "ceo_review" not in ctx.state["outputs"]:
                ctx.state["outputs"]["ceo_review"] = []
            
            ctx.state["outputs"]["ceo_review"].append({
                "timestamp": datetime.now().isoformat(),
                "output": review_result.data
            })
            
            try:
                # Clean up the response to ensure it's valid JSON
                if isinstance(review_result.data, str):
                    # Extract JSON if it's within a code block
                    if "```json" in review_result.data:
                        json_str = review_result.data.split("```json")[1].split("```")[0].strip()
                    elif "```" in review_result.data:
                        json_str = review_result.data.split("```")[1].strip()
                    else:
                        # Find the first { and last }
                        start = review_result.data.find("{")
                        end = review_result.data.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_str = review_result.data[start:end]
                        else:
                            json_str = review_result.data
                    evaluation = json.loads(json_str)
                else:
                    evaluation = review_result.data
                
                log_message(
                    f"CEO Review Results:\n{json.dumps(evaluation, indent=2)}",
                    agent_name="CEO Agent"
                )

                if evaluation.get("needs_iteration", False):
                    log_message(
                        f"Quality standards not met. Issues:\n{json.dumps(evaluation.get('issues', []), indent=2)}",
                        level="warning",
                        agent_name="CEO Agent"
                    )
                    
                    # If iteration is needed, create new implementation details with fixes
                    iteration_prompt = f"""Based on the review findings:
                    Issues: {json.dumps(evaluation.get('issues', []), indent=2)}
                    Recommendations: {json.dumps(evaluation.get('recommendations', []), indent=2)}
                    
                    Generate updated implementation details that address all issues."""
                    
                    iteration_result = await self.agent.run(iteration_prompt)
                    
                    # Store iteration result
                    if "iteration_output" not in ctx.state["outputs"]:
                        ctx.state["outputs"]["iteration_output"] = []
                    
                    ctx.state["outputs"]["iteration_output"].append({
                        "timestamp": datetime.now().isoformat(),
                        "output": iteration_result.data
                    })
                    
                    # Return to ImplementationNode for another iteration
                    return ImplementationNode(self.agent, "Updated requirements", iteration_result.data)
                
                if evaluation.get("final_approval", False):
                    log_message(
                        "CEO approved final output. Delivering to user.",
                        agent_name="CEO Agent"
                    )
                    return End({
                        "code": self.generated_code,
                        "evaluation": evaluation,
                        "agent_outputs": self.agent_outputs
                    })
                else:
                    log_message(
                        "CEO rejected output. Starting new iteration.",
                        level="warning",
                        agent_name="CEO Agent"
                    )
                    return ImplementationNode(self.agent, "Rejected implementation", self.implementation_details)
                
            except json.JSONDecodeError as e:
                log_message(
                    f"Error parsing CEO review results: {str(e)}. Raw response:\n{review_result.data}",
                    level="error",
                    agent_name="CEO Agent"
                )
                # Create a default evaluation with error information
                evaluation = {
                    "meets_requirements": False,
                    "code_quality": False,
                    "agent_performance": False,
                    "technical_implementation": False,
                    "issues": [f"Error parsing review results: {str(e)}"],
                    "recommendations": ["Review and fix the JSON formatting in the CEO review response"],
                    "needs_iteration": True,
                    "final_approval": False
                }
                return ImplementationNode(self.agent, "Error in review", self.implementation_details)
                
        except Exception as e:
            error_msg = f"Error during CEO review: {str(e)}"
            log_message(error_msg, level="error", agent_name="CEO Agent")
            raise

# Agent System Implementation
class AIAgentSystem:
    def __init__(self):
        log_message("Initializing AI Agent System", agent_name="System")
        self.model = create_model()
        self.tools = self._load_tools()
        self.graph = Graph(nodes=[AnalysisNode, ArchitectureNode, ImplementationNode, FinalNode, QualityControlNode])
        self.tasks = []
        log_message("AI Agent System initialized successfully", agent_name="System")
        
    def _load_tools(self) -> List[Tool]:
        log_message("Loading available tools", agent_name="System")
        available_tools = [
            Tool(name="web_scraper", function=self._web_scraper),
            Tool(name="search_tool", function=self._search_tool),
            Tool(name="website_search", function=self._website_search),
            Tool(name="api_caller", function=self._api_caller),
            Tool(name="code_interpreter", function=self._code_interpreter),
            Tool(name="file_reader", function=self._file_reader),
            Tool(name="file_writer", function=self._file_writer),
        ]
        log_message(f"Loaded {len(available_tools)} tools", agent_name="System")
        return available_tools

    @staticmethod
    def _web_scraper(url: str) -> Dict[str, Any]:
        # Implement web scraping logic
        pass

    @staticmethod
    def _search_tool(query: str) -> Dict[str, Any]:
        # Implement search logic
        pass

    @staticmethod
    def _website_search(url: str, query: str) -> Dict[str, Any]:
        # Implement website search logic
        pass

    @staticmethod
    def _api_caller(endpoint: str, method: str, **kwargs) -> Dict[str, Any]:
        # Implement API calling logic
        pass

    @staticmethod
    def _code_interpreter(code: str, **kwargs) -> Dict[str, Any]:
        # Implement code interpretation logic
        pass

    @staticmethod
    def _file_reader(path: str) -> Dict[str, Any]:
        # Implement file reading logic
        pass

    @staticmethod
    def _file_writer(path: str, content: str) -> Dict[str, Any]:
        # Implement file writing logic
        pass

    def create_agents(self) -> (List[Agent], Agent):
        log_message("CEO initiating team creation", agent_name="CEO Agent")
        
        try:
            # Create models for each agent type with fallback handling
            ceo_model = create_groq_model_with_fallback("deepseek-r1-distill-qwen-32b", temperature=0.7)
            architecture_model = create_groq_model_with_fallback("llama-3.3-70b-versatile", temperature=0.7)
            implementation_model = create_groq_model_with_fallback("qwen-2.5-coder-32b", temperature=0.7)
            code_generator_model = create_groq_model_with_fallback("qwen-2.5-coder-32b", temperature=0.7)
            
            ceo_agent = Agent(
                model=ceo_model,
                name="CEO Agent",
                system_prompt=(
                    "You are the CEO of an AI development organization. "
                    "Your role is to:\n"
                    "1. Analyze user requirements with deep strategic thinking\n"
                    "2. Develop comprehensive implementation strategies\n"
                    "3. Delegate tasks to specialized agents effectively\n"
                    "4. Ensure all components work together seamlessly\n"
                    "5. Maintain focus on web functionality and user experience\n"
                    "Lead your team to create production-ready solutions that exceed expectations."
                ),
                tools=self.tools
            )
            
            # Create specialized agents under CEO's direction
            agents = [
                Agent(
                    model=architecture_model,
                    name="Architecture Specialist",
                    system_prompt=(
                        "Design robust system architectures focusing on:\n"
                        "1. Web-centric component organization\n"
                        "2. Scalable agent interactions\n"
                        "3. Efficient tool integration patterns\n"
                        "4. Performance optimization for web deployment"
                    ),
                    tools=self.tools
                ),
                Agent(
                    model=implementation_model,
                    name="Implementation Agent",
                    system_prompt=(
                        "Following CEO's strategy, implement solutions with:\n"
                        "1. Clean, maintainable code structure\n"
                        "2. Robust error handling\n"
                        "3. Comprehensive logging\n"
                        "4. Web-optimized performance"
                    ),
                    tools=self.tools
                ),
                Agent(
                    model=code_generator_model,
                    name="Code Generator",
                    system_prompt=(
                        "Create production-ready code with:\n"
                        "1. Web-centric architecture\n"
                        "2. Robust error handling\n"
                        "3. Performance optimization\n"
                        "4. Comprehensive documentation"
                    ),
                    tools=self.tools
                )
            ]
            
            log_message(f"CEO completed team assembly with {len(agents)} specialized agents", agent_name="CEO Agent")
            return agents, ceo_agent
            
        except Exception as e:
            log_message(f"Error during agent creation: {str(e)}", level="error", agent_name="CEO Agent")
            # Create all agents with fallback model
            fallback_model = create_groq_model_with_fallback("llama-3.3-70b-versatile", temperature=0.7)
            
            ceo_agent = Agent(
                model=fallback_model,
                name="CEO Agent",
                system_prompt=(
                    "You are the CEO of an AI development organization. "
                    "Your role is to:\n"
                    "1. Analyze user requirements with deep strategic thinking\n"
                    "2. Develop comprehensive implementation strategies\n"
                    "3. Delegate tasks to specialized agents effectively\n"
                    "4. Ensure all components work together seamlessly\n"
                    "5. Maintain focus on web functionality and user experience"
                ),
                tools=self.tools
            )
            
            agents = [
                Agent(
                    model=fallback_model,
                    name="Architecture Specialist",
                    system_prompt=(
                        "Design robust system architectures focusing on:\n"
                        "1. Web-centric component organization\n"
                        "2. Scalable agent interactions\n"
                        "3. Efficient tool integration patterns\n"
                        "4. Performance optimization for web deployment"
                    ),
                    tools=self.tools
                ),
                Agent(
                    model=fallback_model,
                    name="Implementation Agent",
                    system_prompt=(
                        "Implement solutions with:\n"
                        "1. Clean, maintainable code structure\n"
                        "2. Robust error handling\n"
                        "3. Comprehensive logging\n"
                        "4. Web-optimized performance"
                    ),
                    tools=self.tools
                ),
                Agent(
                    model=fallback_model,
                    name="Code Generator",
                    system_prompt=(
                        "Create production-ready code with:\n"
                        "1. Web-centric architecture\n"
                        "2. Robust error handling\n"
                        "3. Performance optimization\n"
                        "4. Comprehensive documentation"
                    ),
                    tools=self.tools
                )
            ]
            
            log_message("Created agents with fallback model due to initialization error", level="warning", agent_name="CEO Agent")
            return agents, ceo_agent

    def process_request(self, user_requirements: str) -> str:
        log_message("CEO initiating project execution", agent_name="CEO Agent")
        agents, ceo_agent = self.create_agents()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Start with CEO's analysis
            analysis_node = AnalysisNode(ceo_agent, user_requirements)
            result = loop.run_until_complete(self.process_request_async(analysis_node))
            return result
        finally:
            loop.close()
            log_message("CEO concluding project execution", agent_name="CEO Agent")

    async def process_request_async(self, initial_node: BaseNode) -> str:
        log_message("Starting asynchronous request processing", agent_name="System")
        result, _ = await self.graph.run(initial_node)
        log_message("Request processing completed", agent_name="System")
        return result

# Streamlit UI
def main():
    # Initialize Streamlit state at the start of the app
    initialize_streamlit_state()
    
    st.title("Crew AI Generator")
    st.markdown("""
    ### Welcome to the Crew AI Generator
    Enter your requirements to generate a fully functional Python script that builds a Crew AI system—
    a system of cooperating agents with defined roles, tool integrations, and execution flow.
    """)

    # Create tabs for input, review, and logs
    input_tab, review_tab, logs_tab = st.tabs(["Create Crew AI", "CEO Review", "System Logs"])

    with input_tab:
        user_prompt = st.text_area(
            "Describe your AI requirements:",
            placeholder="Example: I need a Crew AI system that processes and analyzes data to generate insights.",
            help="Describe what you want your AI system to do"
        )

        if st.button("Generate Crew AI Code", type="primary"):
            if not user_prompt:
                st.warning("Please enter your requirements.")
                return

            with st.spinner("Generating and reviewing Crew AI code..."):
                try:
                    agent_system = AIAgentSystem()
                    result = agent_system.process_request(user_prompt)
                    
                    if isinstance(result, dict):
                        st.session_state.current_result = result
                        st.success("Crew AI code generated and reviewed by CEO!")
                        
                        # Switch to review tab
                        st.experimental_set_query_params(active_tab="review")
                    else:
                        st.error("Unexpected result format. Please try again.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    log_message(traceback.format_exc(), level="error")

    with review_tab:
        if hasattr(st.session_state, 'current_result'):
            result = st.session_state.current_result
            
            # Display CEO's evaluation
            st.markdown("### 📊 CEO's Quality Review")
            evaluation = result.get('evaluation', {})
            
            # Create metrics for key aspects
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Requirements Met", "✅" if evaluation.get('meets_requirements') else "❌")
            with col2:
                st.metric("Code Quality", "✅" if evaluation.get('code_quality') else "❌")
            with col3:
                st.metric("Agent Performance", "✅" if evaluation.get('agent_performance') else "❌")
            with col4:
                st.metric("Technical Implementation", "✅" if evaluation.get('technical_implementation') else "❌")
            
            # Display any issues or recommendations
            if evaluation.get('issues'):
                with st.expander("📋 Issues Identified", expanded=True):
                    for issue in evaluation['issues']:
                        st.warning(issue)
            
            if evaluation.get('recommendations'):
                with st.expander("💡 Recommendations", expanded=True):
                    for rec in evaluation['recommendations']:
                        st.info(rec)
            
            # Display the final code if approved
            if evaluation.get('final_approval'):
                st.success("✅ CEO has approved this implementation!")
                with st.expander("View Approved Code", expanded=True):
                    st.code(result['code'], language="python")
                    
                    # Add a download button for the code
                    st.download_button(
                        label="Download Approved Python Script",
                        data=result['code'],
                        file_name="approved_crew_ai_system.py",
                        mime="text/plain"
                    )
            else:
                st.error("❌ This implementation requires iteration. Check the logs for details.")
            
            # Display agent outputs in a collapsible section
            with st.expander("View All Agent Outputs", expanded=False):
                st.json(result['agent_outputs'])
        
        else:
            st.info("No code has been generated yet. Use the 'Create Crew AI' tab to generate code.")

    with logs_tab:
        st.markdown("### 📊 System Logs")
        
        # Add log filtering options
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_levels = st.multiselect(
                "Filter by Log Level",
                ["INFO", "WARNING", "ERROR"],
                default=["INFO", "WARNING", "ERROR"]
            )
        with col2:
            selected_agents = st.multiselect(
                "Filter by Agent",
                ["System", "Code Generator", "Requirements Analyst", "Architecture Specialist", 
                 "Implementation Agent"],
                default=["System", "Code Generator", "Requirements Analyst", "Architecture Specialist", 
                        "Implementation Agent"]
            )
        with col3:
            show_communications = st.checkbox("Show Communications Only", value=False)
        
        # Create a container for logs if it doesn't exist
        if 'log_container' not in st.session_state:
            st.session_state.log_container = st.container()
        
        # Display logs in the container with improved styling
        with st.session_state.log_container:
            if 'logs' in st.session_state:
                for log in reversed(st.session_state.logs):  # Show newest logs first
                    # Apply filters
                    if log['level'] not in selected_levels:
                        continue
                    if log['agent'] and log['agent'] not in selected_agents:
                        continue
                    if show_communications and not log.get('is_communication', False):
                        continue
                    
                    # Create styled log entry
                    with st.container():
                        # Style based on log level and communication type
                        level_color = {
                            'ERROR': '#ff5252',
                            'WARNING': '#ffd740',
                            'INFO': '#2196f3'
                        }.get(log['level'], '#9e9e9e')
                        
                        is_communication = log.get('is_communication', False)
                        bg_color = '#f3f9ff' if is_communication else 'white'
                        
                        # Create header with timestamp, level, and agents
                        header = f"<div style='background-color: {bg_color}; padding: 8px; border-radius: 4px; margin: 4px 0;'>"
                        header += f"<span style='font-weight: bold;'>{log['timestamp']}</span> "
                        header += f"<span style='background-color: {level_color}; color: white; padding: 2px 8px; border-radius: 4px; margin: 0 4px;'>{log['level']}</span>"
                        
                        # Add agent communication visualization
                        if log.get('to_agent'):
                            header += f"<span style='color: #666;'>{log['agent']}</span> "
                            header += f"<span style='color: #2196f3; margin: 0 4px;'>→</span> "
                            header += f"<span style='color: #666;'>{log['to_agent']}</span>"
                        elif log['agent']:
                            header += f"<span style='color: #666;'>{log['agent']}</span>"
                        header += "</div>"
                        
                        st.markdown(header, unsafe_allow_html=True)
                        
                        # Display message with appropriate styling
                        message_style = (
                            f"background-color: {bg_color}; "
                            "padding: 8px 16px; "
                            "border-radius: 4px; "
                            "margin: 4px 0; "
                            f"border-left: 4px solid {level_color};"
                        )
                        st.markdown(f"<div style='{message_style}'>{log['message']}</div>", unsafe_allow_html=True)
                        
                        # Add subtle separator
                        st.markdown("<hr style='margin: 8px 0; opacity: 0.1;'>", unsafe_allow_html=True)

if __name__ == '__main__':
    main() 

     
