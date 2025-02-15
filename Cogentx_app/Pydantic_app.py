import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="AI Agent Creator",
    page_icon="🤖",
    layout="wide"
)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
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

def log_message(message: str, level: str = "info", agent_name: str = None):
    """Log messages to CLI, file, and Streamlit interface with enhanced formatting"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a formatted message with agent name if provided
    if agent_name:
        formatted_message = f"[{agent_name}] {message}"
    else:
        formatted_message = message
    
    # Log to CLI and file with different colors based on level
    if level.lower() == "error":
        logger.error(formatted_message)
    elif level.lower() == "warning":
        logger.warning(formatted_message)
    else:
        logger.info(formatted_message)
    
    # Only update Streamlit if we're in a Streamlit context
    try:
        if st._is_running_with_streamlit:
            if 'logs' not in st.session_state:
                initialize_streamlit_state()
            
            log_entry = {
                'timestamp': timestamp,
                'level': level.upper(),
                'agent': agent_name,
                'message': message
            }
            
            st.session_state.logs.append(log_entry)
            
            # Also write to txt file directly for backup
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} - {agent_name if agent_name else 'System'} - {level.upper()} - {message}\n")
    except:
        pass  # Not in Streamlit context, skip UI updates

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Create a Streamlit container for logs
if 'log_container' not in st.session_state:
    st.session_state.log_container = st.empty()

# OpenAI API Key Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    log_message("OpenAI API key not found in environment variables", level="error")
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

# Set the API key in environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
def create_openai_model(model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> OpenAIModel:
    return OpenAIModel(
        api_key=OPENAI_API_KEY,
        model_name=model_name
    )

def create_groq_model(model_name: str = "mixtral-8x7b-32768", temperature: float = 0.7) -> GroqModel:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("Groq API key not set in .env file")
    return GroqModel(
        api_key=api_key,
        model_name=model_name
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
    "OpenAI": create_openai_model,
    "Groq": create_groq_model,
    "Anthropic": create_anthropic_model
}

def create_model(provider_and_model: str = "OpenAI:gpt-4o-mini", temperature: float = 0.7) -> Any:
    if ":" in provider_and_model:
        provider, model = provider_and_model.split(":")
        if provider in MODEL_CONFIG:
            return MODEL_CONFIG[provider](model, temperature)
    return create_openai_model("gpt-4o-mini", temperature)

# Define node classes for the graph
@dataclass
class AnalysisNode(BaseNode):
    agent: Agent
    requirements: str

    async def run(self, ctx: GraphRunContext) -> 'ArchitectureNode':
        log_message(
            f"Starting comprehensive analysis of requirements: {self.requirements}",
            agent_name="Requirements Analyst"
        )
        
        # Create the analysis prompt without f-string nesting
        analysis_prompt = (
            "Perform a detailed analysis of the following user requirements:\n"
            f"User Requirements: {self.requirements}\n\n"
            "Follow these steps in your analysis:\n\n"
            "1. Core Requirements Analysis:\n"
            "   - Identify explicit requirements\n"
            "   - Identify implicit requirements\n"
            "   - List any potential edge cases or special considerations\n"
            "   - Determine priority levels for each requirement\n\n"
            "2. Agent Requirements:\n"
            "   - Determine minimum number of agents needed\n"
            "   - Identify specific roles required\n"
            "   - Define interaction patterns between agents\n"
            "   - Specify expertise level needed for each agent\n\n"
            "3. Tool Selection:\n"
            "   - Match each requirement with necessary tools\n"
            "   - Justify each tool selection\n"
            "   - Identify any custom tool needs\n"
            "   - Available Tools List:\n"
            "     - web_scraper - for web scraping\n"
            "     - search_tool - for search results\n"
            "     - website_search - for website searching\n"
            "     - api_caller - for API integrations\n"
            "     - code_interpreter - for code interpretation\n"
            "     - file_reader - for reading files\n"
            "     - file_writer - for writing files\n"
            "   - Available Tools List:\n"
            "     - web_scraper - for web scraping\n"
            "     - search_tool - for search results\n"
            "     - website_search - for website searching\n"
            "     - api_caller - for API integrations\n"
            "     - code_interpreter - for code interpretation\n"
            "     - file_reader - for reading files\n"
            "     - file_writer - for writing files\n\n"
            "4. Process Analysis:\n"
            "   - Determine optimal process type (sequence/horizontal)\n"
            "   - Identify critical paths\n"
            "   - Define dependencies\n"
            "   - Estimate resource requirements\n\n"
            "5. Constraints and Limitations:\n"
            "   - Identify technical constraints\n"
            "   - List resource limitations\n"
            "   - Note any timing requirements\n"
            "   - Specify performance expectations\n\n"
            "6. Success Criteria:\n"
            "   - Define measurable outcomes\n"
            "   - List validation methods\n"
            "   - Specify quality metrics\n"
            "   - Outline acceptance criteria\n\n"
            "Format your response as follows:\n"
            "{\n"
            '    "core_requirements": {\n'
            '        "explicit": ["list of explicit requirements"],\n'
            '        "implicit": ["list of implicit requirements"],\n'
            '        "edge_cases": ["list of edge cases"],\n'
            '        "priorities": {"high": [], "medium": [], "low": []}\n'
            "    },\n"
            '    "agent_needs": {\n'
            '        "count": "number",\n'
            '        "roles": ["list of required roles"],\n'
            '        "interactions": ["list of interaction patterns"],\n'
            '        "expertise_levels": {"role": "required expertise"}\n'
            "    },\n"
            '    "tool_selection": {\n'
            '        "tool_mapping": {"requirement": "tool"},\n'
            '        "justifications": {"tool": "justification"},\n'
            '        "custom_needs": ["list of custom tool needs"]\n'
            "    },\n"
            '    "process_details": {\n'
            '        "type": "sequence/horizontal",\n'
            '        "critical_paths": ["list of critical paths"],\n'
            '        "dependencies": ["list of dependencies"],\n'
            '        "resource_needs": ["list of resource requirements"]\n'
            "    },\n"
            '    "constraints": {\n'
            '        "technical": ["list of technical constraints"],\n'
            '        "resources": ["list of resource limitations"],\n'
            '        "timing": ["list of timing requirements"],\n'
            '        "performance": ["list of performance expectations"]\n'
            "    },\n"
            '    "success_criteria": {\n'
            '        "outcomes": ["list of measurable outcomes"],\n'
            '        "validation": ["list of validation methods"],\n'
            '        "metrics": ["list of quality metrics"],\n'
            '        "acceptance": ["list of acceptance criteria"]\n'
            "    }\n"
            "}\n\n"
            "IMPORTANT:\n"
            "1. Be specific and detailed in your analysis\n"
            "2. Every element must directly relate to the user requirements\n"
            "3. Ensure all tool selections are from the available list\n"
            "4. Provide clear justifications for each decision\n"
            "5. Focus on practical, implementable solutions"
        )
        
        result = await self.agent.run(analysis_prompt)
        
        log_message(
            "Completed comprehensive requirements analysis",
            agent_name="Requirements Analyst"
        )
        return ArchitectureNode(self.agent, self.requirements, result.data)

@dataclass
class ArchitectureNode(BaseNode):
    agent: Agent
    requirements: str
    analysis: str

    async def run(self, ctx: GraphRunContext) -> 'ImplementationNode':
        log_message(
            f"Starting architecture design based on analysis: {self.analysis}",
            agent_name="Architecture Specialist"
        )
        result = await self.agent.run(
            f"""Based on the initial analysis and {self.requirements}, create a minimal technical specification.
            Write in clear text format:
            1. Minimum necessary components
            2. Tool selection (ONLY from the available tools list - if needed functionality isn't available in standard tools, specify how to implement it using api_caller or code_interpreter)
            3. Efficient process flow
            4. Resource optimization

            Available Tools List:
            - web_scraper - for web scraping
            - search_tool - for search results
            - website_search - for website searching
            - api_caller - for API integrations
            - code_interpreter - for code interpretation
            - file_reader - for reading files
            - file_writer - for writing files

            Remember: If you need functionality not covered by standard tools, use:
            - api_caller - for external API integrations
            - code_interpreter - for custom code execution
            - Do NOT suggest external tools outside the available list"""
        )
        log_message(
            f"Completed architecture design: {result.data}",
            agent_name="Architecture Specialist"
        )
        return ImplementationNode(self.agent, self.requirements, result.data)

@dataclass
class ImplementationNode(BaseNode):
    agent: Agent
    requirements: str
    architecture: str

    async def run(self, ctx: GraphRunContext) -> 'FinalNode':
        log_message(
            f"Starting implementation planning based on architecture: {self.architecture}",
            agent_name="Implementation Agent"
        )
        result = await self.agent.run(
            f"""Create an optimized implementation plan that addresses ONLY what's needed for {self.requirements}.
            Write in clear text format:
            1. Minimal agent usage
            2. Tool utilization (ONLY use tools from the available list - use api_caller or code_interpreter for custom needs)
            3. Resource optimization
            4. Clear task dependencies

            Available Tools List:
            - web_scraper - for web scraping
            - search_tool - for search results
            - website_search - for website searching
            - api_caller - for API integrations
            - code_interpreter - for code interpretation
            - file_reader - for reading files
            - file_writer - for writing files

            Remember: Stay within the available tools ecosystem. If you need custom functionality:
            1. Use api_caller for API integrations
            2. Use code_interpreter for custom code execution
            3. DO NOT suggest external tools"""
        )
        log_message(
            f"Completed implementation planning: {result.data}",
            agent_name="Implementation Agent"
        )
        return FinalNode(self.agent, result.data)

@dataclass
class FinalNode(BaseNode):
    agent: Agent
    implementation: str

    async def run(self, ctx: GraphRunContext) -> End[Dict[str, Any]]:
        log_message(
            f"Starting solution generation based on implementation: {self.implementation}",
            agent_name="CEO Agent"
        )
        
        # Extract the implementation details
        implementation_str = str(self.implementation)
        
        result = await self.agent.run(
            f"""Based on the following implementation plan and previous analyses:

Implementation Details:
{implementation_str}

Create a solution in strict JSON format that EXACTLY implements the above plan. The solution must:
1. Follow the exact crew structure from the implementation
2. Include only the specific tools mentioned in the implementation
3. Create agents with roles and responsibilities as defined in the implementation
4. Define tasks exactly as outlined in the implementation plan

Use this structure:
{{
    "crew_details": {{
        "name": "name based on implementation focus",
        "process": "sequence/horizontal as defined in implementation",
        "agents": ["agents from implementation"],
        "tasks": ["tasks from implementation"],
        "manager_llm": "model name",
        "manager_agent": "managing agent name",
        "verbose": true/false,
        "memory": "enabled/disabled",
        "cache": "enabled/disabled",
        "planning": "planning approach from implementation",
        "max_rpm": 1000
    }},
    "tools": [
        {{
            "name": "tool from implementation",
            "purpose": "specific purpose from implementation"
        }}
    ],
    "agents": [
        {{
            "name": "agent name from implementation",
            "role": "role from implementation",
            "backstory": "backstory matching implementation",
            "goal": "goal from implementation",
            "allow_delegation": true/false based on implementation,
            "verbose": true/false,
            "cache": "enabled/disabled",
            "llm": "model name",
            "temperature": 0.7,
            "max_iterations": 5,
            "tools": ["tools assigned in implementation"]
        }}
    ],
    "tasks": [
        {{
            "name": "task name from implementation",
            "description": "task description from implementation",
            "expected_output": "output defined in implementation",
            "assigned_agent": "agent assigned in implementation",
            "async_execution": true/false based on implementation,
            "context_requirements": {{
                "async": ["async requirements from implementation"],
                "sync": ["sync requirements from implementation"]
            }}
        }}
    ]
}}

Available Tools List:
- web_scraper - for web scraping
- search_tool - for search results
- website_search - for website searching
- api_caller - for API integrations
- code_interpreter - for code interpretation
- file_reader - for reading files
- file_writer - for writing files

IMPORTANT: 
1. Respond ONLY with valid JSON
2. Follow the exact structure above
3. Make sure every component directly matches the implementation plan
4. Only use tools mentioned in the implementation
5. Ensure all tasks match the implementation sequence
6. Keep all agents and roles aligned with the implementation
"""
        )
        log_message("Generated solution", agent_name="CEO Agent")
        try:
            # Attempt to parse the response to ensure it's valid JSON
            if isinstance(result.data, str):
                try:
                    parsed_json = json.loads(result.data)
                except json.JSONDecodeError:
                    log_message(
                        "Attempting to extract JSON from response",
                        level="warning",
                        agent_name="CEO Agent"
                    )
                    json_str = result.data[result.data.find('{'):result.data.rfind('}')+1]
                    parsed_json = json.loads(json_str)
            else:
                parsed_json = result.data

            # Validate that the output matches the implementation
            if not self._validate_implementation_match(parsed_json, self.implementation):
                log_message(
                    "Generated output does not match implementation plan. Regenerating...",
                    level="warning",
                    agent_name="CEO Agent"
                )
                # Could add regeneration logic here

            formatted_result = json.dumps(parsed_json, indent=2)
            log_message(
                f"Final Solution:\n{formatted_result}",
                agent_name="CEO Agent"
            )
            return End(parsed_json)
        except Exception as e:
            error_msg = f"Error formatting result: {str(e)}\nOriginal response: {result.data}"
            log_message(
                error_msg,
                level="error",
                agent_name="CEO Agent"
            )
            error_response = {
                "error": str(e),
                "original_response": result.data,
                "status": "failed"
            }
            return End(error_response)

    def _validate_implementation_match(self, output: Dict, implementation: str) -> bool:
        """Validate that the output matches the implementation plan"""
        try:
            # Convert implementation to lowercase for case-insensitive matching
            impl_lower = implementation.lower()
            
            # Check if key components from implementation are present in output
            # 1. Check tools
            tools_mentioned = [tool["name"] for tool in output["tools"]]
            for tool in ["web_scraper", "api_caller", "code_interpreter"]:
                if tool in impl_lower and tool not in tools_mentioned:
                    return False
            
            # 2. Check tasks
            impl_tasks = [line.strip().lower() for line in impl_lower.split('\n') 
                         if 'task' in line.lower() and ':' in line]
            output_tasks = [task["name"].lower() for task in output["tasks"]]
            
            for impl_task in impl_tasks:
                if not any(impl_task in out_task for out_task in output_tasks):
                    return False
            
            # 3. Check process type
            if "sequence" in impl_lower and output["crew_details"]["process"] != "sequence":
                return False
            
            return True
        except:
            return False

# Agent System Implementation
class AIAgentSystem:
    def __init__(self):
        log_message("Initializing AI Agent System", agent_name="System")
        self.model = create_model()
        self.tools = self._load_tools()
        self.graph = Graph(nodes=[AnalysisNode, ArchitectureNode, ImplementationNode, FinalNode])
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

    def create_agents(self) -> List[Agent]:
        log_message("Creating agent team", agent_name="System")
        # CEO Agent
        ceo_agent = Agent(
            model=self.model,
            name="CEO Agent",
            system_prompt="""As the strategic overseer, I specialize in analyzing user requirements and creating customized AI solutions. 
            I ensure that every component of the solution is directly derived from and justified by the user's specific needs.

            My goal is to transform user requirements into a precisely tailored AI solution by:
            1. Ensuring every component directly addresses user needs
            2. Eliminating any unnecessary elements
            3. Validating that the solution exactly matches requirements""",
            tools=self.tools
        )
        log_message("Created CEO Agent", agent_name="System")

        # Regular agents
        agents = [
            Agent(
                model=self.model,
                name="Requirements Analyst",
                system_prompt="""I specialize in precise requirement analysis and specification. 
                I focus on extracting exactly what the user needs, nothing more and nothing less.

                My goal is to create exact, focused specifications by:
                1. Identifying core requirements
                2. Eliminating non-essential elements
                3. Ensuring every specification ties directly to user needs

                Available Tools List:
                - web_scraper - for web scraping
                - search_tool - for search results
                - website_search - for website searching
                - api_caller - for API integrations
                - code_interpreter - for code interpretation
                - file_reader - for reading files
                - file_writer - for writing files""",
                tools=self.tools
            ),
            Agent(
                model=self.model,
                name="Architecture Specialist",
                system_prompt="""I design minimal, focused architectures that precisely match requirements.
                I ensure every component has a direct purpose in fulfilling user needs.

                My goal is to create efficient, focused architectures by:
                1. Including only necessary components
                2. Ensuring direct requirement traceability
                3. Eliminating unnecessary complexity

                Available Tools List:
                - web_scraper - for web scraping
                - search_tool - for search results
                - website_search - for website searching
                - api_caller - for API integrations
                - code_interpreter - for code interpretation
                - file_reader - for reading files
                - file_writer - for writing files""",
                tools=self.tools
            ),
            Agent(
                model=self.model,
                name="Implementation Agent",
                system_prompt="""I implement solutions with precise alignment to requirements.
                I ensure every implementation detail serves a specific, necessary purpose.

                My goal is to create focused implementations by:
                1. Building only what's needed
                2. Ensuring direct requirement fulfillment
                3. Maintaining minimal complexity

                Available Tools List:
                - web_scraper - for web scraping
                - search_tool - for search results
                - website_search - for website searching
                - api_caller - for API integrations
                - code_interpreter - for code interpretation
                - file_reader - for reading files
                - file_writer - for writing files""",
                tools=self.tools
            ),
            Agent(
                model=self.model,
                name="Best AI Agent Developer",
                system_prompt="""I specialize in creating precisely tailored AI solutions.
                I ensure every feature and capability directly serves the user's specific needs.

                My goal is to develop optimal AI solutions by:
                1. Implementing only necessary features
                2. Ensuring direct alignment with requirements
                3. Validating solution effectiveness

                Available Tools List:
                - web_scraper - for web scraping
                - search_tool - for search results
                - website_search - for website searching
                - api_caller - for API integrations
                - code_interpreter - for code interpretation
                - file_reader - for reading files
                - file_writer - for writing files""",
                tools=self.tools
            )
        ]
        log_message(f"Created {len(agents)} specialized agents", agent_name="System")
        return agents, ceo_agent

    def process_request(self, user_requirements: str) -> Dict[str, Any]:
        log_message("Starting synchronous request processing", agent_name="System")
        # Create agents and tasks
        agents, ceo_agent = self.create_agents()
        self.tasks = self.load_tasks(agents, ceo_agent, user_requirements)
        
        # Log task creation
        for task in self.tasks:
            log_message(
                f"Created task: {task.name} for {task.agent_role}",
                agent_name="System"
            )
        
        # Process through graph
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            analysis_node = AnalysisNode(agents[0], user_requirements)
            result = loop.run_until_complete(self.process_request_async(analysis_node))
            return result
        finally:
            loop.close()
            log_message("Closed event loop", agent_name="System")

    async def process_request_async(self, initial_node: BaseNode) -> Dict[str, Any]:
        """Process the request through the graph system"""
        log_message("Starting asynchronous request processing", agent_name="System")
        result, _ = await self.graph.run(initial_node)
        log_message("Request processing completed", agent_name="System")
        return result

    def load_tasks(self, agents: List[Agent], ceo_agent: Agent, user_requirements: str) -> List[TaskConfig]:
        """Load tasks based on user requirements and available agents"""
        tasks = [
            TaskConfig(
                name="Requirements Analysis",
                description=f"""Perform a comprehensive analysis of the following user requirements:
{user_requirements}

Follow these steps in your analysis:

1. Core Requirements Analysis:
   - Identify explicit requirements
   - Identify implicit requirements
   - List any potential edge cases
   - Determine priority levels

2. Agent Requirements:
   - Determine minimum number of agents needed
   - Identify specific roles required
   - Define interaction patterns
   - Specify expertise level needed

3. Tool Selection:
   - Match each requirement with necessary tools
   - Justify each tool selection
   - Identify any custom tool needs
   Available Tools:
   - web_scraper - for web scraping
   - search_tool - for search results
   - website_search - for website searching
   - api_caller - for API integrations
   - code_interpreter - for code interpretation
   - file_reader - for reading files
   - file_writer - for writing files

4. Process Analysis:
   - Determine optimal process type
   - Identify critical paths
   - Define dependencies
   - Estimate resource requirements

Provide a structured JSON response with detailed analysis.""",
                expected_output="Structured JSON Analysis",
                agent_role="Requirements Analyst",
                async_execution=False,
                context_requirements={"async": [], "sync": []}
            ),
            TaskConfig(
                name="Architecture Design",
                description=f"""Based on the requirements analysis, design a technical architecture that addresses:
{user_requirements}

1. Component Design:
   - Core components needed
   - Component interactions
   - Data flow patterns
   - Integration points

2. Tool Integration:
   - Specific tool configurations
   - Custom tool requirements
   - Integration patterns
   Available Tools:
   - web_scraper - for web scraping
   - search_tool - for search results
   - website_search - for website searching
   - api_caller - for API integrations
   - code_interpreter - for code interpretation
   - file_reader - for reading files
   - file_writer - for writing files

3. Resource Planning:
   - Component resource needs
   - Scaling considerations
   - Performance requirements
   - Optimization strategies

Provide a detailed technical specification in JSON format.""",
                expected_output="Technical Architecture JSON",
                agent_role="Architecture Specialist",
                async_execution=False,
                context_requirements={"async": [], "sync": ["Requirements Analysis"]}
            ),
            TaskConfig(
                name="Implementation Planning",
                description=f"""Create a detailed implementation plan for:
{user_requirements}

1. Development Phases:
   - Phase breakdown
   - Milestone definitions
   - Dependency management
   - Resource allocation

2. Tool Configuration:
   - Setup requirements
   - Integration steps
   - Custom development needs
   Available Tools:
   - web_scraper - for web scraping
   - search_tool - for search results
   - website_search - for website searching
   - api_caller - for API integrations
   - code_interpreter - for code interpretation
   - file_reader - for reading files
   - file_writer - for writing files

3. Quality Assurance:
   - Testing strategies
   - Validation methods
   - Performance metrics
   - Success criteria

Provide a structured implementation plan in JSON format.""",
                expected_output="Implementation Plan JSON",
                agent_role="Implementation Agent",
                async_execution=False,
                context_requirements={"async": [], "sync": ["Architecture Design"]}
            ),
            TaskConfig(
                name="Solution Generation",
                description=f"""Generate a complete solution based on:
{user_requirements}

Create a comprehensive JSON solution that includes:

1. Crew Details:
   - Team structure
   - Process flow
   - Resource allocation
   - Management approach

2. Tools Configuration:
   - Tool selection
   - Integration details
   - Custom configurations
   Available Tools:
   - web_scraper - for web scraping
   - search_tool - for search results
   - website_search - for website searching
   - api_caller - for API integrations
   - code_interpreter - for code interpretation
   - file_reader - for reading files
   - file_writer - for writing files

3. Agent Configuration:
   - Role definitions
   - Responsibility mapping
   - Interaction patterns
   - Tool access

4. Task Definitions:
   - Detailed descriptions
   - Dependencies
   - Success criteria
   - Resource needs

Follow the exact JSON structure provided in the template.""",
                expected_output="Complete Solution JSON",
                agent_role="CEO Agent",
                async_execution=False,
                context_requirements={"async": [], "sync": ["Implementation Planning"]}
            )
        ]
        return tasks

# Streamlit UI
def main():
    # Initialize Streamlit state at the start of the app
    initialize_streamlit_state()
    
    st.title("AI Agent Creator")
    st.markdown("""
    ### Welcome to the AI Agent Creator
    Please describe the functionalities and features you need in your AI agent.
    Our team of agents will analyze your requirements and generate a final report in JSON format including:
    - Agent List: The names and roles of all agents created.
    - Task List: All tasks generated based on your requirements.
    - Tool List: All available tools used in the AI agent creation.
    """)

    # Create tabs for input and logs
    input_tab, logs_tab = st.tabs(["Create Agent", "System Logs"])

    with input_tab:
        user_prompt = st.text_area(
            "What kind of AI agent would you like to create?",
            placeholder="Example: I need an AI agent that can process and analyze data to generate insights.",
            help="Describe your requirements - what should the agent be able to do?"
        )

        if st.button("Create AI Agent", type="primary"):
            if not user_prompt:
                st.warning("Please describe your requirements.")
                return

            with st.spinner("Creating your AI agent..."):
                try:
                    agent_system = AIAgentSystem()
                    result = agent_system.process_request(user_prompt)
                    
                    st.success("AI Agent created successfully!")
                    
                    # Display the result in a more organized way
                    with st.expander("View Generated AI Agent Details", expanded=True):
                        # Display Crew Details
                        st.subheader("🚀 Crew Details")
                        st.json(result.get("crew_details", {}))
                        
                        # Display Tools
                        st.subheader("🛠️ Tools")
                        st.json(result.get("tools", []))
                        
                        # Display Agents
                        st.subheader("🤖 Agents")
                        st.json(result.get("agents", []))
                        
                        # Display Tasks
                        st.subheader("📋 Tasks")
                        st.json(result.get("tasks", []))
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback
                    log_message(traceback.format_exc(), level="error")

    with logs_tab:
        st.markdown("### 📊 System Logs")
        
        # Add log filtering options
        col1, col2 = st.columns([2, 2])
        with col1:
            selected_levels = st.multiselect(
                "Filter by Log Level",
                ["INFO", "WARNING", "ERROR"],
                default=["INFO", "WARNING", "ERROR"]
            )
        with col2:
            selected_agents = st.multiselect(
                "Filter by Agent",
                ["System", "CEO Agent", "Requirements Analyst", "Architecture Specialist", 
                 "Implementation Agent", "Best AI Agent Developer"],
                default=["System", "CEO Agent", "Requirements Analyst", "Architecture Specialist", 
                        "Implementation Agent", "Best AI Agent Developer"]
            )
        
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
                    
                    # Create styled log entry
                    with st.container():
                        # Style based on log level
                        level_color = {
                            'ERROR': 'red',
                            'WARNING': 'orange',
                            'INFO': 'blue'
                        }.get(log['level'], 'gray')
                        
                        # Create header with timestamp and level
                        header = f"<div style='color: {level_color}; padding: 5px 0;'>"
                        header += f"<span style='font-weight: bold;'>{log['timestamp']}</span> - "
                        header += f"<span style='background-color: {level_color}; color: white; padding: 2px 8px; border-radius: 4px;'>{log['level']}</span>"
                        
                        # Add agent name if present
                        if log['agent']:
                            header += f" - <span style='font-style: italic; color: gray;'>{log['agent']}</span>"
                        header += "</div>"
                        
                        st.markdown(header, unsafe_allow_html=True)
                        
                        # Display message in a box with appropriate styling
                        message_style = f"background-color: {'#ffebee' if log['level'] == 'ERROR' else '#fff3e0' if log['level'] == 'WARNING' else '#e3f2fd'}; padding: 10px; border-radius: 4px; margin: 5px 0;"
                        st.markdown(f"<div style='{message_style}'>{log['message']}</div>", unsafe_allow_html=True)
                        
                        # Add separator
                        st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)

if __name__ == '__main__':
    main() 

     
