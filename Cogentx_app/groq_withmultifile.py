from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
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
from openai import AsyncOpenAI
from supabase import create_client
import hashlib
from thefuzz import fuzz
import httpx
from bs4 import BeautifulSoup
import fnmatch
from urllib.parse import urljoin
import tempfile
import subprocess

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

def log_message(message: str, level: str = "info", agent_name: str = None, to_agent: str = None):
    """Log messages to CLI and file with enhanced communication tracking"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if agent_name and to_agent:
        formatted_message = f"[{agent_name} → {to_agent}] {message}"
    elif agent_name:
        formatted_message = f"[{agent_name}] {message}"
    else:
        formatted_message = f"[System] {message}"
    
    log_entry = f"{timestamp} | {formatted_message} | {level.upper()}"
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Error writing to log file: {str(e)}\nAttempted to log: {log_entry}")
    
    if level.lower() == "error":
        logger.error(formatted_message)
    elif level.lower() == "warning":
        logger.warning(formatted_message)
    else:
        logger.info(formatted_message)

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

# Load environment variables
load_dotenv()

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
def create_model(provider_and_model: str = "OpenAI:gpt-4o-mini", temperature: float = 0.7) -> Any:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment")
    
    # Initialize OpenAIModel without temperature
    model = OpenAIModel(
        api_key=api_key,
        model_name="gpt-4o-mini"
    )
    
    # Set temperature through the model's configuration if needed
    if hasattr(model, 'set_temperature'):
        model.set_temperature(temperature)
    elif hasattr(model, 'config'):
        model.config['temperature'] = temperature
    
    return model

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

# Initialize OpenAI client for embeddings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    log_message("OpenAI API key not found in environment", level="error")
    raise ValueError("OpenAI API key missing")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize Supabase client for vector search
SUPABASE_URL = "https://rzaukiglowabowqevpem.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ6YXVraWdsb3dhYm93cWV2cGVtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTgxODk3NDcsImV4cCI6MjAzMzc2NTc0N30.wSQnUlCio1DpXHj0xa5_6W6KjyUzXv4kKWyhpziUx_s"

if not SUPABASE_URL or not SUPABASE_KEY:
    log_message("Supabase configuration missing", level="error")
    raise ValueError("Supabase configuration missing")

try:
    log_message(f"Attempting to initialize Supabase client with URL: {SUPABASE_URL}", level="info")
    log_message(f"Using Supabase key starting with: {SUPABASE_KEY[:10]}...", level="debug")
    
    # Create the client without testing the connection
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    log_message("Successfully initialized Supabase client", level="info")
except Exception as e:
    error_msg = f"Error initializing Supabase client: {str(e)}"
    log_message(error_msg, level="error")
    log_message("Please check your Supabase configuration", level="error")
    raise ValueError(error_msg)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        log_message(f"Error getting embedding: {e}", level="error")
        return [0] * 1536  # zero vector on error

async def retrieve_relevant_documentation(query: str) -> str:
    """Retrieve relevant documentation chunks based on the query using Supabase RPC. If no relevant docs are found, fallback to fetching content from an available documentation page."""
    try:
        query_embedding = await get_embedding(query)
        result = supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        if result.data and len(result.data) > 0:
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"# {doc['title']}\n\n{doc['content']}"
                formatted_chunks.append(chunk_text)
            return "\n\n---\n\n".join(formatted_chunks)
        else:
            # Fallback: list available documentation pages and fetch content from the first one
            urls = await list_documentation_pages()
            if urls and len(urls) > 0:
                content = await get_page_content(urls[0])
                if content:
                    return content
            return "No relevant documentation available."
    except Exception as e:
        log_message(f"Error retrieving documentation: {e}", level="error")
        return f"Error retrieving documentation: {e}"

async def list_documentation_pages() -> List[str]:
    """Retrieve a list of all documentation page URLs from Supabase."""
    try:
        result = supabase.from_('site_pages').select('url').eq('metadata->>source', 'pydantic_ai_docs').execute()
        if not result.data:
            return []
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
    except Exception as e:
        log_message(f"Error retrieving documentation pages: {e}", level="error")
        return []

async def get_page_content(url: str) -> str:
    """Retrieve full content of a documentation page by URL."""
    try:
        result = supabase.from_('site_pages').select('title, content, chunk_number').eq('url', url).eq('metadata->>source', 'pydantic_ai_docs').order('chunk_number').execute()
        if not result.data:
            return f"No content found for URL: {url}"
        page_title = result.data[0]['title'].split(' - ')[0]
        contents = [f"# {page_title}\n"]
        for chunk in result.data:
            contents.append(chunk['content'])
        return "\n\n".join(contents)
    except Exception as e:
        log_message(f"Error retrieving page content: {e}", level="error")
        return f"Error retrieving page content: {e}"

@dataclass
class FinalNode(BaseNode):
    agent: Agent
    implementation: str

    async def run(self, ctx: GraphRunContext) -> End[Dict[str, Any]]:
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}

        # Retrieve relevant documentation
        relevant_docs = await retrieve_relevant_documentation(self.implementation)

        # Create the multi-file generation prompt
        final_prompt = (
            "You are a CrewAI framework expert. Based on the following implementation plan:\n"
            f"{json.dumps(self.implementation, indent=2)}\n\n"
            "and using the following documentation references:\n"
            f"{relevant_docs}\n\n"
            "Generate a complete CrewAI implementation based on the provided implementation plan and documentation. The output should follow this structure:\n\n"
            "1. app.py - Main CrewAI implementation that must include:\n"
            "   - Required imports (streamlit, crewai, langchain_groq, etc.)\n"
            "   - Proper logging configuration\n" 
            "   - LLM creation function using Groq\n"
            "   - Agent and task creation based on the implementation plan\n"
            "   - Main execution flow with Streamlit UI\n"
            "   - Error handling and logging\n\n"
            "2. main.py - Entry point file\n"
            "3. .env - Environment variables file\n"
            "4. requirements.txt - All required dependencies\n\n"
            "Use the following reference structure for app.py but modify it according to the implementation plan:\n"
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
            "            role='Role Name',  # Replace with actual roles based on requirements\n"
            "            goal='Specific goal',\n"
            "            backstory='Detailed backstory',\n"
            "            verbose=True,\n"
            "            allow_delegation=True,\n"
            "            llm=create_llm()\n"
            "        ),\n"
            "        # Add more agents based on the implementation requirements\n"
            "    ]\n"
            "    return agents\n\n"
            "def load_tasks(agents):\n"
            "    logger.info('Creating tasks with specific assignments to agents')\n"
            "    tasks = [\n"
            "        Task(\n"
            "            description='Task description',  # Replace with actual tasks\n"
            "            expected_output='Expected output format',\n"
            "            agent=next(agent for agent in agents if agent.role == 'Role Name'),\n"
            "            async_execution=False\n"
            "        ),\n"
            "        # Add more tasks based on the implementation requirements\n"
            "    ]\n"
            "    return tasks\n\n"
            "def main():\n"
            "    st.title('CrewAI System')\n"
            "    try:\n"
            "        logger.info('Initializing CrewAI system')\n"
            "        agents = load_agents()\n"
            "        tasks = load_tasks(agents)\n"
            "        crew = Crew(\n"
            "            agents=agents,\n"
            "            tasks=tasks,\n"
            "            process='sequential',\n"
            "            verbose=True,\n"
            "            memory=False,\n"
            "            cache=True,\n"
            "            max_rpm=1000\n"
            "        )\n"
            "        with st.spinner('Running crew...'):\n"
            "            logger.info('Starting crew execution')\n"
            "            result = crew.kickoff()\n"
            "            with st.expander('Final output', expanded=True):\n"
            "                if hasattr(result, 'raw'):\n"
            "                    st.write(result.raw)\n"
            "                else:\n"
            "                    st.write(result)\n"
            "            logger.info('Crew execution completed successfully')\n"
            "    except Exception as e:\n"
            "        error_msg = f'An error occurred: {str(e)}'\n"
            "        logger.error(error_msg)\n"
            "        st.error(error_msg)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
            "```\n\n"
            "Return the complete implementation as a JSON object with these exact keys: 'app.py', 'main.py', '.env', 'requirements.txt'.\n"
            "Each file's content should be based on the implementation plan while maintaining proper CrewAI framework patterns.\n"
        )

        # Log the prompt being sent to the agent
        log_message(
            f"Sending multi-file generation prompt to Code Generator:\n{final_prompt}",
            agent_name="Code Generator"
        )

        try:
            result = await self.agent.run(final_prompt)

            # Parse the JSON output
            try:
                if isinstance(result.data, str):
                    # Find the first { and last } to extract JSON
                    start = result.data.find('{')
                    end = result.data.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = result.data[start:end]
                        generated_files = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found in response")
                else:
                    generated_files = result.data

                # Validate required files
                required_files = {"app.py", "main.py", ".env", "requirements.txt"}
                if not all(file in generated_files for file in required_files):
                    missing = required_files - set(generated_files.keys())
                    raise ValueError(f"Missing required files: {missing}")

                # Store Code Generator's output
                if "code_generator_output" not in ctx.state["outputs"]:
                    ctx.state["outputs"]["code_generator_output"] = []

                ctx.state["outputs"]["code_generator_output"].append({
                    "timestamp": datetime.now().isoformat(),
                    "output": generated_files
                })

                # Log successful generation
                log_message(
                    "Code generation completed successfully. Generated files:\n" +
                    "\n".join(generated_files.keys()),
                    agent_name="Code Generator"
                )

                return QualityControlNode(
                    self.agent,
                    generated_files,
                    self.implementation,
                    ctx.state["outputs"]
                )

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing JSON output from Code Generator: {str(e)}"
                log_message(error_msg, level="error", agent_name="Code Generator")
                raise
            except Exception as e:
                error_msg = f"Error processing Code Generator output: {str(e)}"
                log_message(error_msg, level="error", agent_name="Code Generator")
                raise

        except Exception as e:
            error_msg = f"Error during code generation: {str(e)}"
            log_message(error_msg, level="error", agent_name="Code Generator")
            raise

@dataclass
class QualityControlNode(BaseNode):
    agent: Agent
    generated_files: Dict[str, str]
    implementation_details: Any
    agent_outputs: Dict[str, Any]

    async def run(self, ctx: GraphRunContext) -> End[Dict[str, Any]]:
        # Ensure state exists
        if not hasattr(ctx, 'state') or ctx.state is None:
            ctx.state = {"outputs": {}}
        elif "outputs" not in ctx.state:
            ctx.state["outputs"] = {}

        log_message(
            "Starting CEO quality control review of generated files",
            agent_name="CEO Agent"
        )

        # Format implementation details
        if isinstance(self.implementation_details, dict):
            impl_details = self.implementation_details
        else:
            impl_details = {"raw": str(self.implementation_details)}

        # Create review prompt for multi-file project
        review_prompt = f"""As the CEO, perform a comprehensive review of the generated multi-file AI project:

1. Generated Files Overview:
{json.dumps({k: f"<{len(v)} characters>" for k, v in self.generated_files.items()}, indent=2)}

2. Implementation Details:
{json.dumps(impl_details, indent=2)}

Evaluate the following aspects:
1. Requirements Alignment:
   - Do all files work together to meet requirements?
   - Is the project structure logical and complete?
   - Are environment variables properly configured?

2. Code Quality:
   - Is the code well-structured across files?
   - Are dependencies properly listed?
   - Is error handling comprehensive?
   - Is logging implemented effectively?

3. Documentation:
   - Are files properly documented?
   - Are configuration requirements clear?
   - Is the project setup process documented?

4. Technical Implementation:
   - Is the CrewAI framework used correctly?
   - Are the file interactions clear and efficient?
   - Is the project easily deployable?

Provide your evaluation in this JSON format:
{{
    "meets_requirements": true/false,
    "code_quality": true/false,
    "documentation": true/false,
    "technical_implementation": true/false,
    "issues": ["list of issues if any"],
    "recommendations": ["list of recommendations if any"],
    "needs_iteration": true/false,
    "final_approval": true/false
}}

If any aspect fails, provide specific details about what needs improvement."""

        try:
            review_result = await self.agent.run(review_prompt)
            
            try:
                # Parse the evaluation JSON
                if isinstance(review_result.data, str):
                    # Extract JSON if it's within a code block
                    if "```json" in review_result.data:
                        json_str = review_result.data.split("```json")[1].split("```")[0].strip()
                    elif "```" in review_result.data:
                        json_str = review_result.data.split("```")[1].strip()
                    else:
                        start = review_result.data.find("{")
                        end = review_result.data.rfind("}") + 1
                        if start >= 0 and end > start:
                            json_str = review_result.data[start:end]
                        else:
                            raise ValueError("No valid JSON found in response")
                    evaluation = json.loads(json_str)
                else:
                    evaluation = review_result.data

                # Store CEO's review output
                if "ceo_review" not in ctx.state["outputs"]:
                    ctx.state["outputs"]["ceo_review"] = []

                ctx.state["outputs"]["ceo_review"].append({
                    "timestamp": datetime.now().isoformat(),
                    "output": evaluation
                })

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
                    return ImplementationNode(self.agent, "Updated requirements", self.implementation_details)

                if evaluation.get("final_approval", False):
                    log_message(
                        "CEO approved final output. Saving generated files.",
                        agent_name="CEO Agent"
                    )
                    
                    # Save the generated files
                    output_dir = "generated_project"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    for filename, content in self.generated_files.items():
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        log_message(f"Saved {filename} to {filepath}", agent_name="System")

                    return End({
                        "files": self.generated_files,
                        "evaluation": evaluation,
                        "agent_outputs": self.agent_outputs,
                        "output_directory": output_dir
                    })
                else:
                    log_message(
                        "CEO rejected output. Starting new iteration.",
                        level="warning",
                        agent_name="CEO Agent"
                    )
                    return ImplementationNode(self.agent, "Rejected implementation", self.implementation_details)

            except json.JSONDecodeError as e:
                error_msg = f"Error parsing CEO review results: {str(e)}"
                log_message(error_msg, level="error", agent_name="CEO Agent")
                raise
            except Exception as e:
                error_msg = f"Error processing CEO review: {str(e)}"
                log_message(error_msg, level="error", agent_name="CEO Agent")
                raise

        except Exception as e:
            error_msg = f"Error during CEO review: {str(e)}"
            log_message(error_msg, level="error", agent_name="CEO Agent")
            raise

# Custom documentation manager implementation
@dataclass
class DocumentationManager:
    supabase_client: Any
    openai_client: AsyncOpenAI
    cache: Dict[str, Any] = None

    def __init__(self, supabase_client: Any, openai_client: AsyncOpenAI):
        self.supabase_client = supabase_client
        self.openai_client = openai_client
        self.cache = {}

    async def retrieve_documentation(self, query: str) -> str:
        """Retrieve relevant documentation chunks based on the query."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(query.encode()).hexdigest()
            
            # Check cache first
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get embedding for the query
            embedding_response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            embedding = embedding_response.data[0].embedding
            
            # Search in Supabase
            result = self.supabase_client.rpc(
                'match_site_pages',
                {
                    'query_embedding': embedding,
                    'match_count': 5,
                    'filter': {'source': 'pydantic_ai_docs'}
                }
            ).execute()
            
            if result.data and len(result.data) > 0:
                # Format the results
                formatted_chunks = []
                for doc in result.data:
                    chunk_text = f"# {doc['title']}\n\n{doc['content']}"
                    formatted_chunks.append(chunk_text)
                
                docs = "\n\n---\n\n".join(formatted_chunks)
                
                # Cache the result
                self.cache[cache_key] = docs
                return docs
            
            return "No relevant documentation found."
            
        except Exception as e:
            return f"Error retrieving documentation: {str(e)}"

# Agent System Implementation
class AIAgentSystem:
    def __init__(self):
        log_message("Initializing AI Agent System", agent_name="System")
        self.model = create_model()
        self.doc_manager = DocumentationManager(supabase, openai_client)
        self.tools = self._load_tools()
        self.graph = Graph(nodes=[AnalysisNode, ArchitectureNode, ImplementationNode, FinalNode, QualityControlNode])
        self.tasks = []
        self.documentation_cache = {}
        log_message("AI Agent System initialized successfully", agent_name="System")
        
    def _load_tools(self) -> List[Tool]:
        """Load all available tools with proper documentation."""
        log_message("Loading available tools", agent_name="System")
        
        # Define tool categories and their tools
        tool_categories = {
            "Web and Search": [
                ("browserbase_loader", self._browserbase_loader, "Load and process web content using Browserbase"),
                ("website_search", self._website_search, "Search within a website's content"),
                ("scrape_website", self._scrape_website, "Scrape content from a website"),
                ("selenium_scraper", self._selenium_scraper, "Scrape dynamic content using Selenium"),
                ("spider_scraper", self._spider_scraper, "Crawl and scrape websites recursively"),
                ("serper_search", self._serper_search, "Perform web search using Serper API")
            ],
            "File and Document": [
                ("file_read", self._file_read, "Read file contents"),
                ("file_write", self._file_write, "Write content to a file"),
                ("directory_read", self._directory_read, "Read directory contents"),
                ("directory_search", self._directory_search, "Search for files in directory")
            ],
            "Database": [
                ("mysql_search", self._mysql_search, "Search MySQL database"),
                ("pg_search", self._pg_search, "Search PostgreSQL database"),
                ("nl2sql", self._nl2sql, "Convert natural language to SQL")
            ]
        }
        
        # Create tools with proper documentation
        available_tools = []
        for category, tools in tool_categories.items():
            for name, func, description in tools:
                if hasattr(self, func.__name__):  # Only add tools that have been implemented
                    tool = Tool(
                        name=name,
                        function=func,
                        description=f"[{category}] {description}"
                    )
                    available_tools.append(tool)
        
        log_message(f"Loaded {len(available_tools)} tools across {len(tool_categories)} categories", agent_name="System")
        return available_tools

    async def get_tool_documentation(self, tool_name: str) -> str:
        """Fetch documentation for a specific tool."""
        query = f"tool documentation for {tool_name}"
        docs = await self.doc_manager.retrieve_documentation(query)
        return docs

    async def create_agents(self) -> (List[Agent], Agent):
        """Create AI agents with proper documentation and configuration."""
        log_message("CEO initiating team creation", agent_name="CEO Agent")
        
        try:
            # Create models for each agent type
            ceo_model = OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
            architecture_model = OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
            implementation_model = OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
            code_generator_model = OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
            
            # Get documentation for each role
            ceo_docs = await self.doc_manager.retrieve_documentation("CEO agent role and responsibilities")
            arch_docs = await self.doc_manager.retrieve_documentation("architecture specialist role and responsibilities")
            impl_docs = await self.doc_manager.retrieve_documentation("implementation agent role and responsibilities")
            code_docs = await self.doc_manager.retrieve_documentation("code generator role and responsibilities")
            
            # Create agents with documentation-enhanced prompts
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
                    "5. Maintain focus on web functionality and user experience\n\n"
                    f"Additional role context:\n{ceo_docs}\n\n"
                    "Lead your team to create production-ready solutions that exceed expectations."
                ),
                tools=self.tools
            )

            # Create specialized agents with documentation-enhanced prompts
            agents = [
                Agent(
                    model=architecture_model,
                    name="Architecture Specialist",
                    system_prompt=(
                        "Design robust system architectures focusing on:\n"
                        "1. Web-centric component organization\n"
                        "2. Scalable agent interactions\n"
                        "3. Efficient tool integration patterns\n"
                        "4. Performance optimization for web deployment\n\n"
                        f"Additional role context:\n{arch_docs}"
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
                        "4. Web-optimized performance\n\n"
                        f"Additional role context:\n{impl_docs}"
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
                        "4. Comprehensive documentation\n\n"
                        f"Additional role context:\n{code_docs}"
                    ),
                    tools=self.tools
                )
            ]
            
            log_message(f"CEO completed team assembly with {len(agents)} specialized agents", agent_name="CEO Agent")
            return agents, ceo_agent
            
        except Exception as e:
            error_msg = f"Error during agent creation: {str(e)}"
            log_message(error_msg, level="error", agent_name="CEO Agent")
            
            # Create agents with fallback model
            fallback_model = OpenAIModel(model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
            
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Create agents asynchronously
            agents, ceo_agent = loop.run_until_complete(self.create_agents())
            
            # Then proceed with CEO's analysis
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

    async def _browserbase_loader(self, url: str) -> Dict[str, Any]:
        """Load and process web content using Browserbase."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return {"content": response.text, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _website_search(self, url: str, query: str) -> Dict[str, Any]:
        """Search within a website's content."""
        try:
            content = await self._browserbase_loader(url)
            if content["status"] == "error":
                return content
            
            # Use fuzzy matching to find relevant content
            matches = []
            for line in content["content"].split("\n"):
                score = fuzz.partial_ratio(query.lower(), line.lower())
                if score > 80:  # Threshold for relevance
                    matches.append({"text": line, "score": score})
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _scrape_website(self, url: str, selectors: List[str] = None) -> Dict[str, Any]:
        """Scrape content from a website using CSS selectors."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = {}
                if selectors:
                    for selector in selectors:
                        elements = soup.select(selector)
                        results[selector] = [el.text.strip() for el in elements]
                else:
                    # Default scraping behavior
                    results["title"] = soup.title.text if soup.title else ""
                    results["headings"] = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3'])]
                    results["paragraphs"] = [p.text.strip() for p in soup.find_all('p')]
                
                return {"data": results, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _selenium_scraper(self, url: str, wait_time: int = 5) -> Dict[str, Any]:
        """Scrape dynamic content using Selenium."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for dynamic content to load
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located(('tag name', 'body'))
            )
            
            page_source = driver.page_source
            driver.quit()
            
            return {"content": page_source, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _spider_scraper(self, start_url: str, max_pages: int = 10) -> Dict[str, Any]:
        """Crawl and scrape websites recursively."""
        visited = set()
        results = []
        
        async def crawl(url: int, depth: int = 0):
            if depth >= max_pages or url in visited:
                return
            
            visited.add(url)
            try:
                content = await self._browserbase_loader(url)
                if content["status"] == "success":
                    soup = BeautifulSoup(content["content"], 'html.parser')
                    results.append({
                        "url": url,
                        "title": soup.title.text if soup.title else "",
                        "content": content["content"]
                    })
                    
                    # Find and follow links
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if next_url.startswith(start_url):
                            await crawl(next_url, depth + 1)
            except Exception:
                pass
        
        try:
            await crawl(start_url)
            return {"pages": results, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _serper_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Perform web search using Serper API."""
        try:
            api_key = os.getenv('SERPER_API_KEY')
            if not api_key:
                return {"error": "Serper API key not found", "status": "error"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://google.serper.dev/search',
                    headers={'X-API-KEY': api_key},
                    json={'q': query, 'num': num_results}
                )
                
                return {"results": response.json(), "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    # File and Document Tools
    async def _file_read(self, path: str) -> Dict[str, Any]:
        """Read file contents."""
        try:
            with open(path, 'r') as f:
                content = f.read()
            return {"content": content, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _file_write(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            with open(path, 'w') as f:
                f.write(content)
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _directory_read(self, path: str) -> Dict[str, Any]:
        """Read directory contents."""
        try:
            contents = os.listdir(path)
            return {"contents": contents, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _directory_search(self, path: str, pattern: str) -> Dict[str, Any]:
        """Search for files in directory."""
        try:
            matches = []
            for root, _, files in os.walk(path):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        matches.append(os.path.join(root, file))
            return {"matches": matches, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    # RAG Search Tools
    async def _code_docs_search(self, query: str) -> Dict[str, Any]:
        """Search through code documentation."""
        try:
            docs = await self.doc_manager.retrieve_documentation(query)
            return {"results": docs, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _txt_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within text files."""
        try:
            content = await self._file_read(file_path)
            if content["status"] == "error":
                return content
            
            matches = []
            for line in content["content"].split("\n"):
                score = fuzz.partial_ratio(query.lower(), line.lower())
                if score > 80:
                    matches.append({"text": line, "score": score})
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _csv_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within CSV files."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # Search across all columns
            matches = []
            for col in df.columns:
                mask = df[col].astype(str).str.contains(query, case=False, na=False)
                if mask.any():
                    matches.extend(df[mask].to_dict('records'))
            
            return {"matches": matches, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _docx_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within DOCX files."""
        try:
            from docx import Document
            doc = Document(file_path)
            
            matches = []
            for para in doc.paragraphs:
                score = fuzz.partial_ratio(query.lower(), para.text.lower())
                if score > 80:
                    matches.append({"text": para.text, "score": score})
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _json_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within JSON files."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            def search_json(obj, matches):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, (dict, list)):
                            search_json(v, matches)
                        else:
                            score = fuzz.partial_ratio(query.lower(), str(v).lower())
                            if score > 80:
                                matches.append({"path": k, "value": v, "score": score})
                elif isinstance(obj, list):
                    for item in obj:
                        search_json(item, matches)
            
            matches = []
            search_json(data, matches)
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _mdx_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within MDX files."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove MDX frontmatter
            if content.startswith('---'):
                content = content.split('---', 2)[-1]
            
            matches = []
            for line in content.split("\n"):
                score = fuzz.partial_ratio(query.lower(), line.lower())
                if score > 80:
                    matches.append({"text": line, "score": score})
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _pdf_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within PDF files."""
        try:
            import PyPDF2
            
            matches = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    text = reader.pages[page_num].extract_text()
                    for line in text.split("\n"):
                        score = fuzz.partial_ratio(query.lower(), line.lower())
                        if score > 80:
                            matches.append({
                                "page": page_num + 1,
                                "text": line,
                                "score": score
                            })
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _xml_search(self, file_path: str, query: str) -> Dict[str, Any]:
        """Search within XML files."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            matches = []
            
            def search_element(element, path=""):
                current_path = f"{path}/{element.tag}"
                if element.text and element.text.strip():
                    score = fuzz.partial_ratio(query.lower(), element.text.lower())
                    if score > 80:
                        matches.append({
                            "path": current_path,
                            "text": element.text.strip(),
                            "score": score
                        })
                for child in element:
                    search_element(child, current_path)
            
            search_element(root)
            
            return {
                "matches": sorted(matches, key=lambda x: x["score"], reverse=True),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _mysql_search(self, query: str, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search MySQL database."""
        try:
            import mysql.connector
            
            conn = mysql.connector.connect(**connection_params)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _pg_search(self, query: str, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search PostgreSQL database."""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(**connection_params)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {"results": results, "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def _nl2sql(self, query: str, database_type: str = "postgresql") -> Dict[str, Any]:
        """Convert natural language to SQL query."""
        try:
            # Create a prompt for SQL conversion
            prompt = f"Convert this natural language query to {database_type} SQL:\n{query}"
            
            # Use the model to generate SQL
            response = await self.agent.run(prompt)
            
            # Extract SQL from response
            sql_query = response.data.strip()
            
            return {
                "query": sql_query,
                "database_type": database_type,
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}

    def get_latest_logs(self) -> List[Dict[str, Any]]:
        """Get the latest logs from the log file."""
        logs = []
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        parts = line.strip().split(' | ')
                        if len(parts) >= 3:
                            timestamp, message, level = parts
                            agent_name = None
                            to_agent = None
                            
                            # Extract agent information from message
                            if '[' in message and ']' in message:
                                agent_info = message[message.find('[')+1:message.find(']')]
                                if '→' in agent_info:
                                    agent_name, to_agent = map(str.strip, agent_info.split('→'))
                                else:
                                    agent_name = agent_info.strip()
                                message = message[message.find(']')+1:].strip()
                            
                            logs.append({
                                'timestamp': timestamp,
                                'message': message,
                                'level': level,
                                'agent': agent_name,
                                'to_agent': to_agent
                            })
                    except Exception:
                        continue
        return logs
