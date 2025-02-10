import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *





load_dotenv()

def create_lmstudio_llm(model, temperature):
    api_base = os.getenv('LMSTUDIO_API_BASE')
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_API_BASE"] = api_base
    if api_base:
        return ChatOpenAI(openai_api_key='lm-studio', openai_api_base=api_base, temperature=temperature)
    else:
        raise ValueError("LM Studio API base not set in .env file")

def create_openai_llm(model, temperature):
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
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Groq API key not set in .env file")

def create_anthropic_llm(model, temperature):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return ChatAnthropic(anthropic_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Anthropic API key not set in .env file")

def safe_pop_env_var(key):
    try:
        os.environ.pop(key)
    except KeyError:
        pass
        
LLM_CONFIG = {
    "OpenAI": {
        "create_llm": create_openai_llm
    },
    "Groq": {
        "create_llm": create_groq_llm
    },
    "LM Studio": {
        "create_llm": create_lmstudio_llm
    },
    "Anthropic": {
        "create_llm": create_anthropic_llm
    }
}

def create_llm(provider_and_model="OpenAI: gpt-4o-mini", temperature=0.7):
    """Create an LLM instance based on provider and model."""
    if isinstance(provider_and_model, str) and ":" in provider_and_model:
        provider, model = provider_and_model.split(": ")
        create_llm_func = LLM_CONFIG.get(provider, {}).get("create_llm")
        if create_llm_func:
            return create_llm_func(model, temperature)
    
    # Default to CrewAI's LLM with OpenAI
    return LLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=temperature
    )

def load_agents():
    llm = create_llm()
    
    # Create CEO Agent as the strategic overseer
    ceo_agent = Agent(
        role="CEO Agent",
        backstory="""As the strategic overseer, I specialize in analyzing user requirements and creating customized AI solutions. 
        I ensure that every component of the solution is directly derived from and justified by the user's specific needs.""",
        goal="""To transform user requirements into a precisely tailored AI solution by:
        1. Ensuring every component directly addresses user needs
        2. Eliminating any unnecessary elements
        3. Validating that the solution exactly matches requirements""",
        allow_delegation=True,
        verbose=True,
        llm=llm
    )

    # Regular agents list (excluding CEO Agent)
    agents = [
        Agent(
            role="Requirements Analyst",
            backstory="""I specialize in precise requirement analysis and specification. 
            I focus on extracting exactly what the user needs, nothing more and nothing less.""",
            goal="""To create exact, focused specifications by:
            1. Identifying core requirements
            2. Eliminating non-essential elements
            3. Ensuring every specification ties directly to user needs""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Architecture Specialist",
            backstory="""I design minimal, focused architectures that precisely match requirements.
            I ensure every component has a direct purpose in fulfilling user needs.""",
            goal="""To create efficient, focused architectures by:
            1. Including only necessary components
            2. Ensuring direct requirement traceability
            3. Eliminating unnecessary complexity""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Implementation Agent",
            backstory="""I implement solutions with precise alignment to requirements.
            I ensure every implementation detail serves a specific, necessary purpose.""",
            goal="""To create focused implementations by:
            1. Building only what's needed
            2. Ensuring direct requirement fulfillment
            3. Maintaining minimal complexity""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Best AI Agent Developer",
            backstory="""I specialize in creating precisely tailored AI solutions.
            I ensure every feature and capability directly serves the user's specific needs.""",
            goal="""To develop optimal AI solutions by:
            1. Implementing only necessary features
            2. Ensuring direct alignment with requirements
            3. Validating solution effectiveness""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        )
    ]
    return agents, ceo_agent

def load_tasks(agents, ceo_agent, user_requirements):
    """Load tasks based on user requirements and available agents"""
    tasks = [
        Task(
            description=f"""First, analyze these specific user requirements and identify EXACTLY what kind of agent is needed:
User Requirements: {user_requirements}

Create a focused analysis that identifies:
1. Core functionality needed
2. Minimum number of agents required
3. Essential tools needed (ONLY select from the available tools list below - if a specific tool isn't available, use CustomApiTool or CustomCodeInterpreterTool)
4. Optimal process type (sequence/horizontal)

Available Tools List:
- ScrapeWebsiteTool - for web scraping
- SerperDevTool - for search results
- WebsiteSearchTool - for website searching
- ScrapeWebsiteToolEnhanced - for advanced scraping
- SeleniumScrapingTool - for dynamic website scraping
- ScrapeElementFromWebsiteTool - for specific element scraping
- CustomApiTool - for custom API integrations
- codeInterpreterTool - for code interpretation
- CustomCodeInterpreterTool - for custom code execution
- FileReadTool - for reading files
- CustomFileWriteTool - for writing files
- DirectorySearchTool - for searching directories
- DirectoryReadTool - for reading directories
- YoutubeVideoSearchTool - for YouTube video search
- YoutubeChannelSearchTool - for YouTube channel search
- GithubSearchTool - for GitHub search
- CodeDocsSearchTool - for code documentation search
- YahooFinanceNewsTool - for financial news
- TXTSearchTool - for text file search
- CSVSearchTool - for CSV file search
- CSVSearchToolEnhanced - for advanced CSV search
- DOCXSearchTool - for Word document search
- EXASearchTool - for EXA file search
- JSONSearchTool - for JSON file search
- MDXSearchTool - for MDX file search
- PDFSearchTool - for PDF file search

Write your analysis in clear, simple text. Focus ONLY on what's explicitly mentioned in the user requirements.""",
            expected_output="Text Analysis of Requirements",
            agent=next(agent for agent in agents if agent.role == "Requirements Analyst")
        ),
        Task(
            description=f"""Based on the initial analysis and {user_requirements}, create a minimal technical specification.
Write in clear text format:
1. Minimum necessary components
2. Tool selection (ONLY from the available tools list - if needed functionality isn't available in standard tools, specify how to implement it using CustomApiTool or CustomCodeInterpreterTool)
3. Efficient process flow
4. Resource optimization

Remember: If you need functionality not covered by standard tools, use:
- CustomApiTool - for external API integrations
- CustomCodeInterpreterTool - for custom code execution
- Do NOT suggest external tools outside the available list""",
            expected_output="Technical Specification Text",
            agent=next(agent for agent in agents if agent.role == "Architecture Specialist")
        ),
        Task(
            description=f"""Create an optimized implementation plan that addresses ONLY what's needed for {user_requirements}.
Write in clear text format:
1. Minimal agent usage
2. Tool utilization (ONLY use tools from the available list - use CustomApiTool or CustomCodeInterpreterTool for custom needs)
3. Resource optimization
4. Clear task dependencies

Remember: Stay within the available tools ecosystem. If you need custom functionality:
1. Use CustomApiTool for API integrations
2. Use CustomCodeInterpreterTool for custom code execution
3. DO NOT suggest external tools""",
            expected_output="Implementation Plan Text",
            agent=next(agent for agent in agents if agent.role == "Implementation Agent")
        ),
        Task(
            description=f"""Based on all previous analyses, create a final solution in JSON format with these exact fields:

1. Crew creation details:
Name: [simple id name]
Process: [sequence/horizontal]
Agents: [list of required agents]
Tasks: [list of required tasks]
Manager LLM: [LLM model name]
Manager Agent: [managing agent name]
Verbose: [true/false]
Memory: [enabled/disabled]
Cache: [enabled/disabled]
Planning: [planning approach]
Max req/min: 1000

2. Tools:
CRITICAL: You must ONLY select tools from this list. If you need custom functionality:
- Use CustomApiTool for API integrations
- Use CustomCodeInterpreterTool for custom code execution
DO NOT suggest or use any external tools not in this list:

[Available Tools - Select ONLY from this list]
ScrapeWebsiteTool
SerperDevTool
WebsiteSearchTool
ScrapeWebsiteToolEnhanced
SeleniumScrapingTool
ScrapeElementFromWebsiteTool
CustomApiTool
codeInterpreterTool
CustomCodeInterpreterTool
FileReadTool
CustomFileWriteTool
DirectorySearchTool
DirectoryReadTool
YoutubeVideoSearchTool
YoutubeChannelSearchTool
GithubSearchTool
CodeDocsSearchTool
YahooFinanceNewsTool
TXTSearchTool
CSVSearchTool
CSVSearchToolEnhanced
DOCXSearchTool
EXASearchTool
JSONSearchTool
MDXSearchTool
PDFSearchTool

3. Agents:
[For each agent you listed in crew details, provide:]
Create agent: [agent name]
Role: [specific purpose]
Backstory: [relevant experience]
Goal: [clear objectives]
Allow delegation: [yes/no]
Verbose: [true/false]
Cache: [enabled/disabled]
LLM: [model name]
Temperature: [0.0-1.0]
Max iteration: [number]
Select tools: [list ONLY tools from the available tools list above]

4. Tasks:
[For each task you listed in crew details, provide:]
Create task: [task name]
Description: [what this task does]
Expected output: [what this task produces]
Agent: [which agent handles this]
Async execution: [yes/no]
Context from async tasks: [what it needs from async tasks]
Context from sync tasks: [what it needs from sync tasks]

CRITICAL RULES:
1. Follow this EXACT order: Crew details -> Tools -> Agents -> Tasks
2. In Crew details: Only list agents and tasks you will fully detail later
3. In Tools: ONLY select from the provided tools list above - use CustomApiTool or CustomCodeInterpreterTool for custom needs
4. In Agents: Provide complete details for EVERY agent listed in crew details
5. In Tasks: Provide complete details for EVERY task listed in crew details
6. Make everything connect logically:
   - Each agent should have tools they need for their tasks (ONLY from available tools list)
   - Each task should have an agent capable of handling it
   - Tools should match what agents need for their tasks
7. NO placeholders - every field must have meaningful content
8. NO extra fields - follow format exactly
9. NO external tools - use ONLY tools from the available list
10. For custom functionality: Use CustomApiTool or CustomCodeInterpreterTool

Return your answer in proper JSON format ONLY for this final output.""",
            expected_output="JSON Solution",
            agent=ceo_agent
        )
    ]
    return tasks

def delegate_work(task, context, coworker):
    """
    Delegate a task to a coworker
    
    Args:
        task (str): The task description
        context (str): The context for the task
        coworker (str): The role/name of the coworker
    
    Returns:
        str: The result of the delegation
    """
    try:
        # Ensure task and context are strings
        task_str = task if isinstance(task, str) else task.get('description', '')
        context_str = context if isinstance(context, str) else str(context)
        
        return {
            "task": task_str,
            "context": context_str, 
            "coworker": coworker
        }
    except Exception as e:
        return f"Error delegating work: {str(e)}"

def main():
    st.title("AI Agent Creator")
    st.markdown("""
    ### Welcome to the AI Agent Creator
    Please describe the functionalities and features you need in your AI agent.
    Our team of agents will analyze your requirements and generate a final report in JSON format including:
    - Agent List: The names and roles of all agents created.
    - Task List: All tasks generated based on your requirements.
    - Tool List: All available tools used in the AI agent creation.
    """)

    # Simple user input
    user_prompt = st.text_area(
        "What kind of AI agent would you like to create?",
        placeholder="Example: I need an AI agent that can process and analyze data to generate insights.",
        help="Describe your requirements - what should the agent be able to do?"
    )

    if st.button("Create AI Agent"):
        if not user_prompt:
            st.warning("Please describe your requirements.")
            return

        with st.spinner("Creating your AI agent..."):
            try:
                # 1. Load the agents including CEO agent
                agents, ceo_agent = load_agents()
                
                # 2. Create tasks based on user requirements
                tasks = load_tasks(agents, ceo_agent, user_prompt)
                
                # 3. Create and run the crew
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    manager_agent=ceo_agent
                )

                # 4. Get the JSON result
                result = crew.kickoff()
                
                # 5. Show success and JSON output
                st.success("AI Agent created successfully!")
                st.json(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
