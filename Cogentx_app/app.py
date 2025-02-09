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
    llm = create_llm()  # Now works without arguments, using default values
    
    # Create CEO Agent as the strategic overseer
    ceo_agent = Agent(
        role="CEO Agent",
        backstory="""As the strategic overseer, I analyze user requirements and coordinate the development process. 
        I ensure all agents work together effectively to create the requested AI solution.""",
        goal="""To transform user requirements into a functional AI agent by coordinating the team:
        1. Analyze and validate requirements
        2. Oversee the development process
        3. Ensure quality and coherence""",
        allow_delegation=True,
        verbose=True,
        llm=llm
    )

    # Regular agents list (excluding CEO Agent)
    agents = [
        Agent(
            role="Requirements Analyst",
            backstory="""I analyze and refine user requirements for AI agent creation. 
            I specialize in understanding user needs and converting them into technical specifications.""",
            goal="""To create detailed specifications for AI agent development:
            1. Extract key requirements
            2. Identify necessary capabilities
            3. Create technical specifications""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Architecture Specialist",
            backstory="""I design the AI agent's architecture and capabilities.
            I determine the best approaches and methods to implement the required functionality.""",
            goal="""To design optimal AI agent architecture:
            1. Design core components
            2. Define processing logic
            3. Plan data flows""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Implementation Agent",
            backstory="""I coordinate the implementation of AI agents.
            I translate architectural designs into concrete implementation plans.""",
            goal="""To implement AI agents according to specifications:
            1. Create implementation plan
            2. Define component interactions
            3. Specify testing approach""",
            allow_delegation=True,
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Best AI Agent Developer",
            backstory="""As the Best AI Agent Developer, I bring cutting-edge expertise and innovative solutions to the AI agent creation process, ensuring that the final product is top-notch and exceeds industry standards.""",
            goal="""To utilize advanced design and development techniques to create the best possible AI agent solution:
            1. Ideate and refine the architecture
            2. Innovate in functionality and performance
            3. Ensure seamless integration of all components""",
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
            description=f"""Analyze the user requirements and create specifications:
            Requirements: {user_requirements}
            
            Create a detailed analysis including:
            1. Core functionality needed
            2. Required capabilities
            3. Processing requirements""",
            expected_output="Detailed specifications in JSON format",
            agent=next(agent for agent in agents if agent.role == "Requirements Analyst")
        ),
        Task(
            description="""Design the AI agent architecture:
            1. Core components design
            2. Processing logic
            3. Data flow patterns""",
            expected_output="Architecture design in JSON format",
            agent=next(agent for agent in agents if agent.role == "Architecture Specialist")
        ),
        Task(
            description="""Create implementation plan:
            1. Component implementation details
            2. Integration steps
            3. Testing strategy""",
            expected_output="Implementation plan in JSON format",
            agent=next(agent for agent in agents if agent.role == "Implementation Agent")
        ),
        Task(
            description=f"""Review the complete solution, modify all agents and redesign the entire crew to incorporate optimal AI agent development, including the newly added 'Best AI Agent Developer'.
User Requirements: {user_requirements}

Based on the above, please produce a final report in JSON format with the following keys:
- \"Agent List\": List the names and roles of all agents created.
- \"Task List\": List all tasks generated based on your requirements.
- \"Tool List\": List all available tools used for AI agent creation (if any).

Ensure that only these keys are present in the final output.""",
            expected_output="Final AI Agent Creator Report in JSON format",
            agent=ceo_agent
        )
    ]
    return tasks

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
                agents, ceo_agent = load_agents()
                tasks = load_tasks(agents, ceo_agent, user_prompt)
                
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    manager_agent=ceo_agent
                )

                result = crew.kickoff()
                
                st.success("AI Agent created successfully!")
                
                st.json(result)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
