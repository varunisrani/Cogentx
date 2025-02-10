import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *


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
    
    # Default to OpenAI
    return create_openai_llm("gpt-4o-mini", temperature)

def load_agents():
    """Load and create agent instances."""
    agents = [
        Agent(
            role="MarketDataCollector",
            backstory="Experience in data collection and analysis in market research.",
            goal="To gather accurate data on market trends and competitor positioning.",
            allow_delegation=true,
            verbose=false,
            tools=[SerperDevTool(), ScrapeWebsiteTool(), YahooFinanceNewsTool()],
            llm=create_llm("OpenAI: gpt-4o-mini", 0.5)
        ),
        Agent(
            role="DataInsightsGenerator",
            backstory="Background in data analytics and report generation.",
            goal="To analyze customer behavior patterns effectively.",
            allow_delegation=true,
            verbose=false,
            tools=[],
            llm=create_llm("OpenAI: gpt-4o-mini", 0.5)
        )
    ]
    return agents

def load_tasks(agents):
    """Load and create task instances."""
    tasks = [
        Task(
            description="Gather consumer feedback through structured surveys.",
            expected_output="Survey results and insights.",
            agent=next(agent for agent in agents if agent.role == "MarketDataCollector"),
            async_execution=true
        ),
        Task(
            description="Examine web traffic and user behavior.",
            expected_output="Web traffic analysis report.",
            agent=next(agent for agent in agents if agent.role == "MarketDataCollector"),
            async_execution=true
        ),
        Task(
            description="Visual representation of market data and trends.",
            expected_output="Visual data reports.",
            agent=next(agent for agent in agents if agent.role == "DataInsightsGenerator"),
            async_execution=false
        ),
        Task(
            description="Ensure timely execution of all tasks.",
            expected_output="Timeline and progress reports.",
            agent=next(agent for agent in agents if agent.role == "Project Manager"),
            async_execution=false
        )
    ]
    return tasks

def main():
    st.title("MarketResearchAgent")

    agents = load_agents()
    tasks = load_tasks(agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process="Process.sequential",
        verbose=false,
        memory=false,
        cache=true,
        max_rpm=1000,
        manager_llm=create_llm("OpenAI: gpt-4o-mini")
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
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
