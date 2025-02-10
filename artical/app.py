import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *
from tools.CustomApiTool import CustomApiTool
from tools.CustomCodeInterpreterTool import CustomCodeInterpreterTool
from tools.ScrapeWebsiteToolEnhanced import ScrapeWebsiteToolEnhanced
from tools.CSVSearchToolEnhanced import CSVSearchToolEnhanced

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
            role="Natural Language Generation Agent",
            backstory="Trained on diverse datasets, specializing in text generation.",
            goal="Produce high-quality drafts for articles.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: OpenAI gpt-4o-mini", 0.7)
        ),
        Agent(
            role="Content Structuring Agent",
            backstory="Utilizes linguistic algorithms for content mapping.",
            goal="Create a structured outline for articles.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: Python NLTK", 0.5)
        ),
        Agent(
            role="Topic Research Agent",
            backstory="Experienced in data scraping and keyword analysis.",
            goal="Inform articles with accurate and up-to-date information.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: Google Search API", 0.5)
        ),
        Agent(
            role="Customization Options Agent",
            backstory="Proficient in user interface design and user experience.",
            goal="Allow users to personalize their articles.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: React", 0.5)
        ),
        Agent(
            role="Editing and Proofreading Agent",
            backstory="Utilizes advanced grammar-checking algorithms.",
            goal="Deliver polished and professional articles.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: Grammarly API", 0.5)
        ),
        Agent(
            role="Feedback Loop Agent",
            backstory="Aims for continuous enhancement based on user input.",
            goal="Iteratively improve the article generation process.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: Google Forms", 0.5)
        ),
        Agent(
            role="Export and Formatting Options Agent",
            backstory="Expert in document conversion and formatting.",
            goal="Provide multiple output formats for user articles.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: Pandoc", 0.5)
        ),
        Agent(
            role="Version Control Agent",
            backstory="Utilizes Git for tracking changes and collaboration.",
            goal="Allow users to revert and compare article versions.",
            allow_delegation=true,
            verbose=true,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: Git", 0.5)
        )
    ]
    return agents

def load_tasks(agents):
    """Load and create task instances."""
    tasks = [
        Task(
            description="Generate an initial draft based on user input.",
            expected_output="Draft text for the article.",
            agent=next(agent for agent in agents if agent.role == "Natural Language Generation Agent"),
            async_execution=true
        ),
        Task(
            description="Organize the draft into a logical structure.",
            expected_output="Structured outline of the article.",
            agent=next(agent for agent in agents if agent.role == "Content Structuring Agent"),
            async_execution=true
        ),
        Task(
            description="Gather relevant information and statistics.",
            expected_output="Research data and references.",
            agent=next(agent for agent in agents if agent.role == "Topic Research Agent"),
            async_execution=true
        ),
        Task(
            description="Allow users to customize the article tone and style.",
            expected_output="User preferences for the article.",
            agent=next(agent for agent in agents if agent.role == "Customization Options Agent"),
            async_execution=true
        ),
        Task(
            description="Review the article for grammar and style.",
            expected_output="Edited and proofread article.",
            agent=next(agent for agent in agents if agent.role == "Editing and Proofreading Agent"),
            async_execution=true
        ),
        Task(
            description="Gather user feedback on the article.",
            expected_output="Feedback data.",
            agent=next(agent for agent in agents if agent.role == "Feedback Loop Agent"),
            async_execution=true
        ),
        Task(
            description="Convert the article into required formats.",
            expected_output="Formatted article in various formats.",
            agent=next(agent for agent in agents if agent.role == "Export and Formatting Options Agent"),
            async_execution=true
        ),
        Task(
            description="Track and manage versions of the article.",
            expected_output="Version history.",
            agent=next(agent for agent in agents if agent.role == "Version Control Agent"),
            async_execution=true
        )
    ]
    return tasks

def main():
    st.title("article_writer_agent")

    agents = load_agents()
    tasks = load_tasks(agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process="Process.sequential",
        verbose=true,
        memory=true,
        cache=true,
        max_rpm=1000,
        manager_llm=create_llm("OpenAI: OpenAI gpt-4o-mini")
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
