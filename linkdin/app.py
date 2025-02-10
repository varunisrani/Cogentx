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
            role="InputAnalysisAgent",
            backstory="Experienced in data collection and user interaction",
            goal="To gather and prepare data for post generation",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="NLPAgent",
            backstory="Specializes in natural language processing",
            goal="To produce engaging and relevant post drafts",
            allow_delegation=true,
            verbose=false,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: GPT-4", 0.7)
        ),
        Agent(
            role="SentimentAnalysisAgent",
            backstory="Expert in sentiment analysis",
            goal="Ensure alignment with user intent",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="HashtagGeneratorAgent",
            backstory="Knowledgeable in social media trends",
            goal="Enhance post visibility",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="ContentAnalysisAgent",
            backstory="Skilled in content optimization",
            goal="Refine the post for publishing",
            allow_delegation=true,
            verbose=false,
            tools=[CustomCodeInterpreterTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="DraftReviewAgent",
            backstory="Experienced in user feedback processes",
            goal="Iterate on user feedback to improve drafts",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="FinalOutputAgent",
            backstory="Expert in content management",
            goal="Ensure readiness of posts for LinkedIn",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        ),
        Agent(
            role="PerformanceMonitoringAgent",
            backstory="Analytical background in social media",
            goal="Optimize future content strategies",
            allow_delegation=true,
            verbose=false,
            tools=[CustomApiTool()],
            llm=create_llm("OpenAI: GPT-4", 0.5)
        )
    ]
    return agents

def load_tasks(agents):
    """Load and create task instances."""
    tasks = [
        Task(
            description="Collect and analyze user-provided data",
            expected_output="Processed user input for further use",
            agent=next(agent for agent in agents if agent.role == "InputAnalysisAgent"),
            async_execution=false
        ),
        Task(
            description="Generate coherent post drafts",
            expected_output="Draft post ready for review",
            agent=next(agent for agent in agents if agent.role == "NLPAgent"),
            async_execution=false
        ),
        Task(
            description="Assess the emotional tone of the content",
            expected_output="Sentiment analysis report",
            agent=next(agent for agent in agents if agent.role == "SentimentAnalysisAgent"),
            async_execution=false
        ),
        Task(
            description="Suggest relevant hashtags for the post",
            expected_output="List of hashtags",
            agent=next(agent for agent in agents if agent.role == "HashtagGeneratorAgent"),
            async_execution=false
        ),
        Task(
            description="Evaluate the generated post for clarity and engagement",
            expected_output="Content analysis report",
            agent=next(agent for agent in agents if agent.role == "ContentAnalysisAgent"),
            async_execution=false
        ),
        Task(
            description="Facilitate user review of draft posts",
            expected_output="User feedback on draft",
            agent=next(agent for agent in agents if agent.role == "DraftReviewAgent"),
            async_execution=true
        ),
        Task(
            description="Compile the approved draft and hashtags into a final post",
            expected_output="Final post ready for publication",
            agent=next(agent for agent in agents if agent.role == "FinalOutputAgent"),
            async_execution=false
        ),
        Task(
            description="Track post engagement metrics",
            expected_output="Performance report",
            agent=next(agent for agent in agents if agent.role == "PerformanceMonitoringAgent"),
            async_execution=true
        )
    ]
    return tasks

def main():
    st.title("LinkedInPostGenerator")

    agents = load_agents()
    tasks = load_tasks(agents)
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process="Process.sequential",
        verbose=false,
        memory=true,
        cache=true,
        max_rpm=1000,
        manager_llm=create_llm("OpenAI: GPT-4")
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
