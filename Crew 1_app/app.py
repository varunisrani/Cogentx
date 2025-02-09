
import streamlit as st
from crewai import Agent, Task, Crew, Process
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

def create_llm(provider_and_model, temperature=0.1):
    provider, model = provider_and_model.split(": ")
    create_llm_func = LLM_CONFIG.get(provider, {}).get("create_llm")
    if create_llm_func:
        return create_llm_func(model, temperature)
    else:
        raise ValueError(f"LLM provider {provider} is not recognized or not supported")

def load_agents():
    agents = [
        
Agent(
    role="Senior Researcher",
    backstory="Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world.",
    goal="Uncover groundbreaking technologies in AI",
    allow_delegation=False,
    verbose=False,
    tools=[],
    llm=create_llm("Groq: groq/llama3-70b-8192", 0.1)
)
            ,
        
Agent(
    role="Senior Researcher",
    backstory="Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world.",
    goal="Uncover groundbreaking technologies in AI",
    allow_delegation=False,
    verbose=False,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o", 0.1)
)
            
    ]
    return agents

def load_tasks(agents):
    tasks = [
        
Task(
    description="Identify the next big trend in AI. Focus on identifying pros and cons and the overall narrative.",
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    agent=next(agent for agent in agents if agent.role == "Senior Researcher"),
    async_execution=False
)
            
    ]
    return tasks

def main():
    st.title("Crew 1")

    agents = load_agents()
    tasks = load_tasks(agents)
    crew = Crew(
        agents=agents, 
        tasks=tasks, 
        process="hierarchical", 
        verbose=True, 
        memory=False, 
        cache=True, 
        max_rpm=1000,
        
    )

    

    placeholders = {
        
    }
        with st.spinner("Running crew..."):
            try:
                result = crew.kickoff(inputs=placeholders)
                with st.expander("Final output", expanded=True):
                    if hasattr(result, 'raw'):
                        st.write(result.raw)                
                with st.expander("Full output", expanded=False):
                    st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
