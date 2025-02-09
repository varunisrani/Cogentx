
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *
from tools.CustomApiTool import CustomApiTool
from tools.CustomFileWriteTool import CustomFileWriteTool
from tools.CustomCodeInterpreterTool import CustomCodeInterpreterTool
from tools.ScrapeWebsiteToolEnhanced import ScrapeWebsiteToolEnhanced
from tools.CSVSearchToolEnhanced import CSVSearchToolEnhanced
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
    role=" Market Data Collector",
    backstory=": A seasoned data collector with extensive experience in web scraping and file processing, capable of gathering raw information from diverse online sources.",
    goal="To fetch market data, financial news, and reports from websites, PDFs, CSVs, DOCXs, JSON files, and plain text.",
    allow_delegation=True,
    verbose=True,
    tools=[WebsiteSearchTool(), ScrapeWebsiteTool(), ScrapeWebsiteToolEnhanced(), SeleniumScrapingTool()],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.7)
)
            ,
        
Agent(
    role=" Market Data Analyst",
    backstory="An analytical expert skilled in interpreting large datasets, capable of extracting actionable insights and generating summaries with statistical rigor.",
    goal="To process and analyze collected market data, providing insights and visual summaries of key trends.",
    allow_delegation=True,
    verbose=True,
    tools=[CodeInterpreterTool(), CustomCodeInterpreterTool()],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.7)
)
            ,
        
Agent(
    role="Market Report Generator",
    backstory="Specializes in transforming analyzed data into structured reports with clarity and actionable recommendations, ensuring all findings are presented in a professional format.",
    goal="To compile the analytical insights and data summaries into a final report, formatted for stakeholder review.",
    allow_delegation=True,
    verbose=True,
    tools=[CodeDocsSearchTool()],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.69)
)
            
    ]
    return agents

def load_tasks(agents):
    tasks = [
        
Task(
    description="\"Collect market data from multiple online sources and process local structured files. This task should leverage all web scraping and file processing tools:\nWeb Scraping & Search: Use ScrapeWebsiteTool, ScrapeWebsiteToolEnhanced, SeleniumScrapingTool, and WebsiteSearchTool to crawl and extract content from target websites.\nData Extraction & File Processing: Use FileReadTool, CSVSearchTool (or CSVSearchToolEnhanced), DOCXSearchTool, PDFSearchTool, JSONSearchTool, and TXTSearchTool to read and search through files.\nFinancial Data: Use YahooFinanceNewsTool to gather current financial news and market trends.\"",
    expected_output="\"A unified raw dataset (e.g., JSON or CSV) containing market data from web pages and files, along with relevant financial news.\"",
    agent=next(agent for agent in agents if agent.role == " Market Data Collector"),
    async_execution=True
)
            ,
        
Task(
    description="Analyze the unified raw dataset to extract market trends, generate insights, and produce visual summaries. This task should employ data analysis tools as follows:\nData Analysis: Use codeInterpreterTool and CustomCodeInterpreterTool to perform statistical evaluations, compute trends, and generate charts or graphs.\"",
    expected_output="A detailed analysis report (structured in JSON, markdown, or similar) that includes key insights, trends, and visualizations derived from the raw data.\"",
    agent=next(agent for agent in agents if agent.role == " Market Data Analyst"),
    async_execution=False
)
            ,
        
Task(
    description="Compile the insights and analysis into a comprehensive market research report. Use the following tools:\nReport Compilation: Use CustomFileWriteTool to format and write the final report document (e.g., DOCX or PDF).\nOptional Multimedia Enrichment: Optionally, use YoutubeVideoSearchTool, YoutubeChannelSearchTool, GithubSearchTool, and CodeDocsSearchTool to embed supporting references or multimedia content.",
    expected_output="A polished, publication-ready market research report file (for example, 'market_report.docx') that clearly presents the analysis, insights, and recommendations.",
    agent=next(agent for agent in agents if agent.role == "Market Report Generator"),
    async_execution=False
)
            
    ]
    return tasks

def main():
    st.title("MarketResearchCrew")

    agents = load_agents()
    tasks = load_tasks(agents)
    crew = Crew(
        agents=agents, 
        tasks=tasks, 
        process="hierarchical", 
        verbose=True, 
        memory=True, 
        cache=True, 
        max_rpm=1000,
        manager_llm=create_llm("OpenAI: gpt-4o-mini")
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
