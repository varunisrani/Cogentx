import streamlit as st
from streamlit import session_state as ss
import db_utils
from pg_agents import PageAgents
from pg_tasks import PageTasks
from pg_crews import PageCrews
from pg_tools import PageTools
from pg_crew_run import PageCrewRun
from pg_export_crew import PageExportCrew
from pg_results import PageResults
from dotenv import load_dotenv
from llms import load_secrets_fron_env
import os

# Custom CSS for dark theme
CUSTOM_CSS = """
<style>
    /* Dark theme colors */
    :root {
        --primary-color: #0066cc;
        --background-color: #1E1E1E;
        --secondary-bg: #2D2D2D;
        --text-color: #FFFFFF;
    }

    /* Override Streamlit's default theme */
    .stApp {
        background-color: var(--background-color);
    }

    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 4px;
    }

    .stTextInput>div>div>input {
        border-radius: 4px;
        background-color: var(--secondary-bg);
        color: var(--text-color);
    }

    /* Header styling */
    .main-header {
        color: var(--text-color);
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
    }

    /* Make text white */
    .stMarkdown, p, h1, h2, h3, label {
        color: var(--text-color) !important;
    }

    /* Style radio buttons */
    .stRadio > label {
        color: var(--text-color) !important;
    }

    /* Style selectbox */
    .stSelectbox > div > div {
        background-color: var(--secondary-bg);
        color: var(--text-color);
    }
</style>
"""

def pages():
    return {
        'Crews': PageCrews(),
        'Tools': PageTools(),
        'Agents': PageAgents(),
        'Tasks': PageTasks(),
        'Kickoff!': PageCrewRun(),
        'Results': PageResults(),
        'Import/export': PageExportCrew()
    }

def load_data():
    ss.agents = db_utils.load_agents()
    ss.tasks = db_utils.load_tasks()
    ss.crews = db_utils.load_crews()
    ss.tools = db_utils.load_tools()
    ss.enabled_tools = db_utils.load_tools_state()

def draw_sidebar():
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='color: #0066cc;'>CogentX Studio</h2>
            </div>
        """, unsafe_allow_html=True)

        if 'page' not in ss:
            ss.page = 'Crews'
        
        selected_page = st.radio('Navigation', list(pages().keys()), index=list(pages().keys()).index(ss.page), key='nav_radio')
        if selected_page != ss.page:
            ss.page = selected_page
            st.rerun()
            
def main():
    st.set_page_config(
        page_title="CogentX Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# CogentX Studio\nPowered by advanced AI technology"
        }
    )
    
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    load_dotenv()
    load_secrets_fron_env()
    if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not ss.get('agentops_failed', False):
        try:
            import agentops
            agentops.init(api_key=os.getenv('AGENTOPS_API_KEY'),auto_start_session=False)    
        except ModuleNotFoundError as e:
            ss.agentops_failed = True
            print(f"Error initializing AgentOps: {str(e)}")            
        
    db_utils.initialize_db()
    load_data()
    draw_sidebar()
    
    # Add main header
    st.markdown("<h1 class='main-header'>Welcome to CogentX Studio</h1>", unsafe_allow_html=True)
    
    PageCrewRun.maintain_session_state()
    pages()[ss.page].draw()
    
if __name__ == '__main__':
    main()
