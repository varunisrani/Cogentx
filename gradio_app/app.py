import gradio as gr
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
import db_utils

class State:
    def __init__(self):
        self.agents = []
        self.tasks = []
        self.crews = []
        self.tools = []
        self.enabled_tools = []
        self.agentops_failed = False
        self.current_page = 'Crews'
        # Initialize crew run state
        self.crew_thread = None
        self.result = None
        self.running = False
        self.message_queue = None
        self.selected_crew_name = None
        self.placeholders = {}
        self.console_output = []
        self.console_expanded = True

state = State()

def get_pages():
    return {
        'Crews': PageCrews(),
        'Tools': PageTools(),
        'Agents': PageAgents(),
        'Tasks': PageTasks(),
        'Kickoff!': PageCrewRun(),
        'Results': PageResults(),
        'Import/Export': PageExportCrew()
    }

def load_data():
    """Load all data into state"""
    state.agents = db_utils.load_agents()
    state.tasks = db_utils.load_tasks()
    state.crews = db_utils.load_crews()
    state.tools = db_utils.load_tools()
    state.enabled_tools = db_utils.load_tools_state()

def initialize_app():
    """Initialize the application"""
    load_dotenv()
    load_secrets_fron_env()
    
    # Initialize AgentOps if enabled
    if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not state.agentops_failed:
        try:
            import agentops
            agentops.init(api_key=os.getenv('AGENTOPS_API_KEY'), auto_start_session=False)    
        except ModuleNotFoundError as e:
            state.agentops_failed = True
            print(f"Error initializing AgentOps: {str(e)}")
    
    db_utils.initialize_db()
    load_data()

def main():
    """Main application entry point"""
    initialize_app()
    
    with gr.Blocks(title="CrewAI Studio", theme=gr.themes.Soft()) as app:
        # Header and Navigation
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image("img/crewai_logo.png", scale=1, min_width=100)
            with gr.Column(scale=3):
                gr.Markdown("# CrewAI Studio")
        
        # Main content area with tabs for different pages
        with gr.Tabs() as tabs:
            pages = get_pages()
            for page_name, page_instance in pages.items():
                with gr.Tab(page_name):
                    page_instance.draw()
        
        # Initialize data in the interface
        def refresh_data():
            load_data()
            if isinstance(pages.get('Kickoff!'), PageCrewRun):
                pages['Kickoff!'].maintain_session_state()
        
        # Refresh data periodically
        app.load(
            fn=refresh_data
        )
        # app.every(
        #     fn=refresh_data,
        #     seconds=5  # Refresh every 5 seconds
        # )
    return app

if __name__ == "__main__":
    app = main()
    app.launch()