import gradio as gr
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

def load_data():
    # Initialize global state
    global agents, tasks, crews, tools, enabled_tools, current_page
    agents = db_utils.load_agents()
    tasks = db_utils.load_tasks()
    crews = db_utils.load_crews()
    tools = db_utils.load_tools()
    enabled_tools = db_utils.load_tools_state()
    current_page = 'Crews'

def get_page_component(page_name):
    pages = {
        'Crews': PageCrews(),
        'Tools': PageTools(),
        'Agents': PageAgents(), 
        'Tasks': PageTasks(),
        'Kickoff!': PageCrewRun(),
        'Results': PageResults(),
        'Import/export': PageExportCrew()
    }
    return pages[page_name]

def main():
    load_dotenv()
    load_secrets_fron_env()
    
    if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']):
        try:
            import agentops
            agentops.init(api_key=os.getenv('AGENTOPS_API_KEY'), auto_start_session=False)
        except ModuleNotFoundError as e:
            print(f"Error initializing AgentOps: {str(e)}")

    db_utils.initialize_db()
    load_data()

    with gr.Blocks(title="CrewAI Studio") as app:
        # Header
        with gr.Row():
            gr.Image("img/crewai_logo.png", scale=1)

        # Navigation
        page_names = ['Crews', 'Tools', 'Agents', 'Tasks', 'Kickoff!', 'Results', 'Import/export']
        with gr.Row():
            page_radio = gr.Radio(
                choices=page_names,
                value='Crews',
                label="Navigation",
                interactive=True
            )

        # Content Area
        content_area = gr.Group(visible=True)
        
        def render_page(page_name):
            page = get_page_component(page_name)
            if hasattr(page, 'build_interface'):
                return page.build_interface()
            elif hasattr(page, 'draw'):
                return page.draw()
            else:
                return gr.Markdown(f"Error: Page {page_name} has no interface")

        # Initial page
        with content_area:
            initial_page = get_page_component('Crews')
            initial_interface = initial_page.build_interface()

        # Handle page changes
        page_radio.change(
            fn=render_page,
            inputs=[page_radio],
            outputs=[content_area],
            queue=False
        )

    app.queue()
    app.launch()

if __name__ == '__main__':
    main()
