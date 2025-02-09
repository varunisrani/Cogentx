import gradio as gr
import zipfile
import os
import re
import json
import shutil
import db_utils
from utils import escape_quotes
from my_tools import TOOL_CLASSES
from crewai import Process
from my_crew import MyCrew
from my_agent import MyAgent
from my_task import MyTask
from datetime import datetime

class PageExportCrew:
    def __init__(self):
        self.name = "Import/export"
        self.state = {
            'crews': [],
            'selected_crew': None,
            'upload_error': None
        }
        self.load_data()

    def load_data(self):
        """Load current crews data"""
        self.state['crews'] = db_utils.load_crews()

    def extract_placeholders(self, text):
        return re.findall(r'\{(.*?)\}', text)

    def get_placeholders_from_crew(self, crew):
        placeholders = set()
        for task in crew.tasks:
            placeholders.update(self.extract_placeholders(task.description))
            placeholders.update(self.extract_placeholders(task.expected_output))
        return list(placeholders)

    def export_crew_to_json(self, crew):
        """Export crew data to JSON format"""
        crew_data = {
            'id': crew.id,
            'name': crew.name,
            'process': crew.process,
            'verbose': crew.verbose,
            'memory': crew.memory,
            'cache': crew.cache,
            'max_rpm': crew.max_rpm,
            'manager_llm': crew.manager_llm,
            'manager_agent': crew.manager_agent.id if crew.manager_agent else None,
            'created_at': crew.created_at,
            'agents': [],
            'tasks': [],
            'tools': []
        }

        tool_ids = set()
        
        # Export agents
        for agent in crew.agents:
            agent_data = {
                'id': agent.id,
                'role': agent.role,
                'backstory': agent.backstory,
                'goal': agent.goal,
                'allow_delegation': agent.allow_delegation,
                'verbose': agent.verbose,
                'cache': agent.cache,
                'llm_provider_model': agent.llm_provider_model,
                'temperature': agent.temperature,
                'max_iter': agent.max_iter,
                'tool_ids': [tool.tool_id for tool in agent.tools]
            }
            crew_data['agents'].append(agent_data)
            tool_ids.update(agent_data['tool_ids'])

        # Export tasks
        for task in crew.tasks:
            task_data = {
                'id': task.id,
                'description': task.description,
                'expected_output': task.expected_output,
                'async_execution': task.async_execution,
                'agent_id': task.agent.id if task.agent else None,
                'context_from_async_tasks_ids': task.context_from_async_tasks_ids,
                'created_at': task.created_at
            }
            crew_data['tasks'].append(task_data)

        # Export tools
        tools = db_utils.load_tools()
        tools_dict = {tool.tool_id: tool for tool in tools}
        for tool_id in tool_ids:
            if tool_id in tools_dict:
                tool = tools_dict[tool_id]
                tool_data = {
                    'tool_id': tool.tool_id,
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.get_parameters()
                }
                crew_data['tools'].append(tool_data)

        return json.dumps(crew_data, indent=2)

    def import_crew_from_json(self, crew_data):
        """Import crew from JSON data"""
        tools = []
        agents = []
        tasks = []

        # Create tools
        for tool_data in crew_data['tools']:
            tool_class = TOOL_CLASSES[tool_data['name']]
            tool = tool_class(tool_id=tool_data['tool_id'])
            tool.set_parameters(**tool_data['parameters'])
            tools.append(tool)
            db_utils.save_tool(tool)

        # Create agents
        for agent_data in crew_data['agents']:
            agent = MyAgent(
                id=agent_data['id'],
                role=agent_data['role'],
                backstory=agent_data['backstory'],
                goal=agent_data['goal'],
                allow_delegation=agent_data['allow_delegation'],
                verbose=agent_data['verbose'],
                cache=agent_data.get('cache', True),
                llm_provider_model=agent_data['llm_provider_model'],
                temperature=agent_data['temperature'],
                max_iter=agent_data['max_iter'],
                created_at=agent_data.get('created_at')
            )
            agent.tools = [tool for tool in tools if tool.tool_id in agent_data['tool_ids']]
            agents.append(agent)
            db_utils.save_agent(agent)

        # Create tasks
        for task_data in crew_data['tasks']:
            task = MyTask(
                id=task_data['id'],
                description=task_data['description'],
                expected_output=task_data['expected_output'],
                async_execution=task_data['async_execution'],
                agent=next((agent for agent in agents if agent.id == task_data['agent_id']), None),
                context_from_async_tasks_ids=task_data['context_from_async_tasks_ids'],
                created_at=task_data['created_at']
            )
            tasks.append(task)
            db_utils.save_task(task)

        # Create crew
        crew = MyCrew(
            id=crew_data['id'],
            name=crew_data['name'],
            process=crew_data['process'],
            verbose=crew_data['verbose'],
            memory=crew_data['memory'],
            cache=crew_data['cache'],
            max_rpm=crew_data['max_rpm'],
            manager_llm=crew_data['manager_llm'],
            manager_agent=next((agent for agent in agents if agent.id == crew_data['manager_agent']), None),
            created_at=crew_data['created_at']
        )
        crew.agents = agents
        crew.tasks = tasks
        db_utils.save_crew(crew)

        return crew

    def export_all_to_json(self):
        """Export all database content to JSON"""
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"all_crews_{current_datetime}.json"
        db_utils.export_to_json(file_path)
        return file_path

    def create_interface(self):
        """Create the Gradio interface for import/export functionality"""
        def handle_file_upload(file):
            """Handle JSON file upload"""
            try:
                content = file.decode('utf-8')
                json_data = json.loads(content)
                
                if isinstance(json_data, list):  # Full database export
                    temp_file = "uploaded_file.json"
                    with open(temp_file, "w") as f:
                        json.dump(json_data, f)
                    db_utils.import_from_json(temp_file)
                    os.remove(temp_file)
                    return "Full database JSON file imported successfully!", gr.update(value=None)
                elif isinstance(json_data, dict) and 'id' in json_data:  # Single crew export
                    crew = self.import_crew_from_json(json_data)
                    return f"Crew '{crew.name}' imported successfully!", gr.update(value=None)
                else:
                    return "Invalid JSON format. Please upload a valid crew or full database export file.", gr.update(value=None)
            except Exception as e:
                return f"Error importing file: {str(e)}", gr.update(value=None)

        def export_crew(crew_name):
            """Handle crew export"""
            if not crew_name:
                return "Please select a crew to export."
            
            crew = next((c for c in self.state['crews'] if c.name == crew_name), None)
            if crew:
                json_data = self.export_crew_to_json(crew)
                return json_data
            return "Selected crew not found."

        with gr.Blocks() as export_interface:
            gr.Markdown(f"## {self.name}")

            with gr.Row():
                # Export section
                with gr.Column():
                    gr.Markdown("### Export")
                    export_all_btn = gr.Button("Export All to JSON")
                    crew_select = gr.Dropdown(
                        choices=[crew.name for crew in self.state['crews']],
                        label="Select crew to export",
                        interactive=True
                    )
                    export_crew_btn = gr.Button("Export Selected Crew")
                    export_output = gr.JSON(label="Export Result")

                # Import section
                with gr.Column():
                    gr.Markdown("### Import")
                    file_upload = gr.File(
                        label="Upload JSON File",
                        file_types=[".json"]
                    )
                    import_status = gr.Textbox(
                        label="Import Status",
                        interactive=False
                    )

            # Event handlers
            export_all_btn.click(
                fn=self.export_all_to_json,
                outputs=[export_output]
            )

            export_crew_btn.click(
                fn=export_crew,
                inputs=[crew_select],
                outputs=[export_output]
            )

            file_upload.upload(
                fn=handle_file_upload,
                outputs=[import_status, file_upload]
            )

        return export_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()