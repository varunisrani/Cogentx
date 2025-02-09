import gradio as gr
from my_crew import MyCrew, create_crew_interface
from crewai import Process
import db_utils

class PageCrews:
    def __init__(self):
        self.name = "Crews"
        self.state = {
            'crews': [],
            'editing': False,
            'available_agents': [],
            'available_tasks': []
        }
        self.load_data()

    def load_data(self):
        """Load all necessary data"""
        self.load_crews()
        self.state['available_agents'] = db_utils.load_agents()
        self.state['available_tasks'] = db_utils.load_tasks()

    def load_crews(self):
        """Load crews from database"""
        self.state['crews'] = db_utils.load_crews()

    def create_crew(self):
        """Create a new crew"""
        crew = MyCrew()
        self.state['crews'].append(crew)
        db_utils.save_crew(crew)
        return True, crew.to_dict()

    def update_crews_list(self):
        """Update the crews list display"""
        if not self.state['crews']:
            return []
        
        crews_data = []
        for crew in self.state['crews']:
            crews_data.append([
                crew.name,
                crew.description,
                ", ".join([a.role for a in crew.agents]),
                ", ".join([t.description for t in crew.tasks]),
                crew.process.value,
                crew.manager_llm if crew.manager_llm else "None",
                crew.manager_agent.role if crew.manager_agent else "None",
                crew.verbose,
                crew.memory,
                crew.cache,
                crew.planning,
                crew.max_rpm
            ])
        return crews_data

    def save_crew(self, crew_data):
        """Save updated crew data"""
        crew = next((c for c in self.state['crews'] if c.id == crew_data['id']), None)
        if crew:
            crew.name = crew_data['name']
            crew.description = crew_data['description']
            crew.process = Process[crew_data['process']]
            crew.verbose = crew_data['verbose']
            crew.memory = crew_data['memory']
            crew.cache = crew_data['cache']
            crew.planning = crew_data['planning']
            crew.max_rpm = crew_data['max_rpm']
            crew.manager_llm = None if crew_data['manager_llm'] == "None" else crew_data['manager_llm']
            
            # Update manager agent
            manager_agent_role = crew_data.get('manager_agent')
            crew.manager_agent = next(
                (a for a in self.state['available_agents'] if a.role == manager_agent_role),
                None
            ) if manager_agent_role and manager_agent_role != "None" else None

            # Update agents
            crew.agents = [
                a for a in self.state['available_agents']
                if a.role in crew_data.get('agents', [])
            ]

            # Update tasks
            crew.tasks = [
                t for t in self.state['available_tasks']
                if t.id in crew_data.get('tasks', []) 
                and t.agent.id in [agent.id for agent in crew.agents]
            ]

            db_utils.save_crew(crew)
            return True
        return False

    def delete_crew(self, crew_id):
        """Delete a crew"""
        crew = next((c for c in self.state['crews'] if c.id == crew_id), None)
        if crew:
            self.state['crews'].remove(crew)
            db_utils.delete_crew(crew_id)
            return True
        return False

    def refresh_view(self):
        self.load_crews()
        crews_data = []
        for crew in self.state['crews']:
            crews_data.append([
                crew.name,
                crew.description,
                ", ".join([a.role for a in crew.agents]),
                ", ".join([t.description for t in crew.tasks]),
                crew.process,
                crew.manager_llm if crew.manager_llm else "None",
                crew.manager_agent.role if crew.manager_agent else "None",
                crew.verbose,
                crew.memory,
                crew.cache,
                crew.planning,
                crew.max_rpm
            ])
        return gr.DataFrame(
            value=crews_data,
            headers=["Name", "Description", "Agents", "Tasks", "Process", "Manager LLM", "Manager Agent", 
                    "Verbose", "Memory", "Cache", "Planning", "Max RPM"]
        )

    def create_interface(self):
        """Create the Gradio interface for crews management"""
        custom_css = """
            .crew-container {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .crew-header {
                color: #2D3748;
                font-weight: 600;
                margin-bottom: 1.5rem;
            }
            .crew-form {
                background: #f8f9fa;
                padding: 1.5rem;
                border-radius: 6px;
                border: 1px solid #eaeaea;
            }
            .crew-actions {
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid #eaeaea;
            }
        """

        with gr.Blocks(css=custom_css) as crews_interface:
            with gr.Row(elem_classes="crew-container"):
                gr.Markdown("## Crew Management", elem_classes="crew-header")
                
                # Status message for feedback
                status_msg = gr.Markdown(visible=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        crews_list = gr.DataFrame(
                            headers=["Name", "Description", "Agents", "Tasks", "Process", "Manager LLM", "Manager Agent", 
                                    "Verbose", "Memory", "Cache", "Planning", "Max RPM"],
                            label="Available Crews",
                            interactive=False
                        )

                        create_button = gr.Button(
                            "➕ Create New Crew",
                            variant="primary",
                            size="lg",
                            interactive=not self.state['editing']
                        )

                    with gr.Column(scale=1, visible=False, elem_classes="crew-form") as crew_edit_box:
                        gr.Markdown("### Configure Your Crew", elem_classes="crew-header")
                        crew_interface_components = create_crew_interface(self.state.get('available_agents', []), 
                                                                       self.state.get('available_tasks', []))

            def handle_create_crew():
                success, crew_data = self.create_crew()
                if success:
                    return {
                        crews_list: self.refresh_view(),
                        crew_edit_box: gr.update(visible=True),
                        create_button: gr.update(interactive=False),
                        status_msg: gr.update(value="✅ New crew created successfully", visible=True)
                    }
                return {
                    crews_list: self.refresh_view(),
                    crew_edit_box: gr.update(visible=False),
                    create_button: gr.update(interactive=True),
                    status_msg: gr.update(value="❌ Failed to create crew", visible=True)
                }

            def handle_crew_save():
                self.state['editing'] = False
                valid, message = self.validate_crew()
                if valid:
                    return {
                        crews_list: self.refresh_view(),
                        crew_edit_box: gr.update(visible=False),
                        create_button: gr.update(interactive=True),
                        status_msg: gr.update(value="✅ Crew saved successfully", visible=True)
                    }
                return {
                    status_msg: gr.update(value=f"❌ {message}", visible=True)
                }

            def handle_crew_cancel():
                self.state['editing'] = False
                return {
                    crew_edit_box: gr.update(visible=False),
                    create_button: gr.update(interactive=True),
                    status_msg: gr.update(visible=False)
                }

            # Event handlers
            create_button.click(
                fn=handle_create_crew,
                outputs=[crews_list, crew_edit_box, create_button, status_msg]
            )

            if 'save_btn' in crew_interface_components:
                crew_interface_components['save_btn'].click(
                    fn=handle_crew_save,
                    outputs=[crews_list, crew_edit_box, create_button, status_msg]
                )

            if 'cancel_btn' in crew_interface_components:
                crew_interface_components['cancel_btn'].click(
                    fn=handle_crew_cancel,
                    outputs=[crew_edit_box, create_button, status_msg]
                )

            # Initialize with data
            crews_list.value = self.refresh_view()
            
            # Setup periodic refresh
            crews_interface.load(
                fn=self.refresh_view,
                outputs=[crews_list]
            )

        return crews_interface

    def validate_crew(self):
        """Validate crew configuration"""
        if not self.state['crews']:
            return False, "No crew data found"
        current_crew = self.state['crews'][-1]
        if not current_crew.name:
            return False, "Crew name is required"
        if not current_crew.agents:
            return False, "At least one agent must be selected"
        if not current_crew.tasks:
            return False, "At least one task must be assigned"
        if current_crew.process == Process.hierarchical:
            if not current_crew.manager_llm and not current_crew.manager_agent:
                return False, "Hierarchical process requires either a Manager LLM or Manager Agent"
        return True, "Crew configuration is valid"

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()