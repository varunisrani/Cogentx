import gradio as gr
from my_crew import MyCrew, Process
import db_utils

class PageCrews:
    def __init__(self):
        self.name = "Crews"
        self.crews = []
        self.available_agents = []
        self.available_tasks = []
        self.load_data()

    def load_data(self):
        """Load all necessary data"""
        self.load_crews()
        self.available_agents = db_utils.load_agents()
        self.available_tasks = db_utils.load_tasks()

    def load_crews(self):
        """Load crews from database"""
        self.crews = db_utils.load_crews()

    def create_crew(self):
        """Create a new crew"""
        crew = MyCrew()
        self.crews.append(crew)
        db_utils.save_crew(crew)
        return self.update_crews_list()

    def update_crews_list(self):
        """Update the crews list display"""
        if not self.crews:
            return gr.HTML(value="<div class='empty-state'>No crews defined yet.</div>")

        crews_html = "<div class='crews-list'>"
        for crew in self.crews:
            crews_html += f"""
            <div class='crew-card'>
                <h3>Crew: {crew.name}</h3>
                <div class='crew-details'>
                    <p><strong>Process:</strong> {crew.process}</p>
                    <p><strong>Agents:</strong> {', '.join([agent.role for agent in crew.agents]) if crew.agents else 'None'}</p>
                    <p><strong>Tasks:</strong> {len(crew.tasks)}</p>
                </div>
            </div>
            """
        crews_html += "</div>"
        
        # Add some CSS styling
        crews_html = f"""
        <style>
            .crews-list {{ padding: 10px; }}
            .crew-card {{
                border: 1px solid #ddd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .crew-card h3 {{ margin-top: 0; color: #2a2a2a; }}
            .crew-details {{ margin-top: 10px; }}
            .crew-details p {{ margin: 5px 0; }}
            .empty-state {{
                text-align: center;
                padding: 40px;
                color: #666;
                font-style: italic;
            }}
        </style>
        {crews_html}
        """
        return gr.HTML(value=crews_html)

    def build_interface(self):
        """Build the Gradio interface"""
        with gr.Column() as interface:
            gr.Markdown(f"## {self.name}")
            
            # Create button at the top
            create_btn = gr.Button("Create New Crew", size="lg", variant="primary")
            
            # Crews list display
            crews_display = self.update_crews_list()
            
            # Handle create button click
            create_btn.click(
                fn=self.create_crew,
                outputs=[crews_display]
            )
            
        return interface
