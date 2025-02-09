import gradio as gr
from my_agent import MyAgent, create_agent_interface
import db_utils

class PageAgents:
    def __init__(self):
        self.name = "Agents"
        self.agents = []
        self.crews = []

    def load_data(self):
        """Load agents and crews from database"""
        self.agents = db_utils.load_agents()
        self.crews = db_utils.load_crews()

    def create_agent(self, crew=None):
        """Create a new agent and optionally assign it to a crew"""
        agent = MyAgent()
        self.agents.append(agent)
        db_utils.save_agent(agent)

        if crew:
            crew.agents.append(agent)
            db_utils.save_crew(crew)

        return agent

    def get_agent_assignments(self):
        """Get dictionary mapping agent IDs to their crew assignments"""
        agent_assignment = {agent.id: [] for agent in self.agents}
        for crew in self.crews:
            for agent in crew.agents:
                agent_assignment[agent.id].append(crew.name)
        return agent_assignment

    def create_interface(self):
        """Create the Gradio interface for the Agents page"""
        def refresh_agents_view():
            self.load_data()
            agent_assignment = self.get_agent_assignments()
            
            # Update all agents tab
            all_agents_html = "<h3>All Agents</h3>"
            for agent in self.agents:
                crews = agent_assignment[agent.id]
                crew_info = f" (In crews: {', '.join(crews)})" if crews else " (Unassigned)"
                all_agents_html += f"<div>{agent.role}{crew_info}</div>"

            # Update unassigned agents tab
            unassigned_html = "<h3>Unassigned Agents</h3>"
            unassigned_agents = [agent for agent in self.agents if not agent_assignment[agent.id]]
            for agent in unassigned_agents:
                unassigned_html += f"<div>{agent.role}</div>"

            # Update crew tabs
            crew_htmls = {}
            for crew in self.crews:
                crew_html = f"<h3>{crew.name} Agents</h3>"
                assigned_agents = [agent for agent in crew.agents]
                for agent in assigned_agents:
                    crew_html += f"<div>{agent.role}</div>"
                crew_htmls[crew.name] = crew_html

            return (
                all_agents_html,
                unassigned_html,
                *[crew_htmls.get(crew.name, "") for crew in self.crews]
            )

        def create_new_agent(crew_name=None):
            crew = next((c for c in self.crews if c.name == crew_name), None) if crew_name else None
            agent = self.create_agent(crew)
            return refresh_agents_view()

        with gr.Blocks() as agents_interface:
            gr.Markdown(f"## {self.name}")
            
            with gr.Tabs() as tabs:
                # All Agents tab
                with gr.Tab("All Agents"):
                    all_agents_view = gr.HTML()
                    create_all_btn = gr.Button("Create Agent")

                # Unassigned Agents tab
                with gr.Tab("Unassigned Agents"):
                    unassigned_view = gr.HTML()
                    create_unassigned_btn = gr.Button("Create Agent")

                # Crew-specific tabs
                crew_views = []
                crew_buttons = []
                for crew in self.crews:
                    with gr.Tab(crew.name):
                        crew_view = gr.HTML()
                        crew_views.append(crew_view)
                        create_crew_btn = gr.Button(f"Create Agent for {crew.name}")
                        crew_buttons.append((create_crew_btn, crew.name))

            # Set up event handlers
            create_all_btn.click(
                lambda: create_new_agent(),
                outputs=[all_agents_view, unassigned_view] + crew_views
            )
            
            create_unassigned_btn.click(
                lambda: create_new_agent(),
                outputs=[all_agents_view, unassigned_view] + crew_views
            )

            for btn, crew_name in crew_buttons:
                btn.click(
                    lambda n=crew_name: create_new_agent(n),
                    outputs=[all_agents_view, unassigned_view] + crew_views
                )

            # Initialize views
            self.load_data()
            initial_views = refresh_agents_view()
            all_agents_view.value = initial_views[0]
            unassigned_view.value = initial_views[1]
            for view, content in zip(crew_views, initial_views[2:]):
                view.value = content

        return agents_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()