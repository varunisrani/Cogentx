import gradio as gr
from my_agent import MyAgent
import db_utils

class PageAgents:
    def __init__(self):
        self.name = "Agents"
        self.agents = db_utils.load_agents()
        self.crews = db_utils.load_crews()
        self.editing = False

    def create_agent(self, crew=None):
        agent = MyAgent()
        self.agents.append(agent)
        agent.edit = True
        db_utils.save_agent(agent)

        if crew:
            crew.agents.append(agent)
            db_utils.save_crew(crew)

        return agent

    def draw(self):
        with gr.Blocks() as interface:
            gr.Markdown(f"## {self.name}")

            # Dictionary to track agent assignment
            agent_assignment = {agent.id: [] for agent in self.agents}

            # Assign agents to crews
            for crew in self.crews:
                for agent in crew.agents:
                    agent_assignment[agent.id].append(crew.name)

            with gr.Tabs() as tabs:
                # All Agents tab
                with gr.Tab("All Agents"):
                    gr.Markdown("#### All Agents")
                    for agent in self.agents:
                        agent.draw()
                        if agent.edit:
                            self.editing = True
                    create_btn = gr.Button("Create agent", interactive=not self.editing)
                    create_btn.click(fn=self.create_agent)

                # Unassigned Agents tab
                with gr.Tab("Unassigned Agents"):
                    gr.Markdown("#### Unassigned Agents")
                    unassigned_agents = [agent for agent in self.agents if not agent_assignment[agent.id]]
                    for agent in unassigned_agents:
                        agent.draw(key=f"{agent.id}_unassigned")
                        if agent.edit:
                            self.editing = True
                    create_btn = gr.Button("Create agent", interactive=not self.editing)
                    create_btn.click(fn=self.create_agent)

                # Crew tabs
                for crew in self.crews:
                    with gr.Tab(crew.name):
                        gr.Markdown(f"#### {crew.name}")
                        assigned_agents = [agent for agent in crew.agents]
                        for agent in assigned_agents:
                            agent.draw(key=f"{agent.id}_{crew.name}")
                            if agent.edit:
                                self.editing = True
                        create_btn = gr.Button("Create agent", interactive=not self.editing)
                        create_btn.click(fn=lambda c=crew: self.create_agent(crew=c))

            if len(self.agents) == 0:
                gr.Markdown("No agents defined yet.")
                create_btn = gr.Button("Create agent", interactive=not self.editing)
                create_btn.click(fn=self.create_agent)

        return interface
