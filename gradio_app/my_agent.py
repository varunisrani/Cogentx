from crewai import Agent
import gradio as gr
from utils import rnd_id
from db_utils import save_agent, delete_agent
from llms import llm_providers_and_models, create_llm
from datetime import datetime

class MyAgent:
    def __init__(self, id=None, role=None, backstory=None, goal=None, temperature=None, allow_delegation=False, verbose=False, cache=None, llm_provider_model=None, max_iter=None, created_at=None, tools=None):
        self.id = id or "A_" + rnd_id()
        self.role = role or "Senior Researcher"
        self.backstory = backstory or "Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world."
        self.goal = goal or "Uncover groundbreaking technologies in AI"
        self.temperature = temperature or 0.1
        self.allow_delegation = allow_delegation if allow_delegation is not None else False
        self.verbose = verbose if verbose is not None else True
        self.llm_provider_model = llm_providers_and_models()[0] if llm_provider_model is None else llm_provider_model
        self.created_at = created_at or datetime.now().isoformat()
        self.tools = tools or []
        self.max_iter = max_iter or 25
        self.cache = cache if cache is not None else True

    def get_crewai_agent(self) -> Agent:
        llm = create_llm(self.llm_provider_model, temperature=self.temperature)
        tools = [tool.create_tool() for tool in self.tools]
        return Agent(
            role=self.role,
            backstory=self.backstory,
            goal=self.goal,
            allow_delegation=self.allow_delegation,
            verbose=self.verbose,
            max_iter=self.max_iter,
            cache=self.cache,
            tools=tools,
            llm=llm
        )

    def delete(self):
        delete_agent(self.id)
        return True

    def get_tool_display_name(self, tool):
        first_param_name = tool.get_parameter_names()[0] if tool.get_parameter_names() else None
        first_param_value = tool.parameters.get(first_param_name, '') if first_param_name else ''
        return f"{tool.name} ({first_param_value if first_param_value else tool.tool_id})"

    def is_valid(self):
        for tool in self.tools:
            if not tool.is_valid():
                return False
        return True

    def validate_llm_provider_model(self):
        available_models = llm_providers_and_models()
        if self.llm_provider_model not in available_models:
            self.llm_provider_model = available_models[0]

    def to_dict(self):
        return {
            "role": self.role,
            "backstory": self.backstory,
            "goal": self.goal,
            "allow_delegation": self.allow_delegation,
            "verbose": self.verbose,
            "cache": self.cache,
            "llm_provider_model": self.llm_provider_model,
            "temperature": self.temperature,
            "max_iter": self.max_iter,
            "tools": [self.get_tool_display_name(tool) for tool in self.tools]
        }

def create_agent_interface(tools_state):
    """Create a Gradio interface for agent management"""
    
    def update_agent(agent, role, backstory, goal, allow_delegation, verbose, cache, 
                    llm_provider_model, temperature, max_iter, selected_tools):
        agent.role = role
        agent.backstory = backstory
        agent.goal = goal
        agent.allow_delegation = allow_delegation
        agent.verbose = verbose
        agent.cache = cache
        agent.llm_provider_model = llm_provider_model
        agent.temperature = temperature
        agent.max_iter = max_iter
        agent.tools = [tool for tool in tools_state if agent.get_tool_display_name(tool) in selected_tools]
        save_agent(agent)
        return gr.update(visible=False), gr.update(visible=True), agent.to_dict()

    def show_edit_form(agent):
        return gr.update(visible=True), gr.update(visible=False)

    def delete_current_agent(agent):
        agent.delete()
        return True

    with gr.Group():
        with gr.Row():
            view_area = gr.JSON(label="Agent Details")
            
        with gr.Row():
            edit_btn = gr.Button("Edit")
            delete_btn = gr.Button("Delete", variant="stop")
            
        with gr.Group() as edit_form:
            role = gr.Textbox(label="Role")
            backstory = gr.Textbox(label="Backstory", lines=3)
            goal = gr.Textbox(label="Goal", lines=2)
            with gr.Row():
                allow_delegation = gr.Checkbox(label="Allow delegation")
                verbose = gr.Checkbox(label="Verbose")
                cache = gr.Checkbox(label="Cache")
            llm_provider_model = gr.Dropdown(
                label="LLM Provider and Model",
                choices=llm_providers_and_models()
            )
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                step=0.1
            )
            max_iter = gr.Number(
                label="Max Iterations",
                minimum=1,
                maximum=100,
                step=1
            )
            tools_list = gr.Dropdown(
                label="Select Tools",
                choices=[],
                multiselect=True
            )
            save_btn = gr.Button("Save", variant="primary")
            cancel_btn = gr.Button("Cancel")

    def initialize_form(agent):
        data = agent.to_dict()
        return [
            data["role"],
            data["backstory"],
            data["goal"],
            data["allow_delegation"],
            data["verbose"],
            data["cache"],
            data["llm_provider_model"],
            data["temperature"],
            data["max_iter"],
            data["tools"]
        ]

    return {
        "view_area": view_area,
        "edit_form": edit_form,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "save_btn": save_btn,
        "cancel_btn": cancel_btn,
        "form_inputs": [role, backstory, goal, allow_delegation, verbose, cache,
                       llm_provider_model, temperature, max_iter, tools_list]
    }