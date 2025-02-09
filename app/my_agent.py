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
        self.edit_mode = False

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
        return gr.update(visible=False)

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

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        return self.create_ui()

    def save_changes(self, role, backstory, goal, allow_delegation, verbose, cache, 
                    llm_provider_model, temperature, max_iter, selected_tools):
        self.role = role
        self.backstory = backstory
        self.goal = goal
        self.allow_delegation = allow_delegation
        self.verbose = verbose
        self.cache = cache
        self.llm_provider_model = llm_provider_model
        self.temperature = temperature
        self.max_iter = max_iter
        
        enabled_tools = [tool for tool in self.tools]  # Assuming tools are stored somewhere
        self.tools = [tool for tool in enabled_tools if self.get_tool_display_name(tool) in selected_tools]
        
        save_agent(self)
        self.edit_mode = False
        return self.create_ui()

    def create_ui(self):
        self.validate_llm_provider_model()
        
        with gr.Group() as agent_group:
            if self.edit_mode:
                with gr.Box():
                    gr.Markdown(f"### Edit Agent: {self.role}")
                    role = gr.Textbox(label="Role", value=self.role)
                    backstory = gr.Textbox(label="Backstory", value=self.backstory, lines=3)
                    goal = gr.Textbox(label="Goal", value=self.goal, lines=2)
                    allow_delegation = gr.Checkbox(label="Allow delegation", value=self.allow_delegation)
                    verbose = gr.Checkbox(label="Verbose", value=self.verbose)
                    cache = gr.Checkbox(label="Cache", value=self.cache)
                    llm_provider_model = gr.Dropdown(
                        choices=llm_providers_and_models(),
                        value=self.llm_provider_model,
                        label="LLM Provider and Model"
                    )
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.0, value=self.temperature,
                        label="Temperature"
                    )
                    max_iter = gr.Number(
                        minimum=1, maximum=100, value=self.max_iter,
                        label="Max Iterations"
                    )
                    
                    enabled_tools = [tool for tool in self.tools]
                    tool_choices = [self.get_tool_display_name(tool) for tool in enabled_tools]
                    selected_tools = gr.Dropdown(
                        choices=tool_choices,
                        value=[self.get_tool_display_name(tool) for tool in self.tools],
                        multiselect=True,
                        label="Select Tools"
                    )
                    
                    save_btn = gr.Button("Save")
                    save_btn.click(
                        fn=self.save_changes,
                        inputs=[role, backstory, goal, allow_delegation, verbose, 
                               cache, llm_provider_model, temperature, max_iter, selected_tools],
                        outputs=agent_group
                    )
            else:
                with gr.Box():
                    expander_title = f"{self.role[:60]} -{self.llm_provider_model.split(':')[1]}" if self.is_valid() else f"❗ {self.role[:20]} -{self.llm_provider_model.split(':')[1]}"
                    gr.Markdown(f"### {expander_title}")
                    gr.Markdown(f"**Role:** {self.role}")
                    gr.Markdown(f"**Backstory:** {self.backstory}")
                    gr.Markdown(f"**Goal:** {self.goal}")
                    gr.Markdown(f"**Allow delegation:** {self.allow_delegation}")
                    gr.Markdown(f"**Verbose:** {self.verbose}")
                    gr.Markdown(f"**Cache:** {self.cache}")
                    gr.Markdown(f"**LLM Provider and Model:** {self.llm_provider_model}")
                    gr.Markdown(f"**Temperature:** {self.temperature}")
                    gr.Markdown(f"**Max Iterations:** {self.max_iter}")
                    gr.Markdown(f"**Tools:** {[self.get_tool_display_name(tool) for tool in self.tools]}")

                    with gr.Row():
                        edit_btn = gr.Button("Edit")
                        delete_btn = gr.Button("Delete")
                        
                    edit_btn.click(fn=self.toggle_edit_mode, outputs=agent_group)
                    delete_btn.click(fn=self.delete, outputs=agent_group)
                    
        return agent_group