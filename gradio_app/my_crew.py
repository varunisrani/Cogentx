from crewai import Crew, Process
import gradio as gr
from utils import rnd_id
from datetime import datetime
from llms import llm_providers_and_models, create_llm
import db_utils

class MyCrew:
    def __init__(self, id=None, name=None, description=None, agents=None, tasks=None, process=None, cache=None, max_rpm=None, verbose=None, manager_llm=None, manager_agent=None, created_at=None, memory=None, planning=None):
        self.id = id or "C_" + rnd_id()
        self.name = name or "Crew 1"
        self.description = description or f"Crew {self.name}"
        self.agents = agents or []
        self.tasks = tasks or []
        self.process = process or Process.sequential
        self.verbose = bool(verbose) if verbose is not None else True
        self.manager_llm = manager_llm
        self.manager_agent = manager_agent
        self.memory = memory if memory is not None else False
        self.cache = cache if cache is not None else True
        self.max_rpm = max_rpm or 1000
        self.planning = planning if planning is not None else False
        self.created_at = created_at or datetime.now().isoformat()

    def get_crewai_crew(self, *args, **kwargs) -> Crew:
        crewai_agents = [agent.get_crewai_agent() for agent in self.agents]

        task_objects = {}

        def create_task(task):
            if task.id in task_objects:
                return task_objects[task.id]

            context_tasks = []
            if task.async_execution or task.context_from_async_tasks_ids or task.context_from_sync_tasks_ids:
                for context_task_id in (task.context_from_async_tasks_ids or []) + (task.context_from_sync_tasks_ids or []):
                    if context_task_id not in task_objects:
                        context_task = next((t for t in self.tasks if t.id == context_task_id), None)
                        if context_task:
                            context_tasks.append(create_task(context_task))
                        else:
                            print(f"Warning: Context task with id {context_task_id} not found for task {task.id}")
                    else:
                        context_tasks.append(task_objects[context_task_id])

            if task.async_execution or context_tasks:
                crewai_task = task.get_crewai_task(context_from_async_tasks=context_tasks)
            else:
                crewai_task = task.get_crewai_task()

            task_objects[task.id] = crewai_task
            return crewai_task

        for task in self.tasks:
            create_task(task)

        crewai_tasks = [task_objects[task.id] for task in self.tasks]

        crew_params = {
            "agents": crewai_agents,
            "tasks": crewai_tasks,
            "cache": self.cache,
            "process": self.process,
            "max_rpm": self.max_rpm,
            "verbose": self.verbose,
            "memory": self.memory,
            "planning": self.planning,
            **kwargs
        }

        if self.manager_llm:
            crew_params["manager_llm"] = create_llm(self.manager_llm)
        elif self.manager_agent:
            crew_params["manager_agent"] = self.manager_agent.get_crewai_agent()

        return Crew(**crew_params)

    def delete(self):
        db_utils.delete_crew(self.id)
        return True

    def is_valid(self):
        if len(self.agents) == 0:
            return False, "Crew has no agents"
        if len(self.tasks) == 0:
            return False, "Crew has no tasks"
        if any([not agent.is_valid() for agent in self.agents]):
            return False, "One or more agents are invalid"
        if any([not task.is_valid() for task in self.tasks]):
            return False, "One or more tasks are invalid"
        if self.process == Process.hierarchical and not (self.manager_llm or self.manager_agent):
            return False, "No manager agent or manager LLM set for hierarchical process"
        return True, ""

    def validate_manager_llm(self):
        available_models = llm_providers_and_models()
        if self.manager_llm and self.manager_llm not in available_models:
            self.manager_llm = None

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "process": self.process,
            "verbose": self.verbose,
            "memory": self.memory,
            "cache": self.cache,
            "planning": self.planning,
            "max_rpm": self.max_rpm,
            "manager_llm": self.manager_llm,
            "manager_agent": self.manager_agent.role if self.manager_agent else None,
            "agents": [agent.role for agent in self.agents],
            "tasks": [
                {
                    "description": task.description,
                    "agent": task.agent.role if task.agent else None,
                    "tools": [tool.name for tool in task.agent.tools] if task.agent else [],
                    "llm": task.agent.llm_provider_model if task.agent else None,
                    "async": task.async_execution
                }
                for task in self.tasks
            ]
        }

def create_crew_interface(available_agents, available_tasks):
    """Create a Gradio interface for crew management"""
    
    def update_crew(crew, name, description, process, verbose, memory, cache, planning, max_rpm,
                   manager_llm, manager_agent, selected_agents, selected_tasks):
        crew.name = name
        crew.description = description
        crew.process = Process[process]
        crew.verbose = verbose
        crew.memory = memory
        crew.cache = cache
        crew.planning = planning
        crew.max_rpm = max_rpm
        crew.manager_llm = None if manager_llm == "None" else manager_llm
        crew.manager_agent = next((agent for agent in available_agents if agent.role == manager_agent), None) if manager_agent != "None" else None
        crew.agents = [agent for agent in available_agents if agent.role in selected_agents]
        
        available_task_ids = [task.id for task in available_tasks if task.agent.id in [agent.id for agent in crew.agents]]
        crew.tasks = [task for task in available_tasks if task.id in selected_tasks and task.id in available_task_ids]
        
        db_utils.save_crew(crew)
        return gr.update(visible=False), gr.update(visible=True), crew.to_dict()

    def show_edit_form(crew):
        return gr.update(visible=True), gr.update(visible=False)

    def delete_current_crew(crew):
        crew.delete()
        return True

    def filter_tasks_by_agents(selected_agents):
        # Filter tasks to only show those belonging to selected agents
        available_task_ids = [
            task.id for task in available_tasks 
            if task.agent and task.agent.role in selected_agents
        ]
        return gr.update(choices=available_task_ids)

    with gr.Group():
        with gr.Row():
            view_area = gr.JSON(label="Crew Details", visible=True)
            
        with gr.Row():
            edit_btn = gr.Button("Edit")
            delete_btn = gr.Button("Delete", variant="stop")
            
        with gr.Group() as edit_form:
            name = gr.Textbox(label="Name")
            description = gr.Textbox(label="Description")
            process = gr.Radio(
                label="Process",
                choices=["sequential", "hierarchical"],
                value="sequential"
            )
            with gr.Row():
                verbose = gr.Checkbox(label="Verbose")
                memory = gr.Checkbox(label="Memory")
                cache = gr.Checkbox(label="Cache")
                planning = gr.Checkbox(label="Planning")
            max_rpm = gr.Number(label="Max req/min", value=1000)
            
            with gr.Group() as manager_group:
                manager_llm = gr.Dropdown(
                    label="Manager LLM",
                    choices=["None"] + llm_providers_and_models()
                )
                manager_agent = gr.Dropdown(
                    label="Manager Agent",
                    choices=["None"] + [agent.role for agent in available_agents]
                )

            agents_list = gr.Dropdown(
                label="Select Agents",
                choices=[agent.role for agent in available_agents],
                multiselect=True
            )
            
            tasks_list = gr.Dropdown(
                label="Select Tasks",
                choices=[task.id for task in available_tasks],
                multiselect=True
            )
            
            save_btn = gr.Button("Save", variant="primary")
            cancel_btn = gr.Button("Cancel")

            # Add dynamic task filtering based on selected agents
            agents_list.change(
                fn=filter_tasks_by_agents,
                inputs=[agents_list],
                outputs=[tasks_list]
            )

            # Add process change handler to enable/disable manager settings
            def update_manager_visibility(process_value):
                is_hierarchical = process_value == "hierarchical"
                return [
                    gr.update(interactive=is_hierarchical),
                    gr.update(interactive=is_hierarchical)
                ]

            process.change(
                fn=update_manager_visibility,
                inputs=[process],
                outputs=[manager_llm, manager_agent]
            )

    def initialize_form(crew):
        data = crew.to_dict()
        return [
            data["name"],
            data.get("description", ""),
            data["process"].value,
            data["verbose"],
            data["memory"],
            data["cache"],
            data["planning"],
            data["max_rpm"],
            data["manager_llm"] or "None",
            data["manager_agent"] or "None",
            data["agents"],
            [task["id"] for task in data["tasks"]]
        ]

    return {
        "view_area": view_area,
        "edit_form": edit_form,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "save_btn": save_btn,
        "cancel_btn": cancel_btn,
        "form_inputs": [name, description, process, verbose, memory, cache, planning,
                       max_rpm, manager_llm, manager_agent, agents_list, tasks_list]
    }