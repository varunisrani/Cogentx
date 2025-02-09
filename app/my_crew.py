from crewai import Crew, Process
import gradio as gr
from utils import rnd_id
from datetime import datetime
from llms import llm_providers_and_models, create_llm
import db_utils

class MyCrew:
    def __init__(self, id=None, name=None, process=Process.sequential, verbose=True, created_at=None, memory=False, cache=True, planning=False, max_rpm=1000, manager_llm=None, manager_agent=None):
        self.id = id or f"C_{rnd_id()}"
        self.name = name or "New Crew"
        self.process = process
        self.verbose = verbose
        self.agents = []
        self.tasks = []
        self.edit = False
        self.memory = memory
        self.cache = cache
        self.planning = planning
        self.max_rpm = max_rpm
        self.manager_llm = manager_llm
        self.manager_agent = manager_agent
        self.created_at = created_at or datetime.now().isoformat()
        self.tasks_order = [task.id for task in self.tasks]

    def get_crewai_crew(self, *args, **kwargs) -> Crew:
        crewai_agents = [agent.get_crewai_agent() for agent in self.agents]

        # Create a dictionary to hold the Task objects
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

        if self.manager_llm:
            return Crew(
                agents=crewai_agents,
                tasks=crewai_tasks,
                cache=self.cache,
                process=self.process,
                max_rpm=self.max_rpm,
                verbose=self.verbose,
                manager_llm=create_llm(self.manager_llm),
                memory=self.memory,
                planning=self.planning,
                *args, **kwargs
            )
        elif self.manager_agent:
            return Crew(
                agents=crewai_agents,
                tasks=crewai_tasks,
                cache=self.cache,
                process=self.process,
                max_rpm=self.max_rpm,
                verbose=self.verbose,
                manager_agent=self.manager_agent.get_crewai_agent(),
                memory=self.memory,
                planning=self.planning,
                *args, **kwargs
            )
        cr = Crew(
            agents=crewai_agents,
            tasks=crewai_tasks,
            cache=self.cache,
            process=self.process,
            max_rpm=self.max_rpm,
            verbose=self.verbose,
            memory=self.memory,
            planning=self.planning,
            *args, **kwargs
        )
        return cr

    def delete(self):
        db_utils.delete_crew(self.id)
        return "Crew deleted successfully"

    def update_name(self, name):
        self.name = name
        db_utils.save_crew(self)
        return f"Name updated to: {name}"

    def update_process(self, process):
        self.process = Process(process)
        db_utils.save_crew(self)
        return f"Process updated to: {process}"

    def update_tasks(self, selected_tasks_ids):
        self.tasks = [task for task in self.tasks if task.id in selected_tasks_ids and task.agent.id in [agent.id for agent in self.agents]]
        self.tasks = sorted(self.tasks, key=lambda task: selected_tasks_ids.index(task.id))
        self.tasks_order = selected_tasks_ids
        db_utils.save_crew(self)
        return "Tasks updated successfully"

    def update_verbose(self, verbose):
        self.verbose = verbose
        db_utils.save_crew(self)
        return f"Verbose set to: {verbose}"

    def update_agents(self, selected_agents):
        self.agents = [agent for agent in self.agents if agent.role in selected_agents]
        db_utils.save_crew(self)
        return "Agents updated successfully"

    def update_manager_llm(self, selected_llm):
        self.manager_llm = selected_llm if selected_llm != "None" else None
        if self.manager_llm:
            self.manager_agent = None
        db_utils.save_crew(self)
        return f"Manager LLM updated to: {selected_llm}"

    def update_manager_agent(self, selected_agent_role):
        self.manager_agent = next((agent for agent in self.agents if agent.role == selected_agent_role), None) if selected_agent_role != "None" else None
        if self.manager_agent:
            self.manager_llm = None
        db_utils.save_crew(self)
        return f"Manager agent updated to: {selected_agent_role}"

    def update_memory(self, memory):
        self.memory = memory
        db_utils.save_crew(self)
        return f"Memory set to: {memory}"

    def update_max_rpm(self, max_rpm):
        self.max_rpm = max_rpm
        db_utils.save_crew(self)
        return f"Max RPM updated to: {max_rpm}"

    def update_cache(self, cache):
        self.cache = cache
        db_utils.save_crew(self)
        return f"Cache set to: {cache}"

    def update_planning(self, planning):
        self.planning = planning
        db_utils.save_crew(self)
        return f"Planning set to: {planning}"

    def is_valid(self):
        validation_messages = []
        if len(self.agents) == 0:
            validation_messages.append(f"Crew {self.name} has no agents")
        if len(self.tasks) == 0:
            validation_messages.append(f"Crew {self.name} has no tasks")
        for agent in self.agents:
            if not agent.is_valid():
                validation_messages.append(f"Invalid agent: {agent.role}")
        for task in self.tasks:
            if not task.is_valid():
                validation_messages.append(f"Invalid task: {task.id}")
        if self.process == Process.hierarchical and not (self.manager_llm or self.manager_agent):
            validation_messages.append(f"Crew {self.name} has no manager agent or manager llm set for hierarchical process")
        
        return len(validation_messages) == 0, "\n".join(validation_messages) if validation_messages else "Crew is valid"

    def validate_manager_llm(self):
        available_models = llm_providers_and_models()
        if self.manager_llm and self.manager_llm not in available_models:
            self.manager_llm = None

    def create_ui(self):
        with gr.Blocks() as interface:
            gr.Markdown(f"## Crew: {self.name}")
            
            with gr.Tab("Basic Settings"):
                name_input = gr.Textbox(label="Name", value=self.name)
                process_dropdown = gr.Dropdown(choices=[p.value for p in Process], value=self.process.value, label="Process")
                verbose_checkbox = gr.Checkbox(label="Verbose", value=self.verbose)
                memory_checkbox = gr.Checkbox(label="Memory", value=self.memory)
                cache_checkbox = gr.Checkbox(label="Cache", value=self.cache)
                planning_checkbox = gr.Checkbox(label="Planning", value=self.planning)
                max_rpm_slider = gr.Slider(minimum=1, maximum=10000, value=self.max_rpm, label="Max req/min")

            with gr.Tab("Agents & Tasks"):
                agents_dropdown = gr.Dropdown(choices=[agent.role for agent in self.agents], multiselect=True, value=[agent.role for agent in self.agents], label="Agents")
                tasks_dropdown = gr.Dropdown(choices=[task.id for task in self.tasks], multiselect=True, value=[task.id for task in self.tasks], label="Tasks")

            if self.process == Process.hierarchical:
                with gr.Tab("Manager Settings"):
                    manager_llm_dropdown = gr.Dropdown(choices=['None'] + llm_providers_and_models(), value=self.manager_llm or "None", label="Manager LLM")
                    manager_agent_dropdown = gr.Dropdown(choices=['None'] + [agent.role for agent in self.agents], value=self.manager_agent.role if self.manager_agent else "None", label="Manager Agent")

            save_btn = gr.Button("Save Changes")
            delete_btn = gr.Button("Delete Crew")
            status_text = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            name_input.change(self.update_name, inputs=[name_input], outputs=[status_text])
            process_dropdown.change(self.update_process, inputs=[process_dropdown], outputs=[status_text])
            verbose_checkbox.change(self.update_verbose, inputs=[verbose_checkbox], outputs=[status_text])
            memory_checkbox.change(self.update_memory, inputs=[memory_checkbox], outputs=[status_text])
            cache_checkbox.change(self.update_cache, inputs=[cache_checkbox], outputs=[status_text])
            planning_checkbox.change(self.update_planning, inputs=[planning_checkbox], outputs=[status_text])
            max_rpm_slider.change(self.update_max_rpm, inputs=[max_rpm_slider], outputs=[status_text])
            agents_dropdown.change(self.update_agents, inputs=[agents_dropdown], outputs=[status_text])
            tasks_dropdown.change(self.update_tasks, inputs=[tasks_dropdown], outputs=[status_text])

            if self.process == Process.hierarchical:
                manager_llm_dropdown.change(self.update_manager_llm, inputs=[manager_llm_dropdown], outputs=[status_text])
                manager_agent_dropdown.change(self.update_manager_agent, inputs=[manager_agent_dropdown], outputs=[status_text])

            delete_btn.click(self.delete, outputs=[status_text])

        return interface