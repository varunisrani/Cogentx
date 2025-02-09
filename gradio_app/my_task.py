from crewai import Task
import gradio as gr
from utils import rnd_id
from db_utils import save_task, delete_task
from datetime import datetime

class MyTask:
    def __init__(self, id=None, description=None, expected_output=None, agent=None, async_execution=None, created_at=None, context_from_async_tasks_ids=None, context_from_sync_tasks_ids=None, **kwargs):
        self.id = id or "T_" + rnd_id()
        self.description = description or "Identify the next big trend in AI. Focus on identifying pros and cons and the overall narrative."
        self.expected_output = expected_output or "A comprehensive 3 paragraphs long report on the latest AI trends."
        self.agent = agent
        self.async_execution = async_execution or False
        self.context_from_async_tasks_ids = context_from_async_tasks_ids or []
        self.context_from_sync_tasks_ids = context_from_sync_tasks_ids or []
        self.created_at = created_at or datetime.now().isoformat()

    def get_crewai_task(self, context_from_async_tasks=None, context_from_sync_tasks=None) -> Task:
        context = []
        if context_from_async_tasks:
            context.extend(context_from_async_tasks)
        if context_from_sync_tasks:
            context.extend(context_from_sync_tasks)
        
        task_params = {
            "description": self.description,
            "expected_output": self.expected_output,
            "async_execution": self.async_execution,
            "agent": self.agent.get_crewai_agent()
        }
        
        if context:
            task_params["context"] = context
            
        return Task(**task_params)

    def delete(self):
        delete_task(self.id)
        return True

    def is_valid(self):
        if not self.agent:
            return False, "Task has no agent"
        if not self.agent.is_valid():
            return False, "Agent is not valid"
        return True, ""

    def to_dict(self):
        return {
            "description": self.description,
            "expected_output": self.expected_output,
            "agent": self.agent.role if self.agent else None,
            "async_execution": self.async_execution,
            "context_from_async_tasks": [
                next((task.description[:120] for task in available_tasks if task.id == task_id), None)
                for task_id in self.context_from_async_tasks_ids
            ] if self.context_from_async_tasks_ids else [],
            "context_from_sync_tasks": [
                next((task.description[:120] for task in available_tasks if task.id == task_id), None)
                for task_id in self.context_from_sync_tasks_ids
            ] if self.context_from_sync_tasks_ids else []
        }

def create_task_interface(available_agents=None, available_tasks=None):
    """Create a Gradio interface for task management"""
    
    def update_task(task, description, expected_output, agent_role, async_execution,
                   async_context_tasks, sync_context_tasks):
        task.description = description
        task.expected_output = expected_output
        task.agent = next((agent for agent in available_agents if agent.role == agent_role), None)
        task.async_execution = async_execution
        task.context_from_async_tasks_ids = async_context_tasks
        task.context_from_sync_tasks_ids = sync_context_tasks
        save_task(task)
        return gr.update(visible=False), gr.update(visible=True), task.to_dict()

    def show_edit_form(task):
        return gr.update(visible=True), gr.update(visible=False)

    def delete_current_task(task):
        task.delete()
        return True

    with gr.Group():
        with gr.Row():
            view_area = gr.JSON(label="Task Details", visible=True)
            
        with gr.Row():
            edit_btn = gr.Button("Edit")
            delete_btn = gr.Button("Delete", variant="stop")
            
        with gr.Group() as edit_form:
            description = gr.Textbox(
                label="Description",
                lines=3,
                placeholder="Describe the task..."
            )
            expected_output = gr.Textbox(
                label="Expected Output",
                lines=2,
                placeholder="Describe the expected output..."
            )
            agent_select = gr.Dropdown(
                label="Agent",
                choices=[agent.role for agent in (available_agents or [])]
            )
            with gr.Row():
                async_execution = gr.Checkbox(label="Async Execution")
            
            # Context task selection
            async_context = gr.Dropdown(
                label="Context from Async Tasks",
                choices=[task.id for task in (available_tasks or []) if task.async_execution],
                multiselect=True
            )
            sync_context = gr.Dropdown(
                label="Context from Sync Tasks",
                choices=[task.id for task in (available_tasks or []) if not task.async_execution],
                multiselect=True
            )
            
            save_btn = gr.Button("Save", variant="primary")
            cancel_btn = gr.Button("Cancel")

    def initialize_form(task):
        data = task.to_dict()
        return [
            data["description"],
            data["expected_output"],
            data["agent"],
            data["async_execution"],
            data["context_from_async_tasks"],
            data["context_from_sync_tasks"]
        ]

    return {
        "view_area": view_area,
        "edit_form": edit_form,
        "edit_btn": edit_btn,
        "delete_btn": delete_btn,
        "save_btn": save_btn,
        "cancel_btn": cancel_btn,
        "form_inputs": [description, expected_output, agent_select,
                       async_execution, async_context, sync_context]
    }

def format_task_description(task_id, available_tasks):
    """Format task description for dropdown display"""
    task = next((t for t in available_tasks if t.id == task_id), None)
    return task.description[:120] if task else "Unknown Task"