import gradio as gr
from crewai import Task
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
        self.context_from_async_tasks_ids = context_from_async_tasks_ids or None
        self.context_from_sync_tasks_ids = context_from_sync_tasks_ids or None
        self.created_at = created_at or datetime.now().isoformat()
        self.edit_mode = False

    def get_crewai_task(self, context_from_async_tasks=None, context_from_sync_tasks=None) -> Task:
        context = []
        if context_from_async_tasks:
            context.extend(context_from_async_tasks)
        if context_from_sync_tasks:
            context.extend(context_from_sync_tasks)
        
        if context:
            return Task(description=self.description, expected_output=self.expected_output, async_execution=self.async_execution, agent=self.agent.get_crewai_agent(), context=context)
        else:
            return Task(description=self.description, expected_output=self.expected_output, async_execution=self.async_execution, agent=self.agent.get_crewai_agent())

    def delete(self):
        delete_task(self.id)
        return gr.update(visible=False)

    def is_valid(self):
        if not self.agent:
            return False, "Task has no agent"
        agent_valid, agent_msg = self.agent.is_valid()
        if not agent_valid:
            return False, agent_msg
        return True, ""

    def save_changes(self, description, expected_output, agent_role, async_execution, context_async, context_sync):
        self.description = description
        self.expected_output = expected_output
        self.agent = next((a for a in self.available_agents if a.role == agent_role), None)
        self.async_execution = async_execution
        self.context_from_async_tasks_ids = context_async
        self.context_from_sync_tasks_ids = context_sync
        save_task(self)
        self.edit_mode = False
        return self.create_view_components()

    def toggle_edit(self, _):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            return self.create_edit_components()
        return self.create_view_components()

    def create_edit_components(self):
        with gr.Group():
            description = gr.Textbox(value=self.description, label="Description")
            expected_output = gr.Textbox(value=self.expected_output, label="Expected output")
            agent_dropdown = gr.Dropdown(
                choices=[agent.role for agent in self.available_agents],
                value=self.agent.role if self.agent else None,
                label="Agent"
            )
            async_checkbox = gr.Checkbox(value=self.async_execution, label="Async execution")
            context_async = gr.Dropdown(
                choices=[task.id for task in self.available_tasks if task.async_execution],
                value=self.context_from_async_tasks_ids,
                multiselect=True,
                label="Context from async tasks"
            )
            context_sync = gr.Dropdown(
                choices=[task.id for task in self.available_tasks if not task.async_execution],
                value=self.context_from_sync_tasks_ids,
                multiselect=True,
                label="Context from sync tasks"
            )
            save_btn = gr.Button("Save")
            save_btn.click(
                fn=self.save_changes,
                inputs=[description, expected_output, agent_dropdown, async_checkbox, context_async, context_sync],
                outputs=[gr.Group()]
            )

    def create_view_components(self):
        valid, msg = self.is_valid()
        title = f"({'❗' if not valid else ''}{self.agent.role if self.agent else 'unassigned'}) - {self.description}"
        
        with gr.Group():
            gr.Markdown(f"**Description:** {self.description}")
            gr.Markdown(f"**Expected output:** {self.expected_output}")
            gr.Markdown(f"**Agent:** {self.agent.role if self.agent else 'None'}")
            gr.Markdown(f"**Async execution:** {self.async_execution}")
            gr.Markdown(f"**Context from async tasks:** {', '.join([task.description[:120] for task in self.available_tasks if task.id in self.context_from_async_tasks_ids]) if self.context_from_async_tasks_ids else 'None'}")
            gr.Markdown(f"**Context from sync tasks:** {', '.join([task.description[:120] for task in self.available_tasks if task.id in self.context_from_sync_tasks_ids]) if self.context_from_sync_tasks_ids else 'None'}")
            
            with gr.Row():
                edit_btn = gr.Button("Edit")
                delete_btn = gr.Button("Delete")
            
            if not valid:
                gr.Markdown(f"⚠️ {msg}")
                
            edit_btn.click(fn=self.toggle_edit, inputs=[], outputs=[gr.Group()])
            delete_btn.click(fn=self.delete, inputs=[], outputs=[gr.Group()])

    def draw(self):
        """Draw the task interface"""
        if self.edit_mode:
            return self.create_edit_components()
        return self.create_view_components()