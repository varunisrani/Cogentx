import gradio as gr
from my_task import MyTask
import db_utils

class PageTasks:
    def __init__(self):
        self.name = "Tasks"
        self.tasks = []
        self.crews = []
        self.load_data()

    def load_data(self):
        """Load all necessary data"""
        self.tasks = db_utils.load_tasks()
        self.crews = db_utils.load_crews()

    def create_task(self, crew=None):
        task = MyTask()
        self.tasks.append(task)
        task.edit = True
        db_utils.save_task(task)

        if crew:
            crew.tasks.append(task)
            db_utils.save_crew(crew)

        return self.update_tasks_display()

    def update_tasks_display(self):
        """Update the tasks display HTML"""
        if not self.tasks:
            return "No tasks defined yet."

        # Dictionary to track task assignment
        task_assignment = {task.id: [] for task in self.tasks}

        # Assign tasks to crews
        for crew in self.crews:
            for task in crew.tasks:
                task_assignment[task.id].append(crew.name)

        # Build HTML for tasks
        tasks_html = "<div class='tasks-container'>"
        
        # All Tasks section
        tasks_html += "<div class='task-section'><h3>All Tasks</h3>"
        for task in self.tasks:
            tasks_html += task.render_html()
        tasks_html += "</div>"

        # Unassigned Tasks section
        tasks_html += "<div class='task-section'><h3>Unassigned Tasks</h3>"
        unassigned_tasks = [task for task in self.tasks if not task_assignment[task.id]]
        for task in unassigned_tasks:
            tasks_html += task.render_html(key=f"{task.id}_unassigned")
        tasks_html += "</div>"

        # Tasks by crew
        for crew in self.crews:
            tasks_html += f"<div class='task-section'><h3>{crew.name}</h3>"
            assigned_tasks = [task for task in crew.tasks]
            for task in assigned_tasks:
                tasks_html += task.render_html(key=f"{task.id}_{crew.name}")
            tasks_html += "</div>"

        tasks_html += "</div>"
        return tasks_html

    def build_interface(self):
        with gr.Blocks() as interface:
            gr.Markdown(f"## {self.name}")
            
            tasks_display = gr.HTML(self.update_tasks_display())
            
            create_btn = gr.Button("Create task")
            create_btn.click(
                fn=self.create_task,
                inputs=None,
                outputs=tasks_display,
                api_name="create_task"
            )
            
            # Disable button when editing
            editing = any(task.edit for task in self.tasks)
            create_btn.interactive = not editing
            
        return interface

    def draw(self):
        """Draw the tasks interface"""
        with gr.Blocks() as interface:
            gr.Markdown(f"## {self.name}")
            
            tasks_display = gr.HTML(self.update_tasks_display())
            
            create_btn = gr.Button("Create task")
            create_btn.click(
                fn=self.create_task,
                inputs=None,
                outputs=tasks_display,
                api_name="create_task"
            )
            
            # Disable button when editing
            editing = any(task.edit for task in self.tasks)
            create_btn.interactive = not editing
            
        return interface
