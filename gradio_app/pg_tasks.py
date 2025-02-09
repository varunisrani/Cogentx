import gradio as gr
from my_task import MyTask
import db_utils

class PageTasks:
    def __init__(self):
        self.name = "Tasks"
        self.state = {
            'tasks': [],
            'crews': [],
            'editing': False
        }
        self.load_data()

    def load_data(self):
        """Load tasks and crews from database"""
        self.state['tasks'] = db_utils.load_tasks()
        self.state['crews'] = db_utils.load_crews()

    def create_task(self, crew=None):
        """Create a new task"""
        task = MyTask()
        self.state['tasks'].append(task)
        db_utils.save_task(task)

        if crew:
            crew.tasks.append(task)
            db_utils.save_crew(crew)

        return task

    def get_task_assignments(self):
        """Get dictionary mapping tasks to crews"""
        task_assignment = {task.id: [] for task in self.state['tasks']}
        for crew in self.state['crews']:
            for task in crew.tasks:
                task_assignment[task.id].append(crew.name)
        return task_assignment

    def create_interface(self):
        def refresh_view():
            """Refresh task data and view"""
            self.load_data()
            assignments = self.get_task_assignments()
            
            # Prepare data for all tasks
            all_tasks = [{
                'id': task.id,
                'description': task.description,
                'agent': task.agent.role if task.agent else 'None',
                'crews': ', '.join(assignments[task.id]) or 'Unassigned'
            } for task in self.state['tasks']]
            
            # Prepare data for unassigned tasks
            unassigned_tasks = [{
                'id': task.id,
                'description': task.description,
                'agent': task.agent.role if task.agent else 'None'
            } for task in self.state['tasks'] if not assignments[task.id]]
            
            # Prepare data for crew-specific tasks
            crew_tasks = {crew.name: [{
                'id': task.id,
                'description': task.description,
                'agent': task.agent.role if task.agent else 'None'
            } for task in crew.tasks] for crew in self.state['crews']}
            
            return (all_tasks, unassigned_tasks, crew_tasks)

        def handle_create_task(crew_name=None):
            """Handle task creation"""
            crew = next((c for c in self.state['crews'] if c.name == crew_name), None) if crew_name else None
            task = self.create_task(crew)
            return refresh_view()

        with gr.Blocks() as tasks_interface:
            gr.Markdown(f"## {self.name}")

            with gr.Tabs() as tabs:
                # All Tasks tab
                with gr.Tab("All Tasks"):
                    all_tasks_table = gr.DataFrame(
                        headers=["ID", "Description", "Agent", "Crews"],
                        datatype=["str", "str", "str", "str"],
                        label="All Tasks"
                    )
                    create_all_btn = gr.Button("Create Task")

                # Unassigned Tasks tab
                with gr.Tab("Unassigned Tasks"):
                    unassigned_tasks_table = gr.DataFrame(
                        headers=["ID", "Description", "Agent"],
                        datatype=["str", "str", "str"],
                        label="Unassigned Tasks"
                    )
                    create_unassigned_btn = gr.Button("Create Task")

                # Crew-specific tabs
                crew_tables = {}
                crew_buttons = {}
                for crew in self.state['crews']:
                    with gr.Tab(crew.name):
                        crew_tables[crew.name] = gr.DataFrame(
                            headers=["ID", "Description", "Agent"],
                            datatype=["str", "str", "str"],
                            label=f"{crew.name} Tasks"
                        )
                        crew_buttons[crew.name] = gr.Button(f"Create Task for {crew.name}")

            # Status message
            status_msg = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            create_all_btn.click(
                fn=lambda: handle_create_task(),
                outputs=[all_tasks_table, unassigned_tasks_table] + list(crew_tables.values())
            )

            create_unassigned_btn.click(
                fn=lambda: handle_create_task(),
                outputs=[all_tasks_table, unassigned_tasks_table] + list(crew_tables.values())
            )

            for crew_name, btn in crew_buttons.items():
                btn.click(
                    fn=lambda n=crew_name: handle_create_task(n),
                    outputs=[all_tasks_table, unassigned_tasks_table] + list(crew_tables.values())
                )

            # Initialize with data
            initial_data = refresh_view()
            all_tasks_table.value = initial_data[0]
            unassigned_tasks_table.value = initial_data[1]
            for crew_name, table in crew_tables.items():
                table.value = initial_data[2].get(crew_name, [])

            # Setup periodic refresh
            tasks_interface.load(
                fn=refresh_view,
                outputs=[all_tasks_table, unassigned_tasks_table] + list(crew_tables.values())
            )
        return tasks_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()
