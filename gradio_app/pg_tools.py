import gradio as gr
from utils import rnd_id
from my_tools import TOOL_CLASSES
import db_utils

class PageTools:
    def __init__(self):
        self.name = "Tools"
        self.available_tools = TOOL_CLASSES
        self.state = {
            'tools': [],
            'editing': False
        }
        self.load_data()

    def load_data(self):
        """Load tools from database"""
        self.state['tools'] = db_utils.load_tools()

    def create_tool(self, tool_name):
        """Create a new tool instance"""
        tool_class = self.available_tools[tool_name]
        tool_instance = tool_class(rnd_id())
        self.state['tools'].append(tool_instance)
        db_utils.save_tool(tool_instance)
        return self.get_tool_display_data()

    def remove_tool(self, tool_id):
        """Remove a tool"""
        self.state['tools'] = [tool for tool in self.state['tools'] if tool.tool_id != tool_id]
        db_utils.delete_tool(tool_id)
        return self.get_tool_display_data()

    def set_tool_parameter(self, tool_id, param_name, value):
        """Update tool parameter"""
        if value == "":
            value = None
        for tool in self.state['tools']:
            if tool.tool_id == tool_id:
                tool.set_parameters(**{param_name: value})
                db_utils.save_tool(tool)
                break
        return self.get_tool_display_data()

    def get_tool_display_name(self, tool):
        """Get display name for a tool"""
        first_param_name = tool.get_parameter_names()[0] if tool.get_parameter_names() else None
        first_param_value = tool.parameters.get(first_param_name, '') if first_param_name else ''
        return f"{tool.name} ({first_param_value if first_param_value else tool.tool_id})"

    def get_tool_display_data(self):
        """Get tool data for display"""
        return [{
            'id': tool.tool_id,
            'name': tool.name,
            'display_name': self.get_tool_display_name(tool),
            'description': tool.description,
            'parameters': tool.get_parameters(),
            'is_valid': tool.is_valid()
        } for tool in self.state['tools']]

    def create_interface(self):
        def handle_tool_creation(tool_name):
            """Handle tool creation"""
            if tool_name:
                self.create_tool(tool_name)
                return self.get_tool_display_data(), f"Tool '{tool_name}' created successfully"
            return self.get_tool_display_data(), "Please select a tool to create"

        def handle_tool_deletion(tool_id):
            """Handle tool deletion"""
            if tool_id:
                self.remove_tool(tool_id)
                return self.get_tool_display_data(), f"Tool {tool_id} deleted successfully"
            return self.get_tool_display_data(), "No tool selected for deletion"

        def handle_parameter_update(tool_id, param_name, value):
            """Handle parameter update"""
            if tool_id and param_name:
                self.set_tool_parameter(tool_id, param_name, value)
                return self.get_tool_display_data(), f"Updated parameter '{param_name}' for tool {tool_id}"
            return self.get_tool_display_data(), "Invalid parameter update request"

        with gr.Blocks() as tools_interface:
            gr.Markdown(f"## {self.name}")

            with gr.Row():
                # Available tools column
                with gr.Column(scale=1):
                    gr.Markdown("### Available Tools")
                    tool_selector = gr.Dropdown(
                        choices=list(self.available_tools.keys()),
                        label="Select Tool to Create",
                        interactive=True
                    )
                    create_btn = gr.Button("Create Selected Tool")

                # Enabled tools column
                with gr.Column(scale=2):
                    gr.Markdown("### Enabled Tools")
                    tools_list = gr.JSON(
                        label="Current Tools",
                        value=self.get_tool_display_data()
                    )
                    
                    # Tool configuration area
                    with gr.Accordion("Tool Configuration", open=False) as config_area:
                        tool_id = gr.Textbox(label="Tool ID")
                        param_name = gr.Textbox(label="Parameter Name")
                        param_value = gr.Textbox(label="Parameter Value")
                        update_btn = gr.Button("Update Parameter")
                        delete_btn = gr.Button("Delete Tool", variant="secondary")

            # Status message
            status_msg = gr.Textbox(label="Status", interactive=False)

            # Event handlers
            create_btn.click(
                fn=handle_tool_creation,
                inputs=[tool_selector],
                outputs=[tools_list, status_msg]
            )

            delete_btn.click(
                fn=handle_tool_deletion,
                inputs=[tool_id],
                outputs=[tools_list, status_msg]
            )

            update_btn.click(
                fn=handle_parameter_update,
                inputs=[tool_id, param_name, param_value],
                outputs=[tools_list, status_msg]
            )

            # Tool selection handler
            tools_list.change(
                fn=lambda x: gr.update(visible=True),
                outputs=[config_area]
            )

            # Initialize with data
            tools_list.value = self.get_tool_display_data()

            # Setup periodic refresh
            tools_interface.load(
                fn=self.get_tool_display_data,
                outputs=[tools_list]
            )

        return tools_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()