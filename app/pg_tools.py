import gradio as gr
from utils import rnd_id
from my_tools import TOOL_CLASSES
import db_utils

class PageTools:
    def __init__(self):
        self.name = "Tools"
        self.available_tools = TOOL_CLASSES
        self.tools = []

    def create_tool(self, tool_name):
        tool_class = self.available_tools[tool_name]
        tool_instance = tool_class(rnd_id())
        self.tools.append(tool_instance)
        db_utils.save_tool(tool_instance)  # Save tool to database
        return gr.update(visible=True)

    def remove_tool(self, tool_id):
        self.tools = [tool for tool in self.tools if tool.tool_id != tool_id]
        db_utils.delete_tool(tool_id)
        return gr.update(visible=True)

    def set_tool_parameter(self, tool_id, param_name, value):
        if value == "":
            value = None
        for tool in self.tools:
            if tool.tool_id == tool_id:
                tool.set_parameters(**{param_name: value})
                db_utils.save_tool(tool)
                break

    def get_tool_display_name(self, tool):
        first_param_name = tool.get_parameter_names()[0] if tool.get_parameter_names() else None
        first_param_value = tool.parameters.get(first_param_name, '') if first_param_name else ''
        return f"{tool.name} ({first_param_value if first_param_value else tool.tool_id})"

    def create_tool_interface(self, tool):
        with gr.Group() as tool_group:
            gr.Markdown(f"### {self.get_tool_display_name(tool)}")
            gr.Markdown(tool.description)
            
            for param_name in tool.get_parameter_names():
                param_value = tool.parameters.get(param_name, "")
                placeholder = "Required" if tool.is_parameter_mandatory(param_name) else "Optional"
                gr.Textbox(
                    label=param_name,
                    value=param_value,
                    placeholder=placeholder,
                    elem_id=f"{tool.tool_id}_{param_name}",
                    interactive=True
                ).change(
                    fn=lambda x, tid=tool.tool_id, pname=param_name: self.set_tool_parameter(tid, pname, x)
                )
            
            gr.Button("Remove").click(
                fn=lambda: self.remove_tool(tool.tool_id),
                outputs=tool_group
            )

    def draw(self):
        with gr.Blocks() as interface:
            gr.Markdown(f"## {self.name}")
            
            with gr.Row():
                with gr.Column(scale=1):
                    for tool_name in self.available_tools.keys():
                        tool_class = self.available_tools[tool_name]
                        tool_instance = tool_class()
                        gr.Button(
                            tool_name
                        ).click(
                            fn=lambda n=tool_name: self.create_tool(n)
                        )
                
                with gr.Column(scale=3):
                    gr.Markdown("### Enabled Tools")
                    for tool in self.tools:
                        self.create_tool_interface(tool)
        
        return interface
