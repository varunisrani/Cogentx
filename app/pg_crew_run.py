import re
import gradio as gr
import threading
import ctypes
import queue
import time
import traceback
import os
from console_capture import ConsoleCapture
from db_utils import load_results, save_result
from utils import format_result, generate_printable_view, rnd_id

class PageCrewRun:
    def __init__(self):
        self.name = "Kickoff!"
        self.state = {
            'crew_thread': None,
            'result': None,
            'running': False,
            'message_queue': queue.Queue(),
            'selected_crew_name': None,
            'placeholders': {},
            'console_output': [],
            'last_update': time.time(),
            'console_expanded': True,
            'results': load_results()
        }

    @staticmethod
    def extract_placeholders(text):
        return re.findall(r'\{(.*?)\}', text)

    def get_placeholders_from_crew(self, crew):
        placeholders = set()
        attributes = ['description', 'expected_output', 'role', 'backstory', 'goal']
        
        for task in crew.tasks:
            placeholders.update(self.extract_placeholders(task.description))
            placeholders.update(self.extract_placeholders(task.expected_output))
        
        for agent in crew.agents:
            for attr in attributes[2:]:
                placeholders.update(self.extract_placeholders(getattr(agent, attr)))
        
        return placeholders

    def run_crew(self, crewai_crew, inputs, message_queue):
        if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not self.state.get('agentops_failed', False):
            import agentops
            agentops.start_session()
        try:
            result = crewai_crew.kickoff(inputs=inputs)
            message_queue.put({"result": result})
        except Exception as e:
            if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']) and not self.state.get('agentops_failed', False):                       
                agentops.end_session()
            stack_trace = traceback.format_exc()
            print(f"Error running crew: {str(e)}\n{stack_trace}")
            message_queue.put({"result": f"Error running crew: {str(e)}", "stack_trace": stack_trace})
        finally:
            if hasattr(self, 'console_capture'):
                self.console_capture.stop()

    def get_mycrew_by_name(self, crewname):
        return next((crew for crew in self.crews if crew.name == crewname), None)

    def handle_crew_selection(self, crew_name):
        selected_crew = self.get_mycrew_by_name(crew_name)
        self.state['selected_crew_name'] = crew_name
        
        if selected_crew:
            placeholders = self.get_placeholders_from_crew(selected_crew)
            placeholder_inputs = []
            for placeholder in placeholders:
                placeholder_inputs.append(
                    gr.Textbox(
                        label=placeholder,
                        interactive=not self.state['running']
                    )
                )
            return placeholder_inputs
        return []

    def run_crew_callback(self, selected_crew, *placeholder_values):
        if not selected_crew.is_valid():
            return "Selected crew is not valid. Please fix the issues."
            
        inputs = dict(zip(self.get_placeholders_from_crew(selected_crew), placeholder_values))
        self.state['result'] = None
        
        try:
            crew = selected_crew.get_crewai_crew(full_output=True)
        except Exception as e:
            return f"Error: {str(e)}"

        self.console_capture = ConsoleCapture()
        self.console_capture.start()
        self.state['console_output'] = []

        self.state['running'] = True
        self.state['crew_thread'] = threading.Thread(
            target=self.run_crew,
            kwargs={
                "crewai_crew": crew,
                "inputs": inputs,
                "message_queue": self.state['message_queue']
            }
        )
        self.state['crew_thread'].start()
        
        return "Crew started running..."

    def stop_crew_callback(self):
        if self.state['running']:
            self.force_stop_thread(self.state['crew_thread'])
            if hasattr(self, 'console_capture'):
                self.console_capture.stop()
            self.state['message_queue'].queue.clear()
            self.state['running'] = False
            self.state['crew_thread'] = None
            self.state['result'] = None
            return "Crew stopped successfully."
        return "No crew running."

    def serialize_result(self, result):
        if isinstance(result, dict):
            serialized = {}
            for key, value in result.items():
                if hasattr(value, 'raw'):
                    serialized[key] = {
                        'raw': value.raw,
                        'type': 'CrewOutput'
                    }
                elif hasattr(value, '__dict__'):
                    serialized[key] = {
                        'data': value.__dict__,
                        'type': value.__class__.__name__
                    }
                else:
                    serialized[key] = value
            return serialized
        return str(result)

    @staticmethod
    def force_stop_thread(thread):
        if thread:
            tid = ctypes.c_long(thread.ident)
            if tid:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
                return "Thread stopped successfully." if res > 0 else "Nonexistent thread id"
        return "No thread to stop"

    def create_interface(self):
        with gr.Blocks(title=self.name) as interface:
            gr.Markdown(f"# {self.name}")
            
            with gr.Row():
                crew_dropdown = gr.Dropdown(
                    choices=[crew.name for crew in self.crews] if hasattr(self, 'crews') else [],
                    label="Select crew to run",
                    interactive=not self.state['running']
                )
                
            placeholder_container = gr.Column()
            
            with gr.Row():
                run_button = gr.Button("Run crew!")
                stop_button = gr.Button("Stop crew!")
                
            console_output = gr.TextArea(
                label="Console Output",
                interactive=False
            )
            
            result_output = gr.Markdown(label="Result")

            crew_dropdown.change(
                fn=self.handle_crew_selection,
                inputs=[crew_dropdown],
                outputs=[placeholder_container]
            )
            
            run_button.click(
                fn=self.run_crew_callback,
                inputs=[crew_dropdown, placeholder_container],
                outputs=[result_output]
            )
            
            stop_button.click(
                fn=self.stop_crew_callback,
                inputs=[],
                outputs=[result_output]
            )

        return interface

    def launch(self):
        interface = self.create_interface()
        interface.launch()