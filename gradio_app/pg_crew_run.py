import re
import threading
import ctypes
import queue
import time
import traceback
import os
import gradio as gr
from console_capture import ConsoleCapture
from db_utils import load_results, save_result
from utils import format_result, generate_printable_view, rnd_id
from result import Result

class PageCrewRun:
    def __init__(self):
        self.name = "Kickoff!"
        # Initialize state
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
            'crews': [],
            'results': load_results(),
            'saved_results': set(),
            'console_capture': None
        }
        self.maintain_session_state()

    def maintain_session_state(self):
        """Initialize or maintain session state"""
        # Load initial data if needed
        if not self.state['crews']:
            from db_utils import load_crews
            self.state['crews'] = load_crews()

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
        if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']):
            try:
                import agentops
                agentops.start_session()
            except Exception as e:
                print(f"Error initializing AgentOps: {str(e)}")
        try:
            result = crewai_crew.kickoff(inputs=inputs)
            message_queue.put({"result": result})
        except Exception as e:
            if (str(os.getenv('AGENTOPS_ENABLED')).lower() in ['true', '1']):
                agentops.end_session()
            stack_trace = traceback.format_exc()
            print(f"Error running crew: {str(e)}\n{stack_trace}")
            message_queue.put({"result": f"Error running crew: {str(e)}", "stack_trace": stack_trace})
        finally:
            if self.state['console_capture']:
                self.state['console_capture'].stop()

    def get_mycrew_by_name(self, crewname):
        return next((crew for crew in self.state['crews'] if crew.name == crewname), None)

    def serialize_result(self, result):
        """Serialize the crew result for database storage."""
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

    def force_stop_thread(self, thread):
        if thread:
            tid = ctypes.c_long(thread.ident)
            if tid:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
                return "Crew stopped successfully." if res != 0 else "Nonexistent thread id"
        return "No thread to stop"

    def create_interface(self):
        """Create the Gradio interface for crew running"""
        def update_crew_selection(crew_name):
            """Handle crew selection changes"""
            self.state['selected_crew_name'] = crew_name
            crew = self.get_mycrew_by_name(crew_name)
            if crew:
                placeholders = self.get_placeholders_from_crew(crew)
                self.state['placeholders'] = {p: "" for p in placeholders}
                return [
                    gr.update(visible=True, label=p, value="") 
                    if i < len(placeholders) else gr.update(visible=False)
                    for i in range(10)  # We pre-create 10 placeholder inputs
                ]
            return [gr.update(visible=False) for _ in range(10)]

        def run_crew_callback(crew_name, *placeholder_values):
            """Handle crew execution"""
            if not crew_name or self.state['running']:
                return "Please select a crew first" if not crew_name else "A crew is already running"
            
            crew = self.get_mycrew_by_name(crew_name)
            if not crew:
                return "Selected crew not found"
            
            if not crew.is_valid():
                return "Selected crew is not valid"

            # Update placeholder values
            placeholders = list(self.state['placeholders'].keys())
            self.state['placeholders'] = {
                p: v for p, v in zip(placeholders, placeholder_values) if v
            }
            
            try:
                crewai_crew = crew.get_crewai_crew(full_output=True)
            except Exception as e:
                return f"Error preparing crew: {str(e)}"

            # Start console capture
            self.state['console_capture'] = ConsoleCapture()
            self.state['console_capture'].start()
            self.state['console_output'] = []
            self.state['running'] = True
            self.state['message_queue'] = queue.Queue()

            # Start crew thread
            self.state['crew_thread'] = threading.Thread(
                target=self.run_crew,
                kwargs={
                    "crewai_crew": crewai_crew,
                    "inputs": self.state['placeholders'],
                    "message_queue": self.state['message_queue']
                }
            )
            self.state['crew_thread'].start()
            return "Crew started successfully"

        def stop_crew_callback():
            """Handle crew stopping"""
            if not self.state['running']:
                return "No crew is running"
            
            result = self.force_stop_thread(self.state['crew_thread'])
            if self.state['console_capture']:
                self.state['console_capture'].stop()
            
            self.state['message_queue'].queue.clear()
            self.state['running'] = False
            self.state['crew_thread'] = None
            self.state['result'] = None
            return result

        def update_output():
            """Update console output and check for crew completion"""
            if not self.state['running']:
                return "", None

            # Get new console output
            if self.state['console_capture']:
                new_output = self.state['console_capture'].get_output()
                if new_output:
                    self.state['console_output'].extend(new_output)

            # Check for completion
            try:
                message = self.state['message_queue'].get_nowait()
                self.state['result'] = message
                self.state['running'] = False
                if self.state['console_capture']:
                    self.state['console_capture'].stop()

                if isinstance(message, dict) and "result" in message:
                    result = message["result"]
                    # Save result
                    result_identifier = str(hash(str(result)))
                    if result_identifier not in self.state['saved_results']:
                        result_obj = Result(
                            id=f"R_{rnd_id()}",
                            crew_id=self.state['selected_crew_name'],
                            crew_name=self.state['selected_crew_name'],
                            inputs=self.state['placeholders'],
                            result=self.serialize_result(result)
                        )
                        save_result(result_obj)
                        self.state['results'].append(result_obj)
                        self.state['saved_results'].add(result_identifier)

                    return "\n".join(self.state['console_output']), format_result(result)
            except queue.Empty:
                pass

            return "\n".join(self.state['console_output']), None

        # Create the interface
        with gr.Blocks() as crew_run_interface:
            gr.Markdown(f"## {self.name}")

            # Crew selection
            crew_dropdown = gr.Dropdown(
                choices=[crew.name for crew in self.state['crews']],
                label="Select crew to run",
                interactive=not self.state['running']
            )

            # Create placeholder inputs container
            with gr.Group() as placeholder_group:
                placeholder_inputs = [
                    gr.Textbox(visible=False, label=f"Placeholder {i}")
                    for i in range(10)  # Pre-create some placeholder inputs
                ]

            # Control buttons
            with gr.Row():
                run_button = gr.Button("Run crew!", interactive=not self.state['running'])
                stop_button = gr.Button("Stop crew!", interactive=self.state['running'])

            # Output display
            console_output = gr.TextArea(label="Console Output", interactive=False)
            result_output = gr.Markdown(label="Result")

            # Event handlers
            crew_dropdown.change(
                fn=update_crew_selection,
                inputs=[crew_dropdown],
                outputs=placeholder_inputs
            )

            run_button.click(
                fn=run_crew_callback,
                inputs=[crew_dropdown] + placeholder_inputs,
                outputs=[result_output],
            ).then(
                fn=lambda: (gr.update(interactive=False), gr.update(interactive=True)),
                outputs=[run_button, stop_button]
            )

            stop_button.click(
                fn=stop_crew_callback,
                outputs=[result_output],
            ).then(
                fn=lambda: (gr.update(interactive=True), gr.update(interactive=False)),
                outputs=[run_button, stop_button]
            )

            # Setup periodic output updates
            crew_run_interface.load(
                fn=update_output,
                outputs=[console_output, result_output]
            )

        return crew_run_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()
        return self.create_interface()