import gradio as gr
from db_utils import delete_result, load_results
from datetime import datetime
from utils import format_result, generate_printable_view
import json

class PageResults:
    def __init__(self):
        self.name = "Results"
        self.state = {
            'results': [],
            'crew_filter': [],
            'date_filter': None
        }
        self.load_data()

    def load_data(self):
        """Load results from database"""
        self.state['results'] = load_results()

    def filter_results(self, crew_filter, date_filter):
        """Filter results based on crew and date"""
        filtered_results = self.state['results']
        
        if crew_filter:
            filtered_results = [r for r in filtered_results if r.crew_name in crew_filter]
        
        if date_filter:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
            filtered_results = [r for r in filtered_results 
                              if datetime.fromisoformat(r.created_at).date() == filter_date]
        
        # Sort by creation time (newest first)
        return sorted(filtered_results,
                     key=lambda x: datetime.fromisoformat(x.created_at),
                     reverse=True)

    def create_interface(self):
        def update_results(crew_names, date_str):
            """Update results based on filters"""
            filtered = self.filter_results(crew_names, date_str)
            results_data = []
            
            for result in filtered:
                created_at = datetime.fromisoformat(result.created_at)
                results_data.append({
                    "id": result.id,
                    "crew_name": result.crew_name,
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": json.dumps(result.inputs),
                    "result": format_result(result.result),
                    "raw_result": json.dumps(result.result, indent=2) if result.result else ""
                })
            
            return gr.DataFrame(value=results_data)

        def handle_delete(result_id):
            """Handle result deletion"""
            delete_result(result_id)
            self.load_data()
            return "Result deleted successfully"

        def generate_printable(result_data):
            """Generate printable view for a result"""
            if not result_data:
                return None
            try:
                html_content = generate_printable_view(
                    result_data["crew_name"],
                    json.loads(result_data["raw_result"]),
                    json.loads(result_data["inputs"]),
                    result_data["result"],
                    result_data["created_at"]
                )
                return html_content
            except Exception as e:
                return f"Error generating printable view: {str(e)}"

        with gr.Blocks() as results_interface:
            gr.Markdown(f"## {self.name}")

            with gr.Row():
                # Filters
                crew_filter = gr.Dropdown(
                    choices=list(set(r.crew_name for r in self.state['results'])),
                    multiselect=True,
                    label="Filter by Crew"
                )
                date_filter = gr.Textbox(
                    label="Filter by Date (YYYY-MM-DD)",
                    placeholder="Enter date..."
                )

            # Results display
            results_table = gr.DataFrame(
                headers=["ID", "Crew", "Result", "Timestamp"],
                datatype=["str", "str", "str", "str"],
                label="Results"
            )

            with gr.Row():
                # Action buttons
                refresh_btn = gr.Button("Refresh")
                clear_filters_btn = gr.Button("Clear Filters")
            
            with gr.Row():
                result_status = gr.Textbox(label="Status", interactive=False)
                printable_iframe = gr.HTML(visible=False)

            # Event handlers
            refresh_btn.click(
                fn=update_results,
                inputs=[crew_filter, date_filter],
                outputs=[results_table]
            )

            clear_filters_btn.click(
                fn=lambda: (
                    None,  # Clear crew filter
                    "",    # Clear date filter
                    update_results(None, None)  # Update results
                ),
                outputs=[crew_filter, date_filter, results_table]
            )

            # Row selection handler
            results_table.select(
                fn=lambda evt: (
                    generate_printable(evt),
                    gr.update(visible=True)
                ),
                outputs=[printable_iframe, printable_iframe]
            )

            # Initialize with data
            results_table.value = update_results(None, None)

        return results_interface

    def draw(self):
        """Create and return the Gradio interface"""
        return self.create_interface()
