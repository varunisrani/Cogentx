import gradio as gr
from db_utils import delete_result, load_results
from datetime import datetime
from utils import rnd_id, format_result, generate_printable_view

class PageResults:
    def __init__(self):
        self.name = "Results"
        self.results = load_results()
        
    def filter_results(self, crew_filter, date_filter):
        filtered_results = self.results
        
        if crew_filter:
            filtered_results = [r for r in filtered_results if r.crew_name in crew_filter]
            
        if date_filter:
            filter_date = datetime.combine(date_filter, datetime.min.time())
            filtered_results = [r for r in filtered_results if datetime.fromisoformat(r.created_at).date() == date_filter]
            
        return sorted(
            filtered_results,
            key=lambda x: datetime.fromisoformat(x.created_at),
            reverse=True
        )

    def delete_result_callback(self, result_id):
        delete_result(result_id)
        self.results = [r for r in self.results if r.id != result_id]
        return "Result deleted successfully"

    def print_view_callback(self, result):
        html_content = generate_printable_view(
            result.crew_name,
            result.result,
            result.inputs,
            format_result(result.result),
            result.created_at
        )
        return html_content

    def draw(self):
        with gr.Blocks() as interface:
            gr.Markdown(f"## {self.name}")
            
            with gr.Row():
                crew_filter = gr.Dropdown(
                    choices=list(set(r.crew_name for r in self.results)),
                    multiselect=True,
                    label="Filter by Crew"
                )
                date_filter = gr.Date(label="Filter by Date")

            results_container = gr.HTML()

            def update_results(crew_names, date):
                filtered = self.filter_results(crew_names, date)
                html = ""
                for result in filtered:
                    formatted_result = format_result(result.result)
                    html += f"""
                    <details>
                        <summary>{result.crew_name} - {datetime.fromisoformat(result.created_at).strftime('%Y-%m-%d %H:%M:%S')}</summary>
                        <h4>Inputs</h4>
                    """
                    for key, value in result.inputs.items():
                        html += f"<p><b>{key}:</b> {value}</p>"
                        
                    html += f"""
                        <h4>Result</h4>
                        <div class="tabs">
                            <div class="rendered">{formatted_result}</div>
                            <pre><code>{formatted_result}</code></pre>
                        </div>
                        <button onclick="deleteResult('{result.id}')">Delete</button>
                        <button onclick="printResult('{result.id}')">Print View</button>
                    </details>
                    """
                return html

            crew_filter.change(
                fn=update_results,
                inputs=[crew_filter, date_filter],
                outputs=results_container
            )
            date_filter.change(
                fn=update_results,
                inputs=[crew_filter, date_filter],
                outputs=results_container
            )

        return interface