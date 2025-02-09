import random
import string
import markdown as md
from datetime import datetime
import json

def rnd_id(length=8):
    """Generate a random ID string"""
    characters = string.ascii_letters + string.digits
    random_text = ''.join(random.choice(characters) for _ in range(length))
    return random_text

def escape_quotes(s):
    """Escape quotes in a string"""
    return s.replace('"', '\\"').replace("'", "\\'")

def format_result(result):
    """Format result for display, extracting relevant data from nested structures"""
    if isinstance(result, dict):
        if 'result' in result:
            if isinstance(result['result'], dict):
                if 'final_output' in result['result']:
                    return result['result']['final_output']
                elif 'raw' in result['result']:
                    return result['result']['raw']
                else:
                    return str(result['result'])
            elif hasattr(result['result'], 'raw'):
                return result['result'].raw
        return str(result)
    return str(result)

def generate_printable_view(crew_name, result, inputs, formatted_result, created_at=None):
    """Generate HTML for printable view"""
    if created_at is None:
        created_at = datetime.now().isoformat()
    created_at_str = datetime.fromisoformat(created_at).strftime('%Y-%m-%d %H:%M:%S')
    markdown_html = md.markdown(formatted_result)

    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>CrewAI-Studio result - {crew_name}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    padding: 2rem;
                    max-width: 1200px;
                    margin: auto;
                    line-height: 1.5;
                }}
                h1 {{
                    color: #2196F3;
                    margin-bottom: 2rem;
                }}
                .section {{
                    background: white;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .input-item {{
                    margin: 0.5rem 0;
                    padding: 0.5rem;
                    background: #f5f5f5;
                    border-radius: 4px;
                }}
                h2, h3, h4, h5, h6 {{
                    color: #1976D2;
                    margin-top: 1.5rem;
                }}
                code {{
                    background-color: #f5f5f5;
                    padding: 0.2rem 0.4rem;
                    border-radius: 4px;
                    font-family: 'SF Mono', 'Consolas', 'Liberation Mono', Menlo, Courier, monospace;
                }}
                pre code {{
                    display: block;
                    padding: 1rem;
                    overflow-x: auto;
                }}
                @media print {{
                    body {{
                        padding: 0;
                    }}
                    #printButton {{
                        display: none;
                    }}
                    .page-break {{
                        page-break-before: always;
                    }}
                    .section {{
                        box-shadow: none;
                        border: 1px solid #eee;
                    }}
                }}
            </style>
        </head>
        <body>
            <button id="printButton" onclick="window.print();" 
                    style="position: fixed; top: 20px; right: 20px; 
                           padding: 10px 20px; background: #2196F3; 
                           color: white; border: none; border-radius: 4px; 
                           cursor: pointer;">
                Print
            </button>

            <h1>CrewAI-Studio Result</h1>
            
            <div class="section">
                <h2>Crew Information</h2>
                <p><strong>Crew Name:</strong> {crew_name}</p>
                <p><strong>Created:</strong> {created_at_str}</p>
            </div>

            <div class="section">
                <h2>Inputs</h2>
                {''.join(f'<div class="input-item"><strong>{k}:</strong> {v}</div>' for k, v in inputs.items())}
            </div>

            <div class="page-break"></div>
            
            <div class="section">
                <h2>Results</h2>
                {markdown_html}
            </div>

            <script>
                // Add print keyboard shortcut
                document.addEventListener('keydown', function(e) {{
                    if (e.ctrlKey && e.key === 'p') {{
                        e.preventDefault();
                        window.print();
                    }}
                }});
            </script>
        </body>
    </html>
    """
    return html_content

def create_grid_style():
    """Create CSS for Gradio grid layouts"""
    return """
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        .grid-item {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """

def format_json(data):
    """Format JSON data for display"""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        return json.dumps(data, indent=2)
    except:
        return str(data)