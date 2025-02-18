import streamlit as st
import traceback
from tenacity import RetryError
from typing import Dict, Any

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Agent System",
    page_icon="🤖",
    layout="wide"
)

# Now import everything else
from groq_app import AIAgentSystem, log_message

def initialize_streamlit_state():
    """Initialize Streamlit session state for logging"""
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'log_container' not in st.session_state:
        st.session_state.log_container = st.empty()
    
    # Initialize AI Agent System
    if 'agent_system' not in st.session_state:
        st.session_state.agent_system = AIAgentSystem()
    
    # Get latest logs from file
    st.session_state.logs = st.session_state.agent_system.get_latest_logs()

def display_file_content(file_name: str, content: str):
    """Display file content with proper formatting and syntax highlighting."""
    st.markdown(f"### 📄 {file_name}")
    
    # Determine language for syntax highlighting
    if file_name.endswith('.py'):
        language = 'python'
    elif file_name.endswith('.env'):
        language = 'properties'
    elif file_name.endswith('.txt'):
        language = 'text'
    elif file_name.endswith('.json'):
        language = 'json'
    else:
        language = 'text'
    
    # Add a copy button and line numbers
    st.code(content, language=language, line_numbers=True)

def display_project_files(files: Dict[str, str], evaluation: Dict[str, Any]):
    """Display project files in an organized manner with evaluation results."""
    
    # Display Quality Control Results
    with st.expander("🔍 Quality Control Evaluation", expanded=True):
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅" if evaluation.get("meets_requirements") else "❌"
            st.metric("Requirements", status)
        
        with col2:
            status = "✅" if evaluation.get("code_quality") else "❌"
            st.metric("Code Quality", status)
            
        with col3:
            status = "✅" if evaluation.get("documentation") else "❌"
            st.metric("Documentation", status)
            
        with col4:
            status = "✅" if evaluation.get("technical_implementation") else "❌"
            st.metric("Implementation", status)
        
        # Display issues and recommendations if any
        if evaluation.get("issues"):
            st.warning("Issues Found:")
            for issue in evaluation["issues"]:
                st.markdown(f"- {issue}")
        
        if evaluation.get("recommendations"):
            st.info("Recommendations:")
            for rec in evaluation["recommendations"]:
                st.markdown(f"- {rec}")
    
    # Display Generated Files
    st.markdown("### 📁 Generated Project Files")
    
    # Create tabs for different file types
    tabs = st.tabs(["Core Files", "Configuration", "Dependencies"])
    
    with tabs[0]:  # Core Files
        if "app.py" in files:
            display_file_content("app.py", files["app.py"])
        if "main.py" in files:
            display_file_content("main.py", files["main.py"])
    
    with tabs[1]:  # Configuration
        if ".env" in files:
            st.warning("⚠️ Note: This is a template. Replace placeholder values with your actual credentials.")
            display_file_content(".env", files[".env"])
    
    with tabs[2]:  # Dependencies
        if "requirements.txt" in files:
            display_file_content("requirements.txt", files["requirements.txt"])
    
    # Add download section
    st.markdown("### 💾 Download Project Files")
    
    # Create columns for download buttons
    cols = st.columns(len(files))
    for idx, (file_name, content) in enumerate(files.items()):
        with cols[idx]:
            st.download_button(
                f"Download {file_name}",
                content,
                file_name=file_name,
                mime='text/plain',
                use_container_width=True
            )
    
    # Add project setup instructions
    with st.expander("🚀 Project Setup Instructions", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Download all files** using the buttons above
        2. **Create a new directory** for your project and place the files there
        3. **Set up your environment**:
           ```bash
           python -m venv venv
           source venv/bin/activate  # On Windows: venv\\Scripts\\activate
           pip install -r requirements.txt
           ```
        4. **Configure environment variables**:
           - Copy `.env.template` to `.env`
           - Fill in your actual API keys and credentials
        5. **Run the project**:
           ```bash
           python main.py
           ```
        """)

def display_logs():
    """Display logs in a formatted way."""
    if 'logs' in st.session_state:
        # Group logs by agent
        agent_logs = {}
        for log in st.session_state.logs:
            agent = log.get('agent', 'System')
            if agent not in agent_logs:
                agent_logs[agent] = []
            agent_logs[agent].append(log)
        
        # Display logs grouped by agent
        for agent, logs in agent_logs.items():
            with st.expander(f"📝 {agent} Logs", expanded=True):
                for log in logs:
                    timestamp = log.get('timestamp', '')
                    level = log.get('level', 'INFO')
                    message = log.get('message', '')
                    to_agent = log.get('to_agent')
                    
                    # Format the message
                    if to_agent:
                        header = f"{timestamp} | {agent} → {to_agent}"
                    else:
                        header = f"{timestamp} | {agent}"
                    
                    # Color-code based on log level
                    if level == 'ERROR':
                        st.error(f"{header}: {message}")
                    elif level == 'WARNING':
                        st.warning(f"{header}: {message}")
                    else:
                        st.info(f"{header}: {message}")
                    
                    # If there's code in the message, display it properly
                    if '```' in message:
                        code_blocks = message.split('```')
                        for i, block in enumerate(code_blocks):
                            if i % 2 == 1:  # This is a code block
                                st.code(block.strip(), language='python')

def main():
    # Initialize Streamlit state
    initialize_streamlit_state()
    
    # Add title and description with improved styling
    st.title("🤖 AI Agent System")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <h4>Welcome to the AI Agent System!</h4>
        <p>This system helps you create production-ready AI agent projects with:</p>
        <ul>
            <li>📝 Properly structured Python files</li>
            <li>⚙️ Environment configuration</li>
            <li>📦 Dependency management</li>
            <li>🔍 Quality control checks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input and output
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Project Requirements")
        user_prompt = st.text_area(
            "Describe your AI agent requirements",
            height=200,
            placeholder="Enter your requirements here...",
            help="Be specific about the functionality you need in your AI agent."
        )
        
        with st.expander("📝 Example Requirements", expanded=False):
            st.markdown("""
            ```
            Create an AI agent system that can:
            1. Process and analyze text documents
            2. Generate summary reports
            3. Integrate with external APIs
            4. Handle error cases gracefully
            5. Include comprehensive logging
            ```
            """)
        
        if st.button("🚀 Generate Project", type="primary", use_container_width=True):
            if not user_prompt:
                st.warning("⚠️ Please describe your requirements.")
                return
            
            with st.spinner("🔄 Generating your AI agent project..."):
                try:
                    result = st.session_state.agent_system.process_request(user_prompt)
                    st.session_state.result = result
                    st.success("✅ Project generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    log_message(traceback.format_exc(), level="error")
    
    with col2:
        if 'result' in st.session_state:
            result = st.session_state.result
            
            if "files" in result and "evaluation" in result:
                display_project_files(result["files"], result["evaluation"])
            
            # Display Logs
            with st.expander("📝 Generation Logs", expanded=True):
                display_logs()

if __name__ == "__main__":
    main() 
