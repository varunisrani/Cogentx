
#!/bin/bash

# Create a virtual environment
python -m venv venv || { echo "Failed to create venv"; exit 1; }

# Activate the virtual environment
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Install requirements
pip install -r requirements.txt || { echo "Failed to install requirements"; exit 1; }

echo "Installation completed successfully."
