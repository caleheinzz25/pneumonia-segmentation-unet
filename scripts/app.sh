#!/bin/bash
# Launch Gradio web app

set -e

echo "Starting web app..."
python -m app.gradio_app
echo "App stopped."
