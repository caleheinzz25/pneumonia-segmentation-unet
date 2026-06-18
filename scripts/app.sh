#!/bin/bash
# Launch Gradio web app

set -e

echo "Starting web app..."
uv run python -m app.gradio_app
echo "App stopped."
