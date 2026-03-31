"""
main.py — Entry point for the Voice Agent POC.

Run locally:
    export GEMINI_API_KEY="your_key"
    python main.py

Run with Docker:
    docker compose up --build
"""

import yaml

from app.ui.gradio_app import create_app


def load_config(path: str = "configs/parameters.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    app_cfg = config.get("app", {})

    demo = create_app()
    import gradio as gr
    demo.launch(
        server_name=app_cfg.get("host", "0.0.0.0"),
        server_port=app_cfg.get("port", 7860),
        show_error=True,
        theme=gr.themes.Soft(),
    )
