"""
Tests for create_app() — catches Gradio API mismatches at test time rather than at Docker startup.

Regression targets (Gradio 6.0 breaking changes):
  - `theme` arg moved from gr.Blocks() to launch()
  - gr.Chatbot() no longer accepts `type` kwarg
"""

import gradio as gr
import pytest

from app.ui.gradio_app import create_app


class TestCreateApp:
    def test_create_app_does_not_raise(self):
        """create_app() must construct the Gradio Blocks without any exception."""
        demo = create_app()
        assert demo is not None

    def test_returns_gradio_blocks(self):
        demo = create_app()
        assert isinstance(demo, gr.Blocks)

    def test_theme_not_in_blocks_constructor(self):
        """Gradio 6.0: theme must be passed to launch(), not Blocks().
        If this test fails, remove theme= from gr.Blocks() in create_app().
        """
        import inspect, ast, textwrap
        import app.ui.gradio_app as module
        src = inspect.getsource(module.create_app)
        # Check that gr.Blocks( does not contain 'theme=' as an argument
        # Simple string check is sufficient — we're not parsing full AST
        assert "gr.Blocks(" in src
        blocks_call_line = next(l for l in src.splitlines() if "gr.Blocks(" in l)
        assert "theme=" not in blocks_call_line, (
            "Gradio 6.0 moved `theme` from gr.Blocks() to launch(). "
            "Remove theme= from gr.Blocks() and pass it in demo.launch() instead."
        )
