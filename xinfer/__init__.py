"""Top-level package for xinfer."""

__author__ = """Dickson Neoh"""
__email__ = "dickson.neoh@gmail.com"
__version__ = "0.0.1"

from .model_factory import create_model, list_models, register_models

# Ensure models are registered
register_models()
