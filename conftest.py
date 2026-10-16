"""
conftest.py — shared pytest fixtures for QueryPilot.
"""
import sys
import os

# Ensure the project root is on sys.path so `src.*` imports work
# when running `pytest` from the project root.
sys.path.insert(0, os.path.dirname(__file__))
