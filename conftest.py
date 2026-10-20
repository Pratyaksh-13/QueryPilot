"""
conftest.py
-----------
Shared pytest configuration for QueryPilot.

Adds the project root to sys.path so that `src.*` imports resolve correctly
when running `pytest` from the project root without installing the package.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
