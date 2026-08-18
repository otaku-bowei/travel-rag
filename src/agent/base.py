from abc import ABC

from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent


def createAgent():
    create_react_agent()