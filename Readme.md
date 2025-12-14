# LangGraph Claim Decision Orchestrator

## Overview

This project uses LangGraph to orchestrate a workflow built with LangChain and OpenAI, to build an intelligent claim processing system. The design uses a state-based graph workflow where each step is handled by its own specialized node. Each node executes targeted tasks like summarizing documents, checking policies and keeping the overall state in sync. This design keeps the system modular, easy to maintain, and able to scale the  orchestration as needed. The LLM summarizations are heavily prompt based to look for sets of informations and the workflow sequence is regulated in specific sequence to get the result more deterministic. Hence the ReAct design principle is not used here.