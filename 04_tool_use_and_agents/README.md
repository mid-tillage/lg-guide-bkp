# Quickstart

## Table of Contents
- [Description](#description)
- [Virtual environment](#venv)
- [Installation](#installation)
- [Running the App](#running-the-app)

## Description

Initial guide of langchain

## venv
Generate your virtual environment

```bash
cd 04_tool_use_and_agents
python -m venv env
```

Start it
```bash
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\activate
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration
Set your api keys as environment variables!

## Running the app
The following commands allow you to run the application.

```bash
python 01_quickstart/00_create_a_tool.py
python 01_quickstart/01_chains.py
python 01_quickstart/02_invoking_the_tool.py
python 01_quickstart/03_agents.py
python 02_agents/00_create_agent.py
# ...
```
