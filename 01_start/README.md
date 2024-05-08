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
cd 01_start
python -m venv env
```

Start it
```bash
cd 01_start
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\activate
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the app
The following commands allow you to run the application.

Serve
```bash
python quickstart/serve.py
```

Client
```bash
python quickstart/client.py
```

Guide steps
```bash
python quickstart/guide_steps/01_llm_invoke.py
python quickstart/guide_steps/02_chain.py
python quickstart/guide_steps/03_output_parser.py
python quickstart/guide_steps/04_embeddings.py
python quickstart/guide_steps/05_retriever.py
python quickstart/guide_steps/06_history_awareness.py
```
