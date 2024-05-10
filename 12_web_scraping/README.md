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
cd 12_web_scraping
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
python 01_quickstart/00_load_documents.py
python 01_quickstart/01_retrieval_without_query_analysis.py
python 01_quickstart/02_query_analysis.py
python 01_quickstart/03_retrieval_with_query_analysis.py
python 02_how_to/02_handle_multiple_queries.py
# ...
```
