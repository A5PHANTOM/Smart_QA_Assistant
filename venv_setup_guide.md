# Python Virtual Environment Setup Guide

This guide explains how to set up and manage a Python virtual environment for the Smart Document Q&A Assistant project using `venv`.

## 1. Create the Virtual Environment
Navigate to the root of your project directory and run the following command to create a virtual environment named `venv`:

**For Mac/Linux:**
```bash
python3 -m venv venv
```

**For Windows:**
```cmd
python -m venv venv
```

## 2. Activate the Virtual Environment
You need to activate the virtual environment before installing packages or running the application.

**For Mac/Linux:**
```bash
source venv/bin/activate
```

**For Windows:**
```cmd
venv\Scripts\activate
```
*(Once activated, you should see `(venv)` at the beginning of your terminal prompt.)*

## 3. Install Dependencies
After activation, install the required packages:
```bash
pip install -r smart_doc_qa/requirements.txt
```

## 4. Deactivate the Virtual Environment
When you are done working on the project, you can deactivate the virtual environment to return to your global Python environment:
```bash
deactivate
```
