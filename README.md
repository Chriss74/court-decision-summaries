# court-decision-summaries
Master Thesis: This project employs OpenAI's language model to create summaries from structured judicial decisions. 

# Prerequesites
In this section we provide the prerequesites in order to use this project.

## Python
This project was developed using python 3.11.4
Create a virtual environment, using the following commands:
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Environment
This application also requires setting up certain enviroment variables. Create a `.env` file like bellow
```
OPENAI_KEY=api_key
```