# Agent Social Bench

Testing LLM based agent behaviour in social scenarios.

## Developer setup

### Environment variables

It is expected that there will be a `.env` file at the project root directory, to be 
read by `python-dotenv`. The `.env` file should have the following variables:
```
OPENAI_API_KEY=
GEMINI_API_KEY=
```

### Environment setup

The python environment is managed using `uv`. To install the required packages simply 
sync the environment by running:
```bash
uv sync
```

### Run scripts

Using `uv` you can run scripts using `uv run`, for example:
```bash
uv run scripts/main.py
```