# Backend

This project uses FastAPI on Python 3.10

## How to run

1. Create virtual environment, activate it, and install dependencies

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Run the server

    ```bash
    uvicorn main:app
    ```

    or

    ```bash
    python3 -m uvicorn main:app
    ```
