# Backend Application README

## About Project
This is the backend application for the ATS platform. It is built using FastAPI and includes features for processing and analyzing data using Google's Generative AI models and PDFPlumber for extracting text from PDF files.

## Steps to Install for Windows, Linux, Mac

1. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Linux/Mac
    venv\Scripts\activate # On Windows
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set the `GOOGLE_API_KEY` environment variable:**
    Create a `.env` file in the `backend` directory and add your Google API key.
    ```env
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    ```
4. **Run the application:**
    ```bash
    uvicorn main:app --port=8000 --host=0.0.0.0
    ```
    The application will start on `http://127.0.0.1:8000` by default.

## Running Locally

To run the application locally, you need to start the backend server using Uvicorn. The main application file is `backend/main.py`.

### Changing the Host and Port

-   **Host:** The default host is `0.0.0.0`, which means the application will listen on all available network interfaces. You can change this in the `uvicorn.run` command in `backend/main.py` (line 19).
-   **Port:** The default port is `8000`. You can change this in the `uvicorn.run` command in `backend/main.py` (line 19).

For example, to run the application on `localhost` and port `5000`, you would modify the command in `backend/main.py` as follows:

```python
# backend/main.py
# ... (rest of the code)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
```

### CORS Configuration

The Cross-Origin Resource Sharing (CORS) configuration is set in `backend/main.py` (lines 12-19). If you need to allow requests from different origins (e.g., a different port or domain where your frontend is running), you can modify the `origins` list:

```python
# backend/main.py
origins = [
    "http://localhost:3000",
    "localhost:3000",
    "Your_Backend_Deployed_Link"
]
```

## Deployment Process on GCP
The backend is deployed on Google Cloud Platform (GCP) using Cloud Run. The Dockerfile is used to build the container image, which is then deployed to Cloud Run.

## Deployed Links
Frontend: https://heroic-ats-frontend.vercel.app/

## Follow for more:
Linkedin: https://www.linkedin.com/in/wankhede-gaurav/ 
Portfolio: https://gaurav-wankhede.vercel.app/
Instagram: https://www.instagram.com/_gaurav_wankhede_/
