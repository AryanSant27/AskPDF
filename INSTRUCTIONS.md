# AskPDF: Setup & Run Instructions

This guide provides step-by-step instructions to set up the project, configure your local environment (including MongoDB Atlas Vector Search), run the unit/integration tests, and start both the backend and frontend servers.

---

## 🚀 1. Quick Start (Windows PowerShell)

Follow these commands in sequence from your project root:

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt
pip install langchain-google-vertexai

# 3. Configure your .env file
# (Create a `.env` in the project root containing your MONGO_URI, GEMINI_API_KEY, and JWT_SECRET_KEY)

# 4. Start the server (hosts both backend and frontend)
python app.py
```

*The server will start at **`http://127.0.0.1:5000`**. Open this URL in your web browser.*

---

## 🛠️ 2. Detailed Configuration & Setup

### Environment Variables (`.env`)
Ensure you have a `.env` file in the root folder with the following variables:
```env
MONGO_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/askpdf_db"
GEMINI_API_KEY="AIzaSy..."
JWT_SECRET_KEY="your-secure-random-string"
```

### MongoDB Atlas Vector Search Index (CRITICAL!)
Because the embedding model has been upgraded to Google's API-based **`text-embedding-004`**, the vectors generated have **768 dimensions** (up from 384 previously). 

You **must** configure your MongoDB Vector Search index as follows:
1. In your MongoDB Atlas Dashboard, click on **Search** (under services).
2. Click **Create Search Index**.
3. Select the **JSON Editor** option.
4. Set the Index Name to exactly: **`pdf_embeddings_index`**
5. Set the Target Database to: **`askpdf_db`**
6. Set the Target Collection to: **`embeddings`**
7. Paste the following JSON index definition:
   ```json
   {
     "fields": [
       {
         "numDimensions": 768,
         "path": "embedding",
         "similarity": "cosine",
         "type": "vector"
       }
     ]
   }
   ```
8. Save and build the index.

> [!WARNING]
> If you have existing records in your database with the old 384-dimension vectors (from the SentenceTransformer model), please clear the `pdfs` and `embeddings` collections in Atlas first to prevent vector search dimension mismatch errors.

---

## 🧪 3. Running the Verification Suite

Run these tests in Windows PowerShell to verify code correctness:

### 1. Isolated Unit Tests
Verify all node execution units and mocks:
```powershell
.\venv\Scripts\python tests/test_units.py
```

### 2. Integration & HITL Simulator Tests
Verify the orchestrated workflow routes, API requests, and HITL gate interactions:
```powershell
.\venv\Scripts\python tests/test_integration.py
```

### 3. Ragas Pipeline Evaluation
Evaluate faithfulness and relevance metrics using your configured live Gemini keys:
```powershell
.\venv\Scripts\python tests/evaluate_ragas.py
```
*Calculates metrics and outputs a report directly to `tests/ragas_report.md`.*

---

## 🌐 4. Unified Client & Server Architecture
Starting the server via `python app.py` launches **both** the backend APIs and serves the static frontend:
- The backend serves REST API routes under `/register`, `/login`, `/upload_pdf`, and agentic execution checkpoints under `/agent/start` and `/agent/approve`.
- The frontend is located under `/static` and is served automatically. Opening `http://127.0.0.1:5000` delivers the interactive UI, logs viewer, and HITL gate controls.
