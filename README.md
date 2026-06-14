# AskPDF: Your AI-Powered Agentic Document Assistant

AskPDF is an advanced Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a rich conversational chat. This version introduces an **Agentic RAG multi-agent orchestration workflow** built on top of **LangGraph**, complete with Human-in-the-Loop (HITL) checkpoints.

---

## 🌟 Key Features

1. **Multi-Agent Orchestration (LangGraph)**:
   - **Translation Agent**: Detects incoming query languages and translates generated answers back to the user's native tongue.
   - **Query Decomposer Agent**: Automatically splits complex questions into distinct sub-queries.
   - **Synthesis Agent**: Aggregates local PDF vector contexts and queries external web resources to construct a thorough, comprehensive response.
2. **Human-in-the-Loop (HITL) Gates**:
   - Approve, modify, or extend decomposed sub-queries and web queries directly from the UI panel before execution.
3. **Multi-Model Load Balancing**:
   - Request loads are distributed across `gemini-3.5-flash` (for decomposition), `gemini-3.1-flash-lite` (for synthesis), and `gemini-2.5-flash` (for translations) to prevent rate limits on free-tier API keys.
4. **Google text-embedding-004 API**:
   - Modernized embeddings generate high-fidelity 768-dimensional context vectors.
5. **Modernized Premium UI/UX**:
   - Dynamic 3-column layout featuring interactive chat history, PDF management, real-time agent console log streams, and approval drawers.

---

## 🚀 How to Run the Project

Please refer to the detailed step-by-step setup instructions in **[INSTRUCTIONS.md](file:///E:/AskPDF%20Advanced%20Agentic%20Rag/INSTRUCTIONS.md)**.

### Quick Command:
```powershell
# Activate venv and run the unified server (starts both backend and frontend)
.\venv\Scripts\Activate.ps1
python app.py
```
*Go to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.*

---

## 🧪 Testing and Verification

A comprehensive testing suite is provided in the `tests/` folder:
- **Unit Tests**: Run node checks using `python tests/test_units.py`
- **Integration Tests**: Run the programmatic HITL execution simulator using `python tests/test_integration.py`
- **Evaluation**: Generate faithfulness and answer relevancy reports using Ragas and Gemini via `python tests/evaluate_ragas.py`
