from flask import Flask, request, jsonify, send_from_directory
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity
import google.generativeai as genai
from bson.objectid import ObjectId
from pypdf import PdfReader


from datetime import timedelta

load_dotenv()

app = Flask(__name__)

# Flask-JWT-Extended Configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret")  # Change this in production!
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)
jwt = JWTManager(app)

# MongoDB Configuration
mongo_uri = os.getenv("MONGO_URI")
print("Attempting to connect to MongoDB...")
try:
    client = MongoClient(mongo_uri) # Set a 5-second timeout
    client.admin.command('ping')
    print("MongoDB connection successful.")
except Exception as e:
    print(f"FATAL: Could not connect to MongoDB. Error: {e}")
    exit()

db = client.askpdf_db  # You can change your database name here
users_collection = db.users
pdfs_collection = db.pdfs
embeddings_collection = db.embeddings
conversations_collection = db.conversations # New collection for chat history

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-3.5-flash')
    gemini_35_model = genai.GenerativeModel('gemini-3.5-flash')
    gemini_31_model = genai.GenerativeModel('gemini-3.1-flash-lite')
    gemini_25_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("Warning: GEMINI_API_KEY not set in .env. LLM functionality will be limited.")
    gemini_model = None
    gemini_35_model = None
    gemini_31_model = None
    gemini_25_model = None

def generate_embedding(text):
    if not GEMINI_API_KEY:
        print("Warning: generate_embedding called but GEMINI_API_KEY is not set.")
        return None
    import time
    max_retries = 3
    delay = 2.0
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "limit" in err_str:
                if attempt < max_retries - 1:
                    print(f"[Embedding Retry] Rate limit hit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2.0
                    continue
            print(f"Error generating embedding: {e}")
            return None


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/test_db')
def test_db():
    try:
        # Attempt to insert a dummy document to check connection
        db.test_collection.insert_one({"message": "MongoDB connected and tested!"})
        return jsonify({"status": "success", "message": "MongoDB connected and tested!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 400

    if users_collection.find_one({"username": username}):
        return jsonify({"msg": "Username already exists"}), 409

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({"username": username, "password": hashed_password})

    return jsonify({"msg": "User created successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    user = users_collection.find_one({"username": username})

    if user and check_password_hash(user['password'], password):
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)
    else:
        return jsonify({"msg": "Bad username or password"}), 401

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200

@app.route('/get_pdfs', methods=['GET'])
@jwt_required()
def get_pdfs():
    current_user = get_jwt_identity()
    user_pdfs = list(pdfs_collection.find({"user": current_user}, {"_id": 1, "filename": 1}))
    for pdf in user_pdfs:
        pdf['_id'] = str(pdf['_id']) # Convert ObjectId to string
    return jsonify(pdfs=user_pdfs), 200

@app.route('/upload_pdf', methods=['POST'])
@jwt_required()
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"msg": "No pdf_file part in the request"}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({"msg": "No selected file"}), 400

    if pdf_file and pdf_file.filename.endswith('.pdf'):
        try:
            reader = PdfReader(pdf_file)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"

            # Check if PDF already processed by this user
            existing_pdf = pdfs_collection.find_one({"filename": pdf_file.filename, "user": get_jwt_identity()})
            if existing_pdf:
                print(f"DEBUG: PDF '{pdf_file.filename}' already exists for this user.")
                return jsonify({"msg": "PDF already exists", "pdf_id": str(existing_pdf['_id'])}), 200

            pdf_id = pdfs_collection.insert_one({"filename": pdf_file.filename, "user": get_jwt_identity()}).inserted_id
            print(f"DEBUG: PDF metadata saved. PDF ID: {pdf_id}")

            chunks = chunk_text(text_content)
            print(f"DEBUG: Text chunked into {len(chunks)} pieces.")

            for i, chunk in enumerate(chunks):
                print(f"DEBUG: Generating embedding for chunk {i+1}/{len(chunks)}...")
                embedding = generate_embedding(chunk)
                if embedding:
                    embeddings_collection.insert_one({
                        "pdf_id": pdf_id,
                        "chunk_index": i,
                        "text": chunk,
                        "embedding": embedding
                    })
                print(f"DEBUG: Chunk {i+1} saved.")
            
            print("DEBUG: All chunks processed and saved successfully.")
            return jsonify({"msg": "PDF processed and embeddings stored", "pdf_id": str(pdf_id)}), 200
        except Exception as e:
            import traceback
            traceback.print_exc() # This will force a traceback to be printed
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"msg": "Invalid file type"}), 400

@app.route('/ask_pdf', methods=['POST'])
@jwt_required()
def ask_pdf():
    user_query = request.json.get('query', None)
    pdf_id = request.json.get('pdf_id', None)
    conversation_id = request.json.get('conversation_id', None)

    if not user_query or not pdf_id:
        return jsonify({"msg": "Missing query or pdf_id"}), 400

    try:
        pdf_object_id = ObjectId(pdf_id)
    except:
        return jsonify({"msg": "Invalid pdf_id format"}), 400

    # Generate embedding for the user query
    query_embedding = generate_embedding(user_query)
    if not query_embedding:
        return jsonify({"msg": "Failed to generate query embedding"}), 500

    # Perform vector search in MongoDB Atlas
    # This assumes you have created a vector search index named 'pdf_embeddings_index'
    # on your 'embeddings' collection with 'embedding' field as vector type and 'pdf_id' as filter.
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 5,
                "index": "pdf_embeddings_index" # Make sure this matches your index name
            }
        },
        {
            "$match": {"pdf_id": pdf_object_id}
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    retrieved_chunks = list(embeddings_collection.aggregate(pipeline))
    context = "\n".join([chunk['text'] for chunk in retrieved_chunks])

    # Retrieve conversation history
    chat_history = []
    if conversation_id:
        conversation = conversations_collection.find_one({"_id": ObjectId(conversation_id), "user": get_jwt_identity(), "pdf_id": pdf_object_id})
        if conversation:
            chat_history = conversation.get('history', [])

    # Construct prompt for Gemini
    full_prompt = f"""You are a helpful assistant that answers questions based on the provided context and conversation history.\n\nContext from PDF:\n{context}\n\nConversation History:\n{chat_history}\n\nUser Query: {user_query}\n\nAnswer:"""

    if gemini_31_model:
        try:
            response = gemini_31_model.generate_content(full_prompt)
            answer = response.text

            # Update conversation history
            if conversation_id:
                conversations_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {"$push": {"history": {"user": user_query, "model": answer}}}
                )
            else:
                new_conversation = {
                    "user": get_jwt_identity(),
                    "pdf_id": pdf_object_id,
                    "history": [{"user": user_query, "model": answer}]
                }
                conversation_id = conversations_collection.insert_one(new_conversation).inserted_id

            return jsonify({"answer": answer, "conversation_id": str(conversation_id)}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"msg": "Gemini model not configured."}), 500

# --- AGENTIC RAG ROUTES ---

@app.route('/agent/start', methods=['POST'])
@jwt_required()
def agent_start():
    from Agent import compiled_graph
    user_query = request.json.get('query', None)
    pdf_id = request.json.get('pdf_id', None)
    conversation_id = request.json.get('conversation_id', None)
    options = request.json.get('options', {
        "hitl_decomposer": True,
        "hitl_web": True
    })

    if not user_query or not pdf_id:
        return jsonify({"msg": "Missing query or pdf_id"}), 400

    try:
        pdf_object_id = ObjectId(pdf_id)
    except:
        return jsonify({"msg": "Invalid pdf_id format"}), 400

    current_user = get_jwt_identity()

    # Create new session in DB
    session_data = {
        "user": current_user,
        "pdf_id": pdf_object_id,
        "conversation_id": ObjectId(conversation_id) if conversation_id else None,
        "original_query": user_query,
        "status": "init",
        "current_step": "init",
        "logs": ["Session initialized."],
        "options": options
    }
    session_id = db.agent_sessions.insert_one(session_data).inserted_id

    # Create graph state
    initial_state = {
        "session_id": str(session_id),
        "user": current_user,
        "pdf_id": pdf_id,
        "conversation_id": conversation_id,
        "original_query": user_query,
        "english_query": "",
        "detected_language": "English",
        "current_step": "init",
        "decomposed_queries": [],
        "web_queries": [],
        "vector_context": "",
        "scraped_data": [],
        "logs": ["Starting agent workflow..."],
        "options": options,
        "answer": ""
    }

    config = {"configurable": {"thread_id": str(session_id)}}
    try:
        # Run graph until it hits an interrupt or completes
        compiled_graph.invoke(initial_state, config)
        
        # Sync state with DB
        status, current_step, state_values = save_agent_session(session_id, compiled_graph, config)
        
        if status == "completed":
            # Save to chat history
            conv_id = save_to_chat_history(conversation_id, current_user, pdf_id, user_query, state_values["answer"])
            # Update session's conversation_id in case it was a new conversation
            db.agent_sessions.update_one({"_id": session_id}, {"$set": {"conversation_id": ObjectId(conv_id)}})
            return jsonify({
                "status": "completed",
                "session_id": str(session_id),
                "answer": state_values["answer"],
                "logs": state_values["logs"],
                "conversation_id": str(conv_id)
            }), 200
        else:
            return jsonify({
                "status": "pending_approval",
                "session_id": str(session_id),
                "step": current_step,
                "data": {
                    "decomposed_queries": state_values.get("decomposed_queries", []),
                    "web_queries": state_values.get("web_queries", [])
                },
                "logs": state_values["logs"]
            }), 200
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        db.agent_sessions.update_one({"_id": session_id}, {"$set": {"status": "error", "logs": [f"FATAL ERROR: {str(e)}"]}})
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/agent/approve', methods=['POST'])
@jwt_required()
def agent_approve():
    from Agent import compiled_graph
    session_id = request.json.get('session_id', None)
    step = request.json.get('step', None)
    data = request.json.get('data', {})

    if not session_id or not step:
        return jsonify({"msg": "Missing session_id or step"}), 400

    session = db.agent_sessions.find_one({"_id": ObjectId(session_id)})
    if not session:
        return jsonify({"msg": "Session not found"}), 404

    config = {"configurable": {"thread_id": str(session_id)}}
    try:
        # Update graph state with user's approved changes
        if step == "query_decomposition":
            decomposed = data.get("decomposed_queries", [])
            compiled_graph.update_state(config, {"decomposed_queries": decomposed}, as_node="decompose_query")
        elif step == "web_search":
            web_queries = data.get("web_queries", [])
            compiled_graph.update_state(config, {"web_queries": web_queries}, as_node="decide_web_search")

        # Resume execution
        compiled_graph.invoke(None, config)

        # Sync state with DB
        status, current_step, state_values = save_agent_session(session_id, compiled_graph, config)

        if status == "completed":
            conv_id = save_to_chat_history(session.get("conversation_id"), session.get("user"), str(session.get("pdf_id")), session.get("original_query"), state_values["answer"])
            db.agent_sessions.update_one({"_id": ObjectId(session_id)}, {"$set": {"conversation_id": ObjectId(conv_id)}})
            return jsonify({
                "status": "completed",
                "session_id": str(session_id),
                "answer": state_values["answer"],
                "logs": state_values["logs"],
                "conversation_id": str(conv_id)
            }), 200
        else:
            return jsonify({
                "status": "pending_approval",
                "session_id": str(session_id),
                "step": current_step,
                "data": {
                    "decomposed_queries": state_values.get("decomposed_queries", []),
                    "web_queries": state_values.get("web_queries", [])
                },
                "logs": state_values["logs"]
            }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.agent_sessions.update_one({"_id": ObjectId(session_id)}, {"$set": {"status": "error", "logs": [f"FATAL ERROR: {str(e)}"]}})
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/agent/session/<session_id>', methods=['GET'])
@jwt_required()
def agent_session_status(session_id):
    try:
        session = db.agent_sessions.find_one({"_id": ObjectId(session_id)})
        if not session:
            return jsonify({"msg": "Session not found"}), 404
            
        # Convert ObjectIds to strings
        session["_id"] = str(session["_id"])
        session["pdf_id"] = str(session["pdf_id"])
        if session.get("conversation_id"):
            session["conversation_id"] = str(session["conversation_id"])
            
        return jsonify(session), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def save_agent_session(session_id, graph, config):
    state = graph.get_state(config)
    state_values = state.values
    
    next_steps = list(state.next)
    status = "completed" if not next_steps else "pending_approval"
    
    current_step = "synthesis"
    if "decomposer_approval_gate" in next_steps:
        current_step = "query_decomposition"
    elif "web_search_approval_gate" in next_steps:
        current_step = "web_search"
        
    db.agent_sessions.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "status": status,
                "current_step": current_step,
                "decomposed_queries": state_values.get("decomposed_queries", []),
                "web_queries": state_values.get("web_queries", []),
                "vector_context": state_values.get("vector_context", ""),
                "scraped_data": state_values.get("scraped_data", []),
                "logs": state_values.get("logs", []),
                "answer": state_values.get("answer", ""),
                "detected_language": state_values.get("detected_language", "English"),
                "english_query": state_values.get("english_query", "")
            }
        },
        upsert=True
    )
    return status, current_step, state_values


def save_to_chat_history(conversation_id, user, pdf_id, user_query, model_answer):
    if conversation_id:
        db.conversations.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$push": {"history": {"user": user_query, "model": model_answer}}}
        )
        return conversation_id
    else:
        new_conv = {
            "user": user,
            "pdf_id": ObjectId(pdf_id),
            "history": [{"user": user_query, "model": model_answer}]
        }
        inserted_id = db.conversations.insert_one(new_conv).inserted_id
        return inserted_id

if __name__ == '__main__':
    app.run(debug=False)
