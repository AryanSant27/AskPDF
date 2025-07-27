from flask import Flask, request, jsonify, send_from_directory
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, JWTManager, jwt_required, get_jwt_identity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from bson.objectid import ObjectId
from PyPDF2 import PdfReader

load_dotenv()

app = Flask(__name__)

# Flask-JWT-Extended Configuration
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret")  # Change this in production!
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

# Sentence Transformer Model
print("Loading AI model (this may take a moment)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("AI model loaded.")

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("Warning: GEMINI_API_KEY not set in .env. LLM functionality will be limited.")
    gemini_model = None

def generate_embedding(text):
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
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

    if gemini_model:
        try:
            response = gemini_model.generate_content(full_prompt)
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

if __name__ == '__main__':
    app.run(debug=False)