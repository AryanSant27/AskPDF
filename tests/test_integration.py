import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch
from bson.objectid import ObjectId

# Add the project directory to sys.path so we can import app and Agent
sys.path.append(r"E:\AskPDF Advanced Agentic Rag")

# Mock MongoClient to prevent connection crash during app.py import
mock_client = MagicMock()
mock_admin = MagicMock()
mock_admin.command.return_value = {"ok": 1.0}
mock_client.admin = mock_admin
pymongo_patcher = patch('pymongo.mongo_client.MongoClient', return_value=mock_client)
pymongo_patcher.start()

import app
from Agent.orchestrator import compiled_graph

class TestAgenticRAGIntegration(unittest.TestCase):

    def setUp(self):
        # Configure Flask application for testing
        app.app.config['TESTING'] = True
        app.app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        self.client = app.app.test_client()
        
        # Create mock database objects
        self.mock_db = MagicMock()
        self.mock_users = MagicMock()
        self.mock_pdfs = MagicMock()
        self.mock_embeddings = MagicMock()
        self.mock_conversations = MagicMock()
        self.mock_sessions = MagicMock()
        
        # Link collections to the mock DB
        self.mock_db.users = self.mock_users
        self.mock_db.pdfs = self.mock_pdfs
        self.mock_db.embeddings = self.mock_embeddings
        self.mock_db.conversations = self.mock_conversations
        self.mock_db.agent_sessions = self.mock_sessions
        
        # Patch app database references
        app.db = self.mock_db
        app.users_collection = self.mock_users
        app.pdfs_collection = self.mock_pdfs
        app.embeddings_collection = self.mock_embeddings
        app.conversations_collection = self.mock_conversations

    @patch('app.gemini_model')
    @patch('app.generate_embedding')
    @patch('Agent.translation_agent.get_db_and_models')
    @patch('Agent.decomposer_agent.get_db_and_models')
    @patch('Agent.synthesis_agent.get_db_and_models')
    def test_full_hitl_agentic_flow(self, mock_synth_models, mock_decomp_models, mock_trans_models, mock_generate_embedding, mock_gemini_model):
        """
        Test the complete Agentic RAG workflow from start to final response,
        simulating the user interface HITL interactions.
        """
        # Configure the mock db and models returned in the orchestrators
        mock_models = (
            self.mock_db,
            self.mock_embeddings,
            self.mock_conversations,
            mock_gemini_model
        )
        mock_synth_models.return_value = mock_models
        mock_decomp_models.return_value = mock_models
        mock_trans_models.return_value = mock_models
        mock_generate_embedding.return_value = [0.1] * 768
        
        # Mock registration and login response

        self.mock_users.find_one.return_value = None  # for register (user does not exist)
        
        # --- 1. Register a test user ---
        register_payload = {"username": "testuser", "password": "password123"}
        response = self.client.post('/register', 
                                    data=json.dumps(register_payload), 
                                    content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertIn("User created successfully", response.json["msg"])
        
        # --- 2. Login to get JWT ---
        from werkzeug.security import generate_password_hash
        self.mock_users.find_one.return_value = {
            "username": "testuser",
            "password": generate_password_hash("password123")
        }
        
        login_payload = {"username": "testuser", "password": "password123"}
        response = self.client.post('/login', 
                                    data=json.dumps(login_payload), 
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        token = response.json["access_token"]
        self.assertIsNotNone(token)
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # --- 3. Start Agentic RAG Workflow ---
        mock_lang_res = MagicMock()
        mock_lang_res.text = "English"
        mock_decomp_res = MagicMock()
        mock_decomp_res.text = '["subquery 1", "subquery 2"]'
        
        mock_gemini_model.generate_content.side_effect = [
            mock_lang_res,   # Language detection
            mock_decomp_res  # Query decomposition
        ]
        
        # Mock session insertion
        session_id = ObjectId()
        self.mock_sessions.insert_one.return_value.inserted_id = session_id
        
        start_payload = {
            "query": "Compare section 5 and 6 and search the web for AAPL price",
            "pdf_id": str(ObjectId()),
            "options": {
                "hitl_decomposer": True,
                "hitl_web": True
            }
        }
        
        print("\n[Simulator] Calling /agent/start...")
        response = self.client.post('/agent/start',
                                    data=json.dumps(start_payload),
                                    content_type='application/json',
                                    headers=headers)
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json
        self.assertEqual(res_data["status"], "pending_approval")
        self.assertEqual(res_data["step"], "query_decomposition")
        self.assertEqual(res_data["data"]["decomposed_queries"], ["subquery 1", "subquery 2"])
        print(f"[Simulator] Response: {res_data['status']} - waiting at step: {res_data['step']}")
        
        # --- 4. Approve Decomposed Queries (HITL Gate 1) ---
        mock_web_decision = MagicMock()
        mock_web_decision.text = '{"web_needed": true, "queries": ["AAPL stock price"]}'
        mock_gemini_model.generate_content.side_effect = [mock_web_decision]
        
        self.mock_sessions.find_one.return_value = {
            "_id": session_id,
            "user": "testuser",
            "pdf_id": ObjectId(start_payload["pdf_id"]),
            "conversation_id": None,
            "original_query": start_payload["query"],
            "options": start_payload["options"]
        }
        
        self.mock_embeddings.aggregate.return_value = [{"text": "context 1"}]
        
        approve_decomposer_payload = {
            "session_id": str(session_id),
            "step": "query_decomposition",
            "data": {
                "decomposed_queries": ["approved subquery 1", "approved subquery 2"]
            }
        }
        
        print("[Simulator] Approving decomposer queries...")
        response = self.client.post('/agent/approve',
                                    data=json.dumps(approve_decomposer_payload),
                                    content_type='application/json',
                                    headers=headers)
        
        self.assertEqual(response.status_code, 200)
        res_data = response.json
        self.assertEqual(res_data["status"], "pending_approval")
        self.assertEqual(res_data["step"], "web_search")
        self.assertEqual(res_data["data"]["web_queries"], ["AAPL stock price"])
        print(f"[Simulator] Response: {res_data['status']} - waiting at step: {res_data['step']}")
        
        # --- 5. Approve Web Search (HITL Gate 2) ---
        mock_synthesis = MagicMock()
        mock_synthesis.text = "Here is the final answer with PDF section 5 info and AAPL stock price of $180."
        mock_gemini_model.generate_content.side_effect = [mock_synthesis]
        
        # Mock web search and scrape functions
        with patch('Agent.web_scraper_agent.DDGS') as mock_ddgs_class, \
             patch('Agent.web_scraper_agent.requests.get') as mock_requests_get:
            
            mock_ddgs_instance = MagicMock()
            mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance
            mock_ddgs_instance.text.return_value = [{"href": "http://finance.yahoo.com", "title": "Yahoo", "body": "AAPL Stock"}]
            
            mock_requests_get.return_value.status_code = 200
            mock_requests_get.return_value.text = "AAPL Stock price is $180."
            
            self.mock_conversations.insert_one.return_value.inserted_id = ObjectId()
            
            approve_web_payload = {
                "session_id": str(session_id),
                "step": "web_search",
                "data": {
                    "web_queries": ["approved AAPL stock price"]
                }
            }
            
            print("[Simulator] Approving web queries and finishing flow...")
            response = self.client.post('/agent/approve',
                                        data=json.dumps(approve_web_payload),
                                        content_type='application/json',
                                        headers=headers)
            
            self.assertEqual(response.status_code, 200)
            res_data = response.json
            self.assertEqual(res_data["status"], "completed")
            self.assertIn("Here is the final answer", res_data["answer"])
            self.assertIsNotNone(res_data["conversation_id"])
            print(f"[Simulator] Response: {res_data['status']} - Final Answer: {res_data['answer']}")
            
            print("\n[Simulator] Logs of the completed Agentic execution:")
            for idx, log in enumerate(res_data["logs"]):
                print(f"  {idx+1}. {log}")

if __name__ == "__main__":
    unittest.main()
