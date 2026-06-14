import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the project directory to sys.path so we can import Agent
sys.path.append(r"E:\AskPDF Advanced Agentic Rag")

from Agent.translation_agent import detect_language_node, translate_response_node
from Agent.decomposer_agent import decompose_query_node, decomposer_approval_node
from Agent.web_scraper_agent import run_web_search_node, web_search_approval_node
from Agent.synthesis_agent import vector_search_node, decide_web_search_node, synthesis_node

class TestAgentNodes(unittest.TestCase):

    @patch('Agent.translation_agent.get_db_and_models')
    def test_detect_language_node_english(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_get_db_and_models.return_value = (None, None, None, mock_gemini)
        
        mock_res = MagicMock()
        mock_res.text = "English"
        mock_gemini.generate_content.return_value = mock_res
        
        state = {
            "original_query": "What is the capital of France?",
            "logs": []
        }
        
        res = detect_language_node(state)
        
        self.assertEqual(res["detected_language"], "English")
        self.assertEqual(res["english_query"], "What is the capital of France?")
        self.assertIn("Language detected: English", res["logs"])

    @patch('Agent.translation_agent.get_db_and_models')
    def test_detect_language_node_non_english(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_get_db_and_models.return_value = (None, None, None, mock_gemini)
        
        mock_lang_res = MagicMock()
        mock_lang_res.text = "French"
        mock_trans_res = MagicMock()
        mock_trans_res.text = "How are you?"
        
        mock_gemini.generate_content.side_effect = [mock_lang_res, mock_trans_res]
        
        state = {
            "original_query": "Comment ça va?",
            "logs": []
        }
        
        res = detect_language_node(state)
        
        self.assertEqual(res["detected_language"], "French")
        self.assertEqual(res["english_query"], "How are you?")
        self.assertIn("Language detected: French", res["logs"])

    @patch('Agent.decomposer_agent.get_db_and_models')
    def test_decompose_query_node(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_get_db_and_models.return_value = (None, None, None, mock_gemini)
        
        mock_res = MagicMock()
        mock_res.text = '["query one", "query two"]'
        mock_gemini.generate_content.return_value = mock_res
        
        state = {
            "english_query": "What is Section 5 and AAPL stock price?",
            "logs": []
        }
        
        res = decompose_query_node(state)
        
        self.assertEqual(res["decomposed_queries"], ["query one", "query two"])
        self.assertIn("Decomposed into sub-queries: ['query one', 'query two']", res["logs"])

    @patch('app.generate_embedding')
    @patch('Agent.synthesis_agent.get_db_and_models')
    def test_vector_search_node(self, mock_get_db_and_models, mock_generate_embedding):
        mock_db = MagicMock()
        mock_embeddings = MagicMock()
        mock_get_db_and_models.return_value = (mock_db, mock_embeddings, None, None)
        
        mock_generate_embedding.return_value = [0.1, 0.2]
        mock_embeddings.aggregate.return_value = [{"text": "Found context chunk"}]
        
        state = {
            "decomposed_queries": ["sub query"],
            "pdf_id": "60c72b2f9b1d8b1f4c8b4567",
            "logs": []
        }
        
        res = vector_search_node(state)
        
        self.assertEqual(res["vector_context"], "Found context chunk")
        self.assertIn("Retrieved 1 relevant chunks from PDF.", res["logs"])


    @patch('Agent.synthesis_agent.get_db_and_models')
    def test_decide_web_search_node(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_get_db_and_models.return_value = (None, None, None, mock_gemini)
        
        mock_res = MagicMock()
        mock_res.text = '{"web_needed": true, "queries": ["stock price"]}'
        mock_gemini.generate_content.return_value = mock_res
        
        state = {
            "english_query": "stock price info",
            "vector_context": "local text only",
            "logs": []
        }
        
        res = decide_web_search_node(state)
        
        self.assertEqual(res["web_queries"], ["stock price"])

    @patch('Agent.web_scraper_agent.requests.get')
    @patch('Agent.web_scraper_agent.DDGS')
    def test_run_web_search_node(self, mock_ddgs_class, mock_requests_get):
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = [{"href": "http://example.com", "title": "Example", "body": "Snippet"}]
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Scraped readable content</body></html>"
        mock_requests_get.return_value = mock_response
        
        state = {
            "web_queries": ["query"],
            "scraped_data": [],
            "logs": []
        }
        
        res = run_web_search_node(state)
        
        self.assertEqual(len(res["scraped_data"]), 1)
        self.assertEqual(res["scraped_data"][0]["title"], "Example")
        self.assertEqual(res["scraped_data"][0]["url"], "http://example.com")
        self.assertIn("Scraped readable content", res["scraped_data"][0]["text"])

    @patch('Agent.synthesis_agent.get_db_and_models')
    def test_synthesis_node(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_conversations = MagicMock()
        mock_get_db_and_models.return_value = (None, None, mock_conversations, mock_gemini)
        
        mock_conversations.find_one.return_value = {"history": []}
        mock_res = MagicMock()
        mock_res.text = "Final synthesized response text"
        mock_gemini.generate_content.return_value = mock_res
        
        state = {
            "english_query": "question",
            "vector_context": "pdf text",
            "scraped_data": [{"title": "Web", "url": "http://url", "text": "web text"}],
            "conversation_id": "60c72b2f9b1d8b1f4c8b4567",
            "logs": []
        }
        
        res = synthesis_node(state)
        
        self.assertEqual(res["answer"], "Final synthesized response text")
        self.assertIn("Final answer synthesized successfully.", res["logs"])

    @patch('Agent.translation_agent.get_db_and_models')
    def test_translate_response_node(self, mock_get_db_and_models):
        mock_gemini = MagicMock()
        mock_get_db_and_models.return_value = (None, None, None, mock_gemini)
        
        mock_res = MagicMock()
        mock_res.text = "Texto traducido"
        mock_gemini.generate_content.return_value = mock_res
        
        state = {
            "detected_language": "Spanish",
            "answer": "English text",
            "logs": []
        }
        
        res = translate_response_node(state)
        
        self.assertEqual(res["answer"], "Texto traducido")
        self.assertIn("Translating final response back to Spanish...", res["logs"])

if __name__ == "__main__":
    unittest.main()
