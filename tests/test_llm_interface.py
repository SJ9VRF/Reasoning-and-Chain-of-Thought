# tests/test_llm_interface.py

import unittest
from unittest.mock import patch, MagicMock
from src.llm_interface import LLMInterface

class TestLLMInterface(unittest.TestCase):
    def setUp(self):
        self.project_id = "test_project"
        self.location = "us-central1"
        self.model_name = "test-model"
        self.llm_interface = LLMInterface(
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name
        )

    @patch('vertexai.language_models.TextGenerationModel.from_pretrained')
    def test_initialization(self, mock_from_pretrained):
        # Ensure Vertex AI is initialized correctly
        mock_model_instance = MagicMock()
        mock_from_pretrained.return_value = mock_model_instance

        llm = LLMInterface(
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name
        )

        mock_from_pretrained.assert_called_with(self.model_name)
        self.assertEqual(llm.model, mock_model_instance)
        self.assertEqual(llm.parameters['temperature'], 0)
        self.assertEqual(llm.parameters['max_output_tokens'], 1024)
        self.assertEqual(llm.parameters['top_p'], 0.8)
        self.assertEqual(llm.parameters['top_k'], 40)

    @patch('vertexai.language_models.TextGenerationModel.from_pretrained')
    def test_call_llm_show_activity(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_model.predict.return_value.text = "Paris is the capital of France."
        mock_from_pretrained.return_value = mock_model

        with patch('builtins.print') as mock_print:
            response = self.llm_interface.call_llm("What is the capital of France?", show_activity=True)
            mock_model.predict.assert_called_with("What is the capital of France?", **self.llm_interface.parameters)
            mock_print.assert_any_call("The call to the LLM:\nWhat is the capital of France?\n")
            mock_print.assert_any_call("The response:")
            mock_print.assert_any_call("Paris is the capital of France.")
            self.assertEqual(response, "Paris is the capital of France.")

    @patch('vertexai.language_models.TextGenerationModel.from_pretrained')
    def test_call_llm_no_show_activity(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_model.predict.return_value.text = "Berlin is the capital of Germany."
        mock_from_pretrained.return_value = mock_model

        with patch('builtins.print') as mock_print:
            response = self.llm_interface.call_llm("What is the capital of Germany?", show_activity=False)
            mock_model.predict.assert_called_with("What is the capital of Germany?", **self.llm_interface.parameters)
            mock_print.assert_not_called()
            self.assertEqual(response, "Berlin is the capital of Germany.")

    @patch('vertexai.language_models.TextGenerationModel.from_pretrained')
    def test_call_llm_exception_handling(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("LLM Error")
        mock_from_pretrained.return_value = mock_model

        with self.assertRaises(Exception) as context:
            self.llm_interface.call_llm("What is the capital of Italy?", show_activity=False)
        
        self.assertTrue("LLM Error" in str(context.exception))

if __name__ == '__main__':
    unittest.main()

