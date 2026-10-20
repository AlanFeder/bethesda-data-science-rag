"""Offline checks for the dependency upgrade and removal of ChromaDB."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ingestion", ROOT / "get_yt_to_chroma.py")
ingestion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingestion)


class AppTests(unittest.TestCase):
    def test_existing_transcripts_load_with_updated_pandas_numpy(self):
        for path in (ROOT / "video_data").glob("*.pkl"):
            with self.subTest(video=path.stem), path.open("rb") as source:
                raw, chunks, embeddings = pickle.load(source)
                self.assertEqual(len(chunks), embeddings.shape[0])
                self.assertEqual(1536, embeddings.shape[1])
                self.assertIn("text0", raw.columns)
                self.assertTrue(np.isfinite(embeddings).all())

    def test_both_entry_points_start_without_credentials(self):
        for filename in ("streamlit.py", "streamlit_chroma.py"):
            with self.subTest(entry=filename):
                app = AppTest.from_file(str(ROOT / filename)).run(timeout=30)
                self.assertEqual([], [error.message for error in app.exception])
                self.assertEqual(4, len(app.selectbox[0].options))

    def test_question_uses_local_embeddings_and_renders_answer(self):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0] + [0.0] * 1535)]
        )
        client.chat.completions.create.return_value = SimpleNamespace(
            model="test-model", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="Offline test answer"))],
        )
        with patch("openai.OpenAI", return_value=client), patch("tiktoken.get_encoding") as encoding:
            encoding.return_value.encode.return_value = [1]
            app = AppTest.from_file(str(ROOT / "streamlit.py")).run(timeout=30)
            app.text_input[0].set_value("test-placeholder")
            app.text_area[0].set_value("What is this talk about?")
            app.run(timeout=30)
            app.button[0].click().run(timeout=30)
        self.assertEqual([], [error.message for error in app.exception])
        self.assertIn("Offline test answer", [element.value for element in app.markdown])
        client.embeddings.create.assert_called_once()
        client.chat.completions.create.assert_called_once()

    def test_youtube_ingestion_keeps_embedding_order_and_replaces_metadata(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ingestion, "here", return_value=Path(directory)), \
                patch.object(ingestion, "YouTube") as video, \
                patch.object(ingestion, "yta") as transcript, \
                patch.object(ingestion, "OpenAI") as client:
            video.return_value.title = "Test talk"
            video.return_value.description = "Description"
            transcript.return_value.fetch.return_value.to_raw_data.return_value = [
                {"text": "First", "start": 0.0, "duration": 10.0},
                {"text": "Second", "start": 60.0, "duration": 10.0},
                {"text": "Third", "start": 120.0, "duration": 10.0},
            ]
            client.return_value.embeddings.create.return_value.data = [
                SimpleNamespace(index=2, embedding=[0.0, 0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0, 0.0]),
            ]
            ingestion.vector_storize_youtube_vid("test-video")
            ingestion.vector_storize_youtube_vid("test-video")
            with (Path(directory) / "video_data/test-video.pkl").open("rb") as source:
                raw, chunks, embeddings = pickle.load(source)
            self.assertEqual(3, len(raw))
            self.assertEqual(3, len(chunks))
            np.testing.assert_array_equal(embeddings, np.eye(3))
            metadata = pd.read_csv(Path(directory) / "metadata.csv")
            self.assertEqual(["test-video"], metadata.video_id.tolist())

    def test_notebook_code_compiles_without_saved_outputs(self):
        notebook = json.loads((ROOT / "explore.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell-{index}", "exec")
                self.assertEqual([], cell["outputs"])


if __name__ == "__main__":
    unittest.main()
