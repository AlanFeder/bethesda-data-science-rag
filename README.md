# Bethesda Data Science RAG

Ask questions about the included meetup transcripts using local NumPy embeddings
and the OpenAI API. Use Python 3.11 or 3.12.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit.py
```

The `streamlit_chroma.py` entry point now launches the same NumPy app. ChromaDB
has no patched release for GHSA-36p7-vc44-83pf and GHSA-2wm9-hf6c-p5cr, so the
application, ingestion script, and notebook no longer import Chroma or LangChain.
Existing files in `video_data/` are used directly; no re-embedding is needed.
The old `chroma_db/` files are retained as historical data and are no longer read.

To ingest another YouTube transcript, set `OPENAI_API_KEY` in your environment
or an untracked `.env` file, then call `vector_storize_youtube_vid(video_id)`
from `get_yt_to_chroma.py`. Despite its historical name, that script now writes
`video_data/<video_id>.pkl` in the main app's format. The notebook uses the same
NumPy approach. `get_reduct_to_pklembed.py` remains available for Reduct exports
and requires a local Python installation with Tkinter for its overwrite dialog.

The previous notebook contained a hard-coded OpenAI key. The key and saved
outputs have been removed from the current notebook. Revoke or rotate the
previously committed key in the OpenAI dashboard: removing it from a file does
not remove it from Git history.

## Dependency maintenance

`requirements.in` contains the direct dependencies and security minimums;
`requirements.txt` pins the complete resolved dependency graph. NumPy stays on
1.26 and pandas on 2.x for compatibility with the included transcript pickles.
PyArrow stays on its patched 23.x line because PyArrow 26 requires NumPy 2 at
runtime. To regenerate the lockfile with [uv](https://docs.astral.sh/uv/):

```sh
uv pip compile requirements.in --python-version 3.11 --universal --upgrade --no-annotate --output-file requirements.txt
pip install -r requirements.txt
python -m pip check
python -P -m unittest discover -s tests -v
```

Use the `streamlit` executable above, rather than `python -m streamlit`: the
historical local file `streamlit.py` shares its name with the installed package.
The `-P` flag similarly keeps it from shadowing Streamlit during test discovery.
Tests mock API requests and cover both entry points, question answering, the
existing data, and YouTube ingestion. GitHub Actions runs them on Python 3.11
and 3.12 and audits all pinned dependencies with pip-audit. Dependabot groups
related version and security updates; a weekly workflow checks for new advisories.
