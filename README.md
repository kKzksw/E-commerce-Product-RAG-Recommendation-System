# E-commerce Product Recommendation System

A mobile phone recommendation system based on **Agentic RAG + Streamlit**. Supports natural language input, automatically recognizes "recommend" or "compare" intent, and provides results by combining structured parameters and review evidence.

## Features Overview

- Natural language routing: Automatically detects `RECOMMEND` / `COMPARE`
- Multi-source retrieval fusion:
  - Structured retrieval (budget, brand, model, specification preferences)
  - Semantic review retrieval (sentence embeddings + similarity)
  - Description text relevance scoring
- Comparison mode: Outputs the winning model, key differences, and score breakdown
- Review insights: Automatically extracts pros/cons, common complaints, and sentiment summaries
- Streamlit visualization: Top 3 cards, technical details, evidence snippets

## Tech Stack

- Python
- Streamlit
- Pandas
- EdenAI (text generation + embeddings)

## Project Structure

```text
.
├── streamlit_app.py            # Main entry (Web UI)
├── requirements.txt            # Dependencies
├── data/
│   ├── mobile_reviews.csv      # Raw review data
│   └── sentence_embeddings.pkl # Sentence embedding cache (auto-generated on first run)
└── src/
    ├── agent/
    │   ├── router.py           # Query intent and constraint parsing
    │   ├── compare_tool.py     # Comparison scoring and winner logic
    │   └── explainer.py        # Result explanation generation
    ├── retriever/
    │   ├── structured.py       # Structured retrieval
    │   ├── multi_source.py     # Multi-source fusion retrieval
    │   ├── review_insights.py  # Review insights summarization
    │   └── evidence.py         # Evidence snippet extraction
    └── utils/
        ├── data.py             # Data loading and sentence embedding precomputation
        ├── edenai.py           # EdenAI API wrapper
        └── evaluation.py       # Routing/explanation evaluation tools
```

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root directory:

```env
EDENAI_API_KEY=your_edenai_api_key
```

### 4. Launch the app

```bash
streamlit run streamlit_app.py
```

After starting, open the local address in your browser (usually `http://localhost:8501`).

## Usage Examples

- Recommendation: `Recommend a phone under $600 with good battery life`
- Comparison: `Compare iPhone 14 vs Galaxy S24`
- Preference: `I want a latest Google phone with strong camera`

## Data Description

The main fields in `data/mobile_reviews.csv` include:

- Product info: `brand`, `model`, `price_usd`
- Overall rating: `rating`
- Dimension ratings: `battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`
- Review info: `review_text`, `review_date`, `source`

## Evaluation (Optional)

Run the built-in evaluation script:

```bash
python -m src.utils.evaluation
```

## FAQ

- Error `Missing EDENAI_API_KEY`: Check if `.env` exists and the key is valid.
- Slow on first launch: The system will precompute review sentence embeddings and cache them in `data/sentence_embeddings.pkl`.
- Semantic retrieval unavailable: Will automatically fall back to keyword retrieval, but results may be less accurate.

## Notes

- Do not commit your real API Key to the Git repository.
- To force rebuild the embedding cache, click `Recompute embeddings` in the sidebar.
