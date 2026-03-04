# ✈️ Travel Planner — Agentic Chatbot (LangGraph + LangChain)

A **travel-planner chatbot** that orchestrates an **agentic workflow** using **LangGraph / LangChain** to generate curated travel plans and a finalized itinerary.

You tell the bot what kind of vacation you want (e.g., _“Warm beach vacation in December with good nightlife”_).  
It will:

1. Convert your request into a **search-ready question**
2. Use **Tavily Search** to pull fresh context from the web
3. Ask an LLM to propose **1–5 candidate travel plans** in a strict JSON format
4. Put a **human-in-the-loop** step in the middle:
   - **Approve** one of the plans to generate a detailed itinerary
   - Or **reject** and provide feedback to refine and regenerate options

A simple **Gradio chat UI** is included for interactive use.

---

## What this project does

### Output you get

- **Multiple plan options (1–5)**, each containing:

  - Destination
  - Ideal number of days
  - Why it’s worth visiting
  - Highlights & activities
  - Who it suits

- After approval, a **final itinerary** generated from the chosen plan and your original query.

> Note: Your description mentions hotel pricing and stay costs. The current notebook’s plan schema focuses on high-level itinerary planning fields (Destination / Days / Highlights / etc.). If you want **hotel cost estimates** to be guaranteed, you can extend the plan schema and/or add a dedicated “costing” tool/node.

---

## Tech stack

- **LangGraph**: stateful, multi-step agent workflow (graph execution + checkpointing)
- **LangChain**: prompt templates and tool integration
- **Tavily Search**: web context retrieval for travel planning
- **LLMs**
  - `ChatGroq` (configured with `model_name = "openai/gpt-oss-120b"`) for planning + itinerary generation
  - `ChatOpenAI` (`gpt-3.5-turbo-0125`) is instantiated in the notebook (available if you want to switch models)
- **Gradio**: chat-based UI demo
- **Python / Jupyter Notebook**: implementation lives in `Travel_Planner.ipynb`

---

## Project workflow (LangGraph)

The graph is built with a typed `State` object and the following nodes:

1. **initiateTravelPlan**

   - Creates a search-friendly question from the user query
   - Calls Tavily Search (max 3 results)
   - Stores combined text in `tavilyResponse`

2. **suggestTravelPlan**

   - Uses Tavily context to generate **1–5 travel plan options**
   - Enforces **STRICT JSON** output
   - Parses JSON into a normalized dictionary: `{"Plan1": {...}, "Plan2": {...}}`

3. **humanApproval** (Human-in-the-loop gate)

   - If approved: requires a `selected_plan`
   - If rejected: requires `feedback`
   - If neither provided yet: waits (router keeps it pending)

4. **applyFeedback**

   - If rejected, appends feedback into the query
   - Loops back to `initiateTravelPlan` → search again → suggest again

5. **showSelectedPlan**
   - Once approved, formats the selected plan as context
   - Generates a detailed itinerary using the LLM
   - Writes final output to `response`

### Routing logic

A conditional router decides the next step after `humanApproval`:

- `accept` → `showSelectedPlan` → END
- `revise` → `applyFeedback` → back to `initiateTravelPlan`
- `pending` → `humanApproval` (wait for user input)

### Memory / checkpointing

The workflow uses `MemorySaver()` as a checkpointer so the session can be resumed reliably using a `thread_id`.

---

## Repository contents

This project is currently implemented as a single notebook:

- `Travel_Planner.ipynb` — end-to-end implementation:
  - Graph + state definition
  - Tavily retrieval
  - JSON plan generation + parsing helpers
  - “pause & resume” runner functions
  - Gradio chat UI wrapper

The notebook also installs dependencies via:

```bash
pip install -qU -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, create one from your environment (example below).

---

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies

If you have `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you **don’t** have one yet, a starting point based on the notebook imports would include:

- langgraph
- langchain-core
- langchain-community
- langchain-openai
- langchain-groq
- tavily-python (or relevant Tavily package)
- python-dotenv
- gradio
- ipython

You can generate a requirements file with:

```bash
pip freeze > requirements.txt
```

---

## Environment variables

The notebook uses `python-dotenv` and expects the following keys (usually stored in a `.env` file):

- `grokAPIKeyGenAIClass` → used to set `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY` (optional, for tracing)
- `LANGSMITH_TRACING` (optional)
- `LANGSMITH_PROJECT` (optional)

Example `.env`:

```env
grokAPIKeyGenAIClass=YOUR_GROQ_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
TAVILY_API_KEY=YOUR_TAVILY_KEY

# Optional (LangSmith)
LANGSMITH_API_KEY=YOUR_LANGSMITH_KEY
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=travel-planner
```

---

## How to run

### Option A — Run in Jupyter Notebook (recommended for this project)

1. Open `Travel_Planner.ipynb`
2. Run cells top-to-bottom
3. The Gradio UI block will launch the chat experience

### Option B — Use the provided helper functions (programmatic)

The notebook defines a two-step interaction pattern:

#### 1) Start: generate plans (pauses after suggestions)

```python
boot = run_travel_agent("Warm beach vacation in December with good nightlife")
print(boot["plans_text"])
```

#### 2) Continue:

**Approve a plan**

```python
cont = continue_travel_agent(
    boot["thread_id"],
    boot["state_after_suggest"],
    approved=True,
    selected_index=2,   # 1-based index
)
print(cont["final_state"]["response"])
```

**Reject and refine**

```python
cont = continue_travel_agent(
    boot["thread_id"],
    boot["state_after_suggest"],
    approved=False,
    feedback="Prefer Europe, 5-7 days, culture + food markets, moderate budget"
)
print(cont["plans_text"])  # refined suggestions
```

---

## Gradio chat commands

In the chat UI, after the bot shows plan options:

- **Choose a plan**  
  `choose 2`

- **Reject and refine**  
  `reject: Prefer Europe, 5-7 days, museums + food`

- **Show plans again**  
  `show plans`

- **Start over**  
  `start over`

---

## Notes & limitations

- **Web context quality** depends on Tavily results and the clarity of your prompt.
- **Structured JSON enforcement** is implemented; malformed JSON is handled gracefully by a parsing helper.
- The current plan schema is optimized for **high-level planning**. If you want consistent hotel pricing / budget breakdowns, add:
  - Additional schema fields (hotel budget range, flight estimate, daily spend, etc.)
  - A dedicated tool/node for cost lookup or structured budgeting

---

## Upcoming improvements

- Add a **Budgeting / Hotel Search node** with structured outputs
- Add **multi-city itineraries** and day-by-day schedules as structured JSON
- Persist sessions per user with unique `thread_id`s (instead of the fixed `travel_session_1`)
- Add tests for JSON parsing + routing logic
- Export final itinerary as a downloadable PDF/markdown from the UI

---
