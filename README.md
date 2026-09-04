# ✈️ Travel Planner — Agentic Chatbot (LangGraph + LangChain)

An **agentic travel-planning chatbot** built with **LangGraph, LangChain, Groq, and Tavily** that researches travel destinations, generates multiple personalized travel plans, and uses a **human-in-the-loop workflow** to refine and finalize an itinerary.

You describe the type of trip you want, for example:

> _"Warm beach vacation in December with good nightlife."_

The agent then:

1. Converts your request into a search-ready query.
2. Uses **Tavily Search** to retrieve fresh travel information from the web.
3. Generates **1–5 personalized travel plan options**.
4. Presents the plans through an interactive **Gradio chat interface**.
5. Allows you to **select a plan** or **reject and provide feedback**.
6. Re-runs the research and planning process when feedback is provided.
7. Generates a **detailed final itinerary** for the selected plan.

A simple **Gradio chat UI** is included for interactive use.

---

## 🚀 Features

- 🤖 **Agentic workflow** powered by LangGraph
- 🔎 **Real-time web research** using Tavily
- 🧠 **LLM-powered travel planning** using Groq
- 🔄 **Iterative refinement** based on user feedback
- 👤 **Human-in-the-loop approval**
- 📋 Generates multiple travel options
- 🗂️ Structured travel-plan generation using JSON
- 💬 Interactive **Gradio chatbot UI**
- 💾 LangGraph checkpointing for workflow state
- 🔐 API keys managed securely through environment variables
- 🐍 Standalone Python application — no Jupyter Notebook required

---

## 🏗️ Architecture

The application uses a stateful LangGraph workflow:

```text
                    ┌─────────────────────┐
                    │    User Query       │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   initiateTravelPlan      │
                 │                          │
                 │ Convert query to search  │
                 │ Retrieve web context     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    suggestTravelPlan     │
                 │                          │
                 │ Generate 1–5 plans       │
                 │ Structured JSON output   │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Human Approval      │
                 │                          │
                 │   Select / Reject        │
                 └───────┬──────────┬───────┘
                         │          │
                    Accept│          │Reject + Feedback
                         │          │
                         ▼          ▼
              ┌──────────────┐  ┌──────────────────┐
              │showSelected  │  │   applyFeedback  │
              │    Plan      │  │                  │
              │              │  │ Update query     │
              │Generate      │  │ & restart search │
              │final itinerary│ └────────┬─────────┘
              └───────┬──────┘          │
                      │                  │
                      ▼                  │
                    END ◄────────────────┘
```

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

- `app.py` — end-to-end implementation:
  - Graph + state definition
  - Tavily retrieval
  - JSON plan generation + parsing helpers
  - “pause & resume” runner functions
  - Gradio chat UI wrapper

Before executing `python app.py` you need to install dependencies via:

```bash
pip install -qU -r requirements.txt
```

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

1. Install dependecies after initializing environment if needed.
2. Run `python app.py`.
3. The Gradio UI block will launch the chat experience

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
