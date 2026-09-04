"""
Travel Planner Agent

A standalone Gradio application that:
1. Accepts a natural-language travel request.
2. Searches the web for relevant travel information.
3. Generates up to five travel plan options.
4. Lets the user choose a plan or provide feedback for refinement.
5. Generates a detailed itinerary for the selected plan.

Environment variables are loaded from .env.
"""

import logging
import os
import re
import uuid
from typing import Any, Dict, Optional, TypedDict

import gradio as gr
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Validate the environment variables required by the application."""
    required_variables = ("GROQ_API_KEY", "TAVILY_API_KEY")
    missing = [name for name in required_variables if not os.getenv(name)]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env file."
        )


validate_environment()


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class State(TypedDict):
    query: str
    plans: Dict[str, Dict[str, Any]]
    approved: Optional[bool]
    selected_plan: Optional[Dict[str, Any]]
    feedback: Optional[str]
    response: Optional[str]
    tavilyResponse: Optional[str]


# ---------------------------------------------------------------------------
# Travel-plan parsing
# ---------------------------------------------------------------------------

PLAN_KEYS = [
    "Destination",
    "Number of Days",
    "Why It's a Must-See",
    "Highlights & Activities",
    "Who It Suits",
]


def build_plans_dict(json_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Convert model-generated JSON into a normalized travel-plan dictionary.

    The function accepts either a JSON object or an array and limits the
    result to five valid plans.
    """
    try:
        data = __import__("json").loads(json_text)
    except __import__("json").JSONDecodeError:
        start = json_text.find("[")
        end = json_text.rfind("]")

        if start == -1 or end <= start:
            logger.warning("Could not parse travel plans as JSON.")
            return {}

        try:
            data = __import__("json").loads(json_text[start : end + 1])
        except __import__("json").JSONDecodeError:
            logger.warning("Could not recover valid JSON from model output.")
            return {}

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return {}

    plans = []

    for item in data[:5]:
        if isinstance(item, dict) and item.get("Destination"):
            plan = {
                key: str(item.get(key, "")).strip().replace("\n", " ")
                for key in PLAN_KEYS
            }
            plans.append(plan)

    return {f"Plan{i + 1}": plan for i, plan in enumerate(plans)}


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def initiate_travel_plan(state: State) -> State:
    """Create a search query and retrieve relevant travel information."""
    prompt = ChatPromptTemplate.from_template(
        "You are a travel planner. "
        "Create a question that satisfies the customer's query and can be "
        "used to search the internet for trip-planning information.\n"
        "Query: {query}"
    )

    chain = prompt | ChatGroq(model=MODEL_NAME, temperature=0)
    travel_plan_question = chain.invoke({"query": state["query"]}).content

    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(travel_plan_question)

    tavily_response = ""
    for result in search_docs:
        content = result.get("content", "")
        tavily_response += f"\n\n{content}\n"

    return {"tavilyResponse": tavily_response}


def suggest_travel_plan(state: State) -> State:
    """Generate up to five travel plans from the search results."""
    schema_json = """[
    {
        "Destination": "string",
        "Number of Days": "string",
        "Why It's a Must-See": "string",
        "Highlights & Activities": "string",
        "Who It Suits": "string"
    }
    ]"""

    prompt = ChatPromptTemplate.from_template(
        "You are a travel planner. Using ONLY the context below, "
        "produce between 1 and 5 plans (never more than 5).\n"
        "Return STRICT JSON, and NOTHING else. Do not include markdown "
        "fences or explanations.\n\n"
        "JSON shape (array):\n"
        "{schema}\n\n"
        "Rules:\n"
        "• Exactly this key spelling and capitalization.\n"
        "• Each field is a single line string (no embedded newlines).\n"
        "• 1–2 sentences per field where applicable.\n"
        "• Max 5 items in the array.\n"
        "• Return ONLY valid JSON (no trailing commas, no comments).\n\n"
        "Context:\n"
        "{tavilyResponse}"
    )

    chain = prompt | ChatGroq(model=MODEL_NAME, temperature=0)

    raw = chain.invoke(
        {
            "tavilyResponse": state["tavilyResponse"],
            "schema": schema_json,
        }
    ).content

    return {"plans": build_plans_dict(raw)}


def human_approval(state: State) -> State:
    """Validate the user's approval or revision request."""
    approved = state.get("approved")

    if approved is True:
        if not state.get("selected_plan"):
            raise ValueError(
                "approved=True but 'selected_plan' is missing."
            )
        return state

    if approved is False:
        if not state.get("feedback"):
            raise ValueError(
                "approved=False but 'feedback' is missing."
            )
        return state

    return state


def apply_feedback(state: State) -> State:
    """Append user feedback to the query for another search cycle."""
    feedback = (state.get("feedback") or "").strip()

    if feedback:
        new_query = f"{state['query']} | preferences: {feedback}"
        return {**state, "query": new_query}

    return state


def show_selected_plan(state: State) -> State:
    """Generate a detailed itinerary for the selected travel plan."""
    plan = state.get("selected_plan") or {}

    if not plan:
        return {**state, "response": "No plan selected."}

    context = "\n".join(
        f"{key}: {value}" for key, value in plan.items()
    )

    prompt = ChatPromptTemplate.from_template(
        "You are a travel planner. Using ONLY the context and customer "
        "query below, provide a detailed travel itinerary.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Query:\n"
        "{query}"
    )

    chain = prompt | ChatGroq(model=MODEL_NAME, temperature=0)

    raw = chain.invoke(
        {
            "query": state["query"],
            "context": context,
        }
    ).content

    return {**state, "response": raw}


# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------

def merge_state(old: State, new: State) -> State:
    """Merge old and new state while ignoring None values."""
    return {
        **old,
        **{key: value for key, value in new.items() if value is not None},
    }


def approval_router(state: State) -> str:
    """Route the workflow based on the user's approval state."""
    if state.get("approved") and state.get("selected_plan"):
        return "accept"

    if state.get("approved") is False:
        return "revise"

    return "pending"


def build_workflow():
    """Build and compile the travel-planning LangGraph workflow."""
    workflow = StateGraph(State, merge=merge_state)

    workflow.add_node("initiateTravelPlan", initiate_travel_plan)
    workflow.add_node("suggestTravelPlan", suggest_travel_plan)
    workflow.add_node("humanApproval", human_approval)
    workflow.add_node("applyFeedback", apply_feedback)
    workflow.add_node("showSelectedPlan", show_selected_plan)

    workflow.add_edge("initiateTravelPlan", "suggestTravelPlan")
    workflow.add_edge("suggestTravelPlan", "humanApproval")

    workflow.add_conditional_edges(
        "humanApproval",
        approval_router,
        {
            "accept": "showSelectedPlan",
            "revise": "applyFeedback",
            "pending": "humanApproval",
        },
    )

    workflow.add_edge("applyFeedback", "initiateTravelPlan")
    workflow.add_edge("showSelectedPlan", END)
    workflow.set_entry_point("initiateTravelPlan")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


app = build_workflow()


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------

def stringify_plans(plans: Dict[str, Dict[str, Any]]) -> str:
    """Format generated travel plans for display in the chat."""
    if not plans:
        return "No plans generated."

    lines = []

    for index, (key, plan) in enumerate(plans.items(), start=1):
        title = plan.get("Destination", key) or f"Plan {index}"
        lines.append(f"✅ Plan {index}: {title}")

        for key, value in plan.items():
            lines.append(f"- {key}: {value}")

        lines.append("")

    return "\n".join(lines).strip()


def select_plan_by_index(
    plans: Dict[str, Dict[str, Any]],
    index_1_based: int,
) -> Dict[str, Any]:
    """Select a plan using its one-based display index."""
    all_plans = list(plans.values())

    if not 1 <= index_1_based <= len(all_plans):
        raise IndexError(
            f"Plan index {index_1_based} is out of range "
            f"1..{len(all_plans)}"
        )

    return all_plans[index_1_based - 1]


def create_thread_id() -> str:
    """Create a unique conversation ID for a new travel request."""
    return f"travel_session_{uuid.uuid4().hex}"


def run_until_suggest(query: str, thread_id: str) -> State:
    """Run the workflow until travel suggestions are generated."""
    initial_state: State = {
        "query": query,
        "plans": {},
        "approved": None,
        "selected_plan": None,
        "feedback": None,
        "response": None,
        "tavilyResponse": None,
    }

    state_after_suggest = None

    for value in app.stream(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
        interrupt_after=["suggestTravelPlan"],
    ):
        state_after_suggest = value

    if not state_after_suggest:
        raise RuntimeError(
            "Did not reach suggestTravelPlan. Check graph wiring."
        )

    return state_after_suggest


def run_travel_agent(query: str) -> Dict[str, Any]:
    """Start a new travel-planning conversation."""
    thread_id = create_thread_id()
    state_after_suggest = run_until_suggest(
        query,
        thread_id=thread_id,
    )

    return {
        "thread_id": thread_id,
        "state_after_suggest": state_after_suggest,
        "plans_text": stringify_plans(
            state_after_suggest.get("plans", {})
        ),
    }


def continue_travel_agent(
    thread_id: str,
    state_after_suggest: State,
    *,
    approved: bool,
    selected_index: Optional[int] = None,
    feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """Continue the workflow after the user chooses or rejects a plan."""
    updated = dict(state_after_suggest)

    if approved:
        if selected_index is None:
            raise ValueError(
                "approved=True requires selected_index (1-based)."
            )

        updated["approved"] = True
        updated["selected_plan"] = select_plan_by_index(
            updated["plans"],
            selected_index,
        )
        updated["feedback"] = None

        last_value = updated

        for value in app.stream(
            updated,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            last_value = value

        return {"final_state": last_value}

    if not feedback or not feedback.strip():
        raise ValueError(
            "approved=False requires a non-empty 'feedback' string."
        )

    updated["approved"] = False
    updated["selected_plan"] = None
    updated["feedback"] = feedback.strip()

    new_state = None

    for value in app.stream(
        updated,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
        interrupt_after=["suggestTravelPlan"],
    ):
        new_state = value

    if new_state is None:
        raise RuntimeError(
            "Did not reach suggestTravelPlan after feedback."
        )

    return {
        "thread_id": thread_id,
        "state_after_suggest": new_state,
        "plans_text": stringify_plans(
            new_state.get("plans", {})
        ),
    }


# ---------------------------------------------------------------------------
# Gradio chat interface
# ---------------------------------------------------------------------------

def init_session_state() -> Dict[str, Any]:
    """Create the state maintained by the Gradio chat session."""
    return {
        "mode": "awaiting_query",
        "thread_id": None,
        "state_after_suggest": None,
    }


CHOOSE_RE = re.compile(
    r"^(?:choose|approve|select)\s+(\d+)\b",
    flags=re.IGNORECASE,
)
REJECT_RE = re.compile(
    r"^(?:reject|feedback|revise)\s*:\s*(.+)",
    flags=re.IGNORECASE,
)
RESTART_RE = re.compile(
    r"^(?:start\s*over|new\s*query)\b",
    flags=re.IGNORECASE,
)
SHOW_RE = re.compile(
    r"^(?:show\s*plans|list\s*plans|plans)$",
    flags=re.IGNORECASE,
)

HELP_TEXT = (
    "Here’s how to continue:\n"
    "- **Choose a plan**: `choose 2`\n"
    "- **Reject and refine**: "
    "`reject: Prefer Europe, 5-7 days, museums + food`\n"
    "- **Start over**: `start over` then send a new query\n"
    "- **Show plans again**: `show plans`"
)


def chat_reply(
    user_msg: str,
    history: Any,
    session: Optional[Dict[str, Any]],
):
    """Handle a user's message and return the response plus session state."""
    del history  # Required by Gradio ChatInterface but not used.

    if (
        session is None
        or not isinstance(session, dict)
        or "mode" not in session
    ):
        session = init_session_state()

    message = (user_msg or "").strip()

    if not message:
        if session["mode"] == "awaiting_query":
            return (
                "Tell me what kind of trip you want "
                "(e.g., 'Beach vacation in December, budget-friendly').",
                session,
            )

        if session["mode"] == "awaiting_choice":
            return (
                "I'm waiting for your choice. " + HELP_TEXT,
                session,
            )

        return "Type `start over` to plan another trip.", session

    try:
        # Initial travel request.
        if session["mode"] == "awaiting_query":
            if RESTART_RE.match(message):
                return (
                    "Okay, tell me what kind of trip you want.",
                    session,
                )

            boot = run_travel_agent(message)

            session["thread_id"] = boot["thread_id"]
            session["state_after_suggest"] = boot["state_after_suggest"]
            session["mode"] = "awaiting_choice"

            response = (
                "I searched and prepared some options for you.\n\n"
                f"{boot['plans_text']}\n\n"
                f"{HELP_TEXT}"
            )
            return response, session

        # User is choosing or refining suggested plans.
        if session["mode"] == "awaiting_choice":
            if RESTART_RE.match(message):
                session = init_session_state()
                return (
                    "Okay, tell me what kind of trip you want.",
                    session,
                )

            if SHOW_RE.match(message):
                if (
                    session["state_after_suggest"]
                    and "plans" in session["state_after_suggest"]
                ):
                    plans_text = stringify_plans(
                        session["state_after_suggest"]["plans"]
                    )
                    return (
                        f"{plans_text}\n\n{HELP_TEXT}",
                        session,
                    )

                return "No plans available yet. Start with a new query.", session

            choose_match = CHOOSE_RE.match(message)

            if choose_match:
                index = int(choose_match.group(1))

                result = continue_travel_agent(
                    session["thread_id"],
                    session["state_after_suggest"],
                    approved=True,
                    selected_index=index,
                )

                final_state = result.get("final_state", {})
                itinerary = (
                    final_state.get("response")
                    or "Itinerary generated."
                )

                session["mode"] = "done"
                return itinerary, session

            reject_match = REJECT_RE.match(message)

            if reject_match:
                feedback = reject_match.group(1).strip()

                if not feedback:
                    return (
                        "Please include your preferences after `reject:`.",
                        session,
                    )

                result = continue_travel_agent(
                    session["thread_id"],
                    session["state_after_suggest"],
                    approved=False,
                    feedback=feedback,
                )

                session["state_after_suggest"] = result[
                    "state_after_suggest"
                ]

                response = (
                    "Got it — I refined the options based on your feedback.\n\n"
                    f"{result['plans_text']}\n\n"
                    f"{HELP_TEXT}"
                )
                return response, session

            # Treat a sufficiently long natural-language message as a new request.
            if (
                len(message.split()) > 3
                and ":" not in message
                and not message.lower().startswith(
                    ("choose", "approve", "select")
                )
            ):
                boot = run_travel_agent(message)

                session["thread_id"] = boot["thread_id"]
                session["state_after_suggest"] = boot[
                    "state_after_suggest"
                ]
                session["mode"] = "awaiting_choice"

                response = (
                    "New request received. Here are fresh suggestions:\n\n"
                    f"{boot['plans_text']}\n\n"
                    f"{HELP_TEXT}"
                )
                return response, session

            return "I didn't catch that. " + HELP_TEXT, session

        # Itinerary already generated.
        if session["mode"] == "done":
            if RESTART_RE.match(message):
                session = init_session_state()
                return (
                    "Okay, tell me what kind of trip you want.",
                    session,
                )

            return (
                "You're all set! Type `start over` to plan another trip.",
                session,
            )

        session = init_session_state()
        return "Let's start a new trip plan.", session

    except Exception:
        logger.exception("Error while processing travel request.")
        return (
            "Sorry, I ran into an error while processing your request. "
            "Please try again.",
            session,
        )


# ---------------------------------------------------------------------------
# Gradio application
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Create the Gradio user interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## ✈️ Travel Planner (Chat Mode)\n"
            "Start by telling me what kind of trip you want."
        )

        session = gr.State(init_session_state())

        gr.ChatInterface(
            fn=chat_reply,
            additional_inputs=[session],
            additional_outputs=[session],
            title="Travel Planner",
            description=(
                "Type your travel request to get suggested plans. "
                "Then reply with `choose N` or "
                "`reject: <feedback>`."
            ),
        )

    return demo


def main() -> None:
    """Application entry point."""
    demo = create_app()
    demo.launch()


if __name__ == "__main__":
    main()
