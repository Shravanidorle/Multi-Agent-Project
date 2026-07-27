import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import logging

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("travel_agent")


DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _invoke_llm(messages):
    return llm.invoke(messages)

from pydantic import BaseModel, Field
from typing import Optional

class DayPlan(BaseModel):
    day: int = Field(description="Day number, e.g. 1, 2, 3")
    title: str = Field(description="Short title for the day, e.g. 'Arrival in Goa'")
    activities: list[str] = Field(description="List of activities/sightseeing/meals for this day")
    estimated_cost: Optional[str] = Field(default=None, description="Estimated cost for this day, e.g. '₹5,000'")

class TravelPlan(BaseModel):
    destination: str = Field(description="Destination city/country")
    origin: Optional[str] = Field(default=None, description="Departure city, if known")
    duration_days: int = Field(description="Total trip length in days")
    flight_summary: str = Field(description="Short summary of flight info found")
    hotel_summary: str = Field(description="Short summary of hotel/accommodation found")
    itinerary: list[DayPlan] = Field(description="Day-by-day plan")
    total_estimated_cost: str = Field(description="Total estimated cost for the trip, e.g. '₹1,20,000'")
    recommendations: list[str] = Field(description="Extra tips or recommendations for the traveler")
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: Annotated[int, operator.add]
    final_plan: Optional[TravelPlan]   # ← NEW
    
def flight_agent(state: TravelState):
    logger.info(f"flight_agent started | query={state['user_query']!r}")
    try:
        flight_data = search_flights(state["user_query"])
        status = "ok"
        logger.info("flight_agent succeeded")
    except Exception as e:
        flight_data = f"⚠️ Flight search unavailable: {e}"
        status = "error"
        logger.error(f"flight_agent failed: {e}", exc_info=True)

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content=f"Flight results fetched ({status})")],
        "llm_calls": 1 
    }
    
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    logger.info(f"hotel_agent started | query={query!r}")
    try:
        hotel_data = tavily_search(query)
        status = "ok"
        logger.info("hotel_agent succeeded")
    except Exception as e:
        hotel_data = f"⚠️ Hotel search unavailable: {e}"
        status = "error"
        logger.error(f"hotel_agent failed: {e}", exc_info=True)

    return {
        "hotel_results": hotel_data,
        "messages": [AIMessage(content=f"Hotel information fetched ({status})")],
        "llm_calls": 1 
    }


def itinerary_agent(state: TravelState):
    logger.info("itinerary_agent started")
    prompt = f"""
    Create a travel itinerary.
    User Query: {state['user_query']}
    Flight Results: {state['flight_results']}
    Hotel Results: {state['hotel_results']}
    """
    try:
        response = _invoke_llm([
            SystemMessage(content="You are a travel assistant that creates travel itineraries based on user queries and flight/hotel information."),
            HumanMessage(content=prompt)
        ])
        itinerary_text = response.content
        msg = response
        logger.info("itinerary_agent succeeded")
    except Exception as e:
        itinerary_text = f"⚠️ Itinerary generation failed: {e}"
        msg = AIMessage(content=itinerary_text)
        logger.error(f"itinerary_agent failed: {e}", exc_info=True)

    return {
        "itinerary": itinerary_text,
        "messages": [msg],
        "llm_calls": 1 
    }
    
     
def final_agent(state: TravelState):
    logger.info("final_agent started")
    final_prompt = f"""
    Generate a final structured travel plan based on the following information:

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}

    User query: {state['user_query']}
    """
    try:
        structured_llm = llm.with_structured_output(TravelPlan)
        plan: TravelPlan = structured_llm.invoke([
            SystemMessage(
                content="You are a travel assistant. Extract and organize the given information into the requested structured travel plan."
            ),
            HumanMessage(content=final_prompt)
        ])

        # build a readable markdown string so messages/frontend.py behavior is unchanged
        lines = [
            f"**Destination:** {plan.destination}",
            f"**Duration:** {plan.duration_days} days",
            f"**Flights:** {plan.flight_summary}",
            f"**Hotels:** {plan.hotel_summary}",
            "",
            "**Itinerary:**",
        ]
        for day in plan.itinerary:
            lines.append(f"- Day {day.day}: {day.title} — {', '.join(day.activities)}"
                          + (f" (est. {day.estimated_cost})" if day.estimated_cost else ""))
        lines.append("")
        lines.append(f"**Total Estimated Cost:** {plan.total_estimated_cost}")
        if plan.recommendations:
            lines.append("")
            lines.append("**Recommendations:**")
            for rec in plan.recommendations:
                lines.append(f"- {rec}")

        final_text = "\n".join(lines)
        msg = AIMessage(content=final_text)
        logger.info("final_agent succeeded")

    except Exception as e:
        plan = None
        msg = AIMessage(content=f"⚠️ Final response generation failed: {e}")
        logger.error(f"final_agent failed: {e}", exc_info=True)

    return {
        "messages": [msg],
        "llm_calls": 1,
        "final_plan": plan
    }
    
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge(START, "hotel_agent")
graph.add_edge("flight_agent", "itinerary_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

try:
    _conn = psycopg.connect(DATABASE_URL, autocommit=True)
    checkpointer = PostgresSaver(_conn)
    checkpointer.setup()
    logger.info("Postgres checkpointer connected and initialized")
except Exception as e:
    logger.critical(f"Failed to connect to Postgres checkpoint DB: {e}", exc_info=True)
    raise RuntimeError(f"Could not connect to Postgres checkpoint DB at startup: {e}") from e

app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable" :{
            "thread_id": "user_shravani"
        }
    }
    user_input = input("Enter travel request: ")
    
    result = app.invoke(
        {
            "messages" : [
                HumanMessage(content = user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config = config
    )
    print("\nFINAL RESPONSE:\n")
    
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")