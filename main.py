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

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile"
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _invoke_llm(messages):
    return llm.invoke(messages)


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: Annotated[int, operator.add]
    
def flight_agent(state: TravelState):
    try:
        flight_data = search_flights(state["user_query"])
        status = "ok"
    except Exception as e:
        flight_data = f"⚠️ Flight search unavailable: {e}"
        status = "error"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content=f"Flight results fetched ({status})")],
        "llm_calls": 1 
    }
    
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    try:
        hotel_data = tavily_search(query)
        status = "ok"
    except Exception as e:
        hotel_data = f"⚠️ Hotel search unavailable: {e}"
        status = "error"

    return {
        "hotel_results": hotel_data,
        "messages": [AIMessage(content=f"Hotel information fetched ({status})")],
        "llm_calls": 1 
    }


def itinerary_agent(state: TravelState):
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
    except Exception as e:
        itinerary_text = f"⚠️ Itinerary generation failed: {e}"
        msg = AIMessage(content=itinerary_text)

    return {
        "itinerary": itinerary_text,
        "messages": [msg],
        "llm_calls": 1 
    }
    
     
def final_agent(state: TravelState):
    final_prompt = f"""
    Generate final travel response based on the following information:

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """
    try:
        response = _invoke_llm([
            SystemMessage(
                content="You are a travel assistant that writes a clear, friendly final travel summary for the user."
            ),
            HumanMessage(content=final_prompt)
        ])
        msg = response
    except Exception as e:
        msg = AIMessage(content=f"⚠️ Final response generation failed: {e}")

    return {
        "messages": [msg],
        "llm_calls": 1 
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
except Exception as e:
    raise RuntimeError(
        f"Could not connect to Postgres checkpoint DB at startup: {e}"
    ) from e

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