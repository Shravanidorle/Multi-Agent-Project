import os
import re
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

# quick lookup table — extend as needed, or swap for a proper airport-code API/dataset later
CITY_TO_IATA = {
    "delhi": "DEL",
    "goa": "GOI",
    "mumbai": "BOM",
    "bangalore": "BLR",
    "chennai": "MAA",
    "kolkata": "CCU",
    "tokyo": "NRT",
    "osaka": "KIX",
    "paris": "CDG",
    "dubai": "DXB",
    "bangkok": "BKK",
    "rome": "FCO",
}

def _extract_cities(query: str):
    """Naive extractor for patterns like 'from Delhi to Goa' or 'Delhi to Goa'."""
    q = query.lower()
    match = re.search(r"(?:from\s+)?([a-z\s]+?)\s+to\s+([a-z\s]+?)(?:[,\.]|\bfor\b|\bunder\b|$)", q)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def search_flights(query):
    origin, dest = _extract_cities(query)
    dep_iata = CITY_TO_IATA.get(origin)
    arr_iata = CITY_TO_IATA.get(dest)

    url = "http://api.aviationstack.com/v1/flights"

    params = {
        "access_key": API_KEY,
        "limit": 5,
    }
    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata

    response = requests.get(url, params=params)
    data = response.json()

    flights = []

    if "data" in data:
        for flight in data["data"][:5]:

            airline = flight.get("airline", {}).get("name", "Unknown Airline")

            departure = flight.get(
                "departure", {}
                ).get("airport", "Unknown Departure Airport")

            arrival = flight.get(
                "arrival", {}
                ).get("airport", "Unknown Arrival Airport")

            status = flight.get("flight_status", "Unknown Status")

            flights.append(
                f"""
                Airline: {airline}
                Departure: {departure}
                Arrival: {arrival}
                Status: {status}
                """
            )

    if not flights:
        return f"No matching flights found for {origin or 'origin'} → {dest or 'destination'}."

    return "\n".join(flights)