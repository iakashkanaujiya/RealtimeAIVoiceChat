import re
from uuid import uuid4
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState


from ..protocols import Agent
from ..utils import init_chat_model
from .._prompts import AIRLINE_CUSTOMER_SUPPORT_PROMPT


class AirlineAgent(Agent):
    """This class implements a comprehensive Emirates Airlines customer service agent.

    Args:
        model : Name of the model default is "gpt-4o-mini"
        provider : Model provider, such as `openai`, `google_genai`, `groq`, default is "openai"
        temperature : Model temperature, default is 0.2
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.2,
    ):
        self.llm = init_chat_model(model, provider, temperature)
        self.agent = self._create_agent()
        self.runnable_config = {"configurable": {"thread_id": uuid4().hex}}
        self._init_mock_data()

    def _init_mock_data(self):
        """Initialize realistic mock data for Emirates Airlines."""
        self.bookings = {
            "EK123ABC": {
                "flight": "EK215",
                "route": "Dubai (DXB) to London (LHR)",
                "date": "2025-12-20",
                "time": "14:30",
                "seat": "12A",
                "class": "Economy",
                "passenger": "Akash Kanaujiya",
                "status": "Confirmed",
                "gate": "A12",
            },
            "EK456DEF": {
                "flight": "EK241",
                "route": "Dubai (DXB) to New York (JFK)",
                "date": "2025-12-22",
                "time": "08:45",
                "seat": "6F",
                "class": "Business",
                "passenger": "Rohan Punia",
                "status": "Confirmed",
                "gate": "B5",
            },
        }

        self.flights = {
            "EK215": {
                "status": "On Time",
                "gate": "A12",
                "departure": "14:30",
                "delay": 0,
            },
            "EK241": {
                "status": "Delayed",
                "gate": "B5",
                "departure": "09:15",
                "delay": 30,
            },
            "EK502": {
                "status": "Boarding",
                "gate": "C8",
                "departure": "11:20",
                "delay": 0,
            },
            "EK185": {
                "status": "Departed",
                "gate": "A15",
                "departure": "06:45",
                "delay": 15,
            },
        }

    def _create_agent(self):
        def _prompt(state: AgentState, config: RunnableConfig):
            messages = [
                SystemMessage(AIRLINE_CUSTOMER_SUPPORT_PROMPT.strip()),
            ] + state["messages"]
            return messages

        agent = create_react_agent(
            model=self.llm,
            prompt=_prompt,
            tools=[
                self.booking_lookup_tool,
                self.flight_status_tool,
                self.update_seat,
                self.faq_lookup_tool,
                self.baggage_tool,
                self.display_seat_map,
                self.cancel_flight,
                self.rebook_flight,
                self.upgrade_request,
                self.meal_preference,
                self.compensation_claim,
                self.check_in_assistance,
                self.special_assistance,
            ],
            checkpointer=InMemorySaver(),
        )
        return agent

    # ============================================
    # Enhanced Tools implementations
    # ============================================

    def booking_lookup_tool(self, confirmation_number: str) -> str:
        """Lookup booking details using confirmation number."""
        confirmation_number = confirmation_number.upper().strip()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            return (
                f"Booking found for {booking['passenger']}:\n"
                f"Flight: {booking['flight']} - {booking['route']}\n"
                f"Date: {booking['date']} at {booking['time']}\n"
                f"Seat: {booking['seat']} ({booking['class']} Class)\n"
                f"Status: {booking['status']}\n"
                f"Gate: {booking['gate']}"
            )
        else:
            return "I couldn't find a booking with that confirmation number. Please check the number and try again, or provide your ticket number."

    def flight_status_tool(self, flight_number: str) -> str:
        """Get real-time flight status information."""
        flight_number = flight_number.upper().strip()

        if flight_number in self.flights:
            flight = self.flights[flight_number]
            if flight["delay"] > 0:
                return (
                    f"Flight {flight_number} is {flight['status']} - "
                    f"Delayed by {flight['delay']} minutes. "
                    f"New departure time: {flight['departure']} from gate {flight['gate']}."
                )
            else:
                return (
                    f"Flight {flight_number} is {flight['status']} - "
                    f"Scheduled departure: {flight['departure']} from gate {flight['gate']}."
                )
        else:
            return f"I couldn't find flight {flight_number}. Please verify the flight number or check your booking confirmation."

    def update_seat(self, confirmation_number: str, new_seat: str) -> str:
        """Update seat assignment for a booking."""
        confirmation_number = confirmation_number.upper().strip()
        new_seat = new_seat.upper().strip()

        # Validate seat format
        if not re.match(r"^\d{1,2}[A-K]$", new_seat):
            return "Please provide a valid seat number (e.g., 12A, 7F)."

        if confirmation_number in self.bookings:
            old_seat = self.bookings[confirmation_number]["seat"]
            self.bookings[confirmation_number]["seat"] = new_seat
            return (
                f"Seat updated successfully from {old_seat} to {new_seat} "
                f"for confirmation {confirmation_number}. "
                f"Your new boarding pass will be available 24 hours before departure."
            )
        else:
            return "Booking not found. Please verify your confirmation number."

    def faq_lookup_tool(self, question: str) -> str:
        """Comprehensive FAQ lookup for Emirates Airlines."""
        q = question.lower()

        if any(word in q for word in ["bag", "baggage", "luggage", "carry"]):
            return (
                "Emirates baggage allowance:\n"
                "• Economy: 30kg checked, 7kg carry-on\n"
                "• Business: 40kg checked, 7kg carry-on\n"
                "• First: 50kg checked, 7kg carry-on\n"
                "Carry-on dimensions: 55x38x25cm."
            )
        elif any(word in q for word in ["wifi", "internet", "online"]):
            return (
                "Emirates offers complimentary Wi-Fi on most flights:\n"
                "• Connect to 'Emirates WiFi'\n"
                "• Complimentary for Skywards members\n"
                "• Browse, chat, and email included\n"
                "• Streaming packages available for purchase."
            )
        elif any(
            word in q
            for word in ["meal", "food", "dietary", "kosher", "halal", "vegetarian"]
        ):
            return (
                "Emirates offers various meal options:\n"
                "• Special dietary meals available (vegetarian, halal, kosher, etc.)\n"
                "• Must be requested 24 hours before departure\n"
                "• Complimentary meals on all long-haul flights\n"
                "• Premium dining in Business and First Class."
            )
        elif any(word in q for word in ["seat", "upgrade", "extra legroom"]):
            return (
                "Seat options:\n"
                "• Free seat selection 48 hours before departure\n"
                "• Premium seats with extra legroom available for fee\n"
                "• Upgrades to Business/First subject to availability\n"
                "• Exit row seats require meeting safety criteria."
            )
        elif any(word in q for word in ["check", "checkin", "online"]):
            return (
                "Online check-in:\n"
                "• Available 48 hours to 90 minutes before departure\n"
                "• Use emirates.com or the Emirates app\n"
                "• Mobile boarding passes available\n"
                "• Seat selection and meal preferences can be updated."
            )
        else:
            return "I'd be happy to help! For detailed information, please visit emirates.com or ask me about specific topics like baggage, meals, seats, or check-in."

    def baggage_tool(self, query: str) -> str:
        """Detailed baggage information and assistance."""
        q = query.lower()

        if any(word in q for word in ["fee", "cost", "price", "charge"]):
            return (
                "Emirates baggage fees:\n"
                "• Excess weight: $30-50 per kg depending on route\n"
                "• Extra pieces: $200-400 per bag\n"
                "• Oversize items: $300-500\n"
                "• Sports equipment: Special rates apply"
            )
        elif any(word in q for word in ["delayed", "lost", "missing"]):
            return (
                "For delayed or lost baggage:\n"
                "• Report immediately at baggage services\n"
                "• Keep your baggage receipt\n"
                "• Track your bag at emirates.com/baggage\n"
                "• Compensation available for essentials during delay"
            )
        elif any(word in q for word in ["restricted", "prohibited", "allowed"]):
            return (
                "Baggage restrictions:\n"
                "• Liquids: 100ml containers in 1L clear bag\n"
                "• Batteries: Power banks under 100Wh in carry-on only\n"
                "• Sharp objects: Not allowed in carry-on\n"
                "• Full list available at emirates.com/baggage-restrictions"
            )
        else:
            return (
                "Emirates baggage allowance varies by class:\n"
                "Economy: 30kg checked + 7kg carry-on\n"
                "Business: 40kg checked + 7kg carry-on\n"
                "First: 50kg checked + 7kg carry-on\n"
                "What specific baggage question can I help with?"
            )

    def display_seat_map(self, flight_number: str = "") -> str:
        """Show interactive seat map."""
        if flight_number:
            return f"DISPLAY_SEAT_MAP_{flight_number.upper()}"
        return "DISPLAY_SEAT_MAP"

    def cancel_flight(self, confirmation_number: str) -> str:
        """Cancel flight booking with proper procedures."""
        confirmation_number = confirmation_number.upper().strip()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            flight_date = datetime.strptime(booking["date"], "%Y-%m-%d")
            days_until_flight = (flight_date - datetime.now()).days

            if days_until_flight > 1:
                self.bookings[confirmation_number]["status"] = "Cancelled"
                return (
                    f"Flight {booking['flight']} cancelled successfully. "
                    f"Refund will be processed within 7-10 business days. "
                    f"Cancellation confirmation: CANC{confirmation_number[:6]}"
                )
            else:
                return (
                    "Cancellation within 24 hours requires special handling. "
                    "Please contact Emirates customer service at +971-4-214-4444 "
                    "or visit the airport for immediate assistance."
                )
        else:
            return "Booking not found. Please verify your confirmation number."

    def rebook_flight(self, confirmation_number: str, new_date: str = "") -> str:
        """Rebook flight to a different date."""
        confirmation_number = confirmation_number.upper().strip()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            if new_date:
                self.bookings[confirmation_number]["date"] = new_date
                return (
                    f"Flight rebooking initiated for {booking['flight']} to {new_date}. "
                    f"Change fee may apply based on fare rules. "
                    f"New booking confirmation will be sent via email."
                )
            else:
                return (
                    f"I can help rebook your flight {booking['flight']}. "
                    f"What new date would you prefer? "
                    f"Please note that fare differences and change fees may apply."
                )
        else:
            return "Booking not found. Please provide a valid confirmation number."

    def upgrade_request(
        self, confirmation_number: str, upgrade_class: str = "Business"
    ) -> str:
        """Request cabin upgrade."""
        confirmation_number = confirmation_number.upper().strip()
        upgrade_class = upgrade_class.title()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            current_class = booking["class"]

            if current_class == "Economy" and upgrade_class == "Business":
                return (
                    f"Business Class upgrade request submitted for flight {booking['flight']}. "
                    f"Upgrade cost: $800-1200 depending on route. "
                    f"I'll check availability and send confirmation within 2 hours."
                )
            elif current_class in ["Economy", "Business"] and upgrade_class == "First":
                return (
                    f"First Class upgrade request submitted for flight {booking['flight']}. "
                    f"Subject to availability and fare difference. "
                    f"Priority given to Skywards Gold/Platinum members."
                )
            else:
                return f"You're already in {current_class} Class. Would you like to explore other services?"
        else:
            return "Booking not found. Please verify your confirmation number."

    def meal_preference(self, confirmation_number: str, meal_type: str = "") -> str:
        """Update meal preferences."""
        confirmation_number = confirmation_number.upper().strip()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            if meal_type:
                return (
                    f"Meal preference updated to {meal_type} for flight {booking['flight']}. "
                    f"Special meals must be requested 24 hours before departure. "
                    f"Your preference has been noted in your booking."
                )
            else:
                return (
                    "Available special meals: Vegetarian, Vegan, Hindu, Kosher, Halal, "
                    "Gluten-free, Diabetic, Low-sodium, Child meal. "
                    "Which would you prefer?"
                )
        else:
            return "Booking not found. Please provide a valid confirmation number."

    def compensation_claim(self, issue_type: str, flight_number: str = "") -> str:
        """Handle compensation claims."""
        issue_type = issue_type.lower()

        if "delay" in issue_type:
            return (
                "For flight delays over 3 hours, you may be entitled to compensation. "
                "Please provide: flight details, delay duration, and reason. "
                "Compensation ranges from €250-€600 based on distance and delay. "
                "Submit claims at emirates.com/compensation"
            )
        elif "cancel" in issue_type:
            return (
                "Flight cancellation compensation depends on notice given and reason. "
                "You're entitled to refund or rebooking plus possible compensation. "
                "Less than 14 days notice: €250-€600 compensation may apply. "
                "Submit claim with booking reference and cancellation details."
            )
        elif "baggage" in issue_type:
            return (
                "Baggage compensation available for delays over 21 hours. "
                "Essential items reimbursement up to $200 for delay. "
                "Lost baggage: compensation up to $1,600 per passenger. "
                "Submit receipts and report reference number."
            )
        else:
            return (
                "Emirates compensation covers flight delays, cancellations, and baggage issues. "
                "What specific issue would you like to claim compensation for?"
            )

    def check_in_assistance(self, confirmation_number: str) -> str:
        """Provide check-in assistance."""
        confirmation_number = confirmation_number.upper().strip()

        if confirmation_number in self.bookings:
            booking = self.bookings[confirmation_number]
            flight_date = datetime.strptime(booking["date"], "%Y-%m-%d")
            hours_until_flight = (flight_date - datetime.now()).total_seconds() / 3600

            if hours_until_flight <= 48 and hours_until_flight > 1.5:
                return (
                    f"Check-in available for flight {booking['flight']}! "
                    f"Complete at emirates.com or Emirates app. "
                    f"Current seat: {booking['seat']}. "
                    f"Download mobile boarding pass after check-in."
                )
            elif hours_until_flight <= 1.5:
                return (
                    f"Check-in closed for flight {booking['flight']}. "
                    f"Please proceed directly to airport check-in counter. "
                    f"Arrive at least 2 hours early for international flights."
                )
            else:
                return (
                    f"Check-in opens 48 hours before departure for flight {booking['flight']}. "
                    f"You can check in starting {(flight_date - timedelta(hours=48)).strftime('%Y-%m-%d at %H:%M')}."
                )
        else:
            return "Booking not found. Please verify your confirmation number."

    def special_assistance(
        self, assistance_type: str = "", confirmation_number: str = ""
    ) -> str:
        """Arrange special assistance services."""
        assistance_type = assistance_type.lower()

        if "wheelchair" in assistance_type:
            return (
                "Wheelchair assistance available at no charge. "
                "Types: WCHR (steps only), WCHS (steps and distances), WCHC (own wheelchair). "
                "Request at booking or call +971-4-214-4444 at least 48 hours before travel."
            )
        elif "medical" in assistance_type:
            return (
                "Medical assistance services: oxygen, stretcher, medical clearance. "
                "Requires advance booking and medical documentation. "
                "Contact Emirates Medical Desk: +971-4-214-5555 "
                "Submit MEDIF form 7 days before travel."
            )
        elif "dietary" in assistance_type or "meal" in assistance_type:
            return (
                "Special dietary assistance: religious, medical, and preference-based meals. "
                "Request 24 hours before departure through manage booking "
                "or call Emirates customer service."
            )
        elif "pet" in assistance_type or "animal" in assistance_type:
            return (
                "Pet travel services available. "
                "Small pets in cabin (max 8kg including carrier). "
                "Larger pets in cargo hold with climate control. "
                "Advance booking required. Health certificates mandatory."
            )
        else:
            return (
                "Emirates special assistance includes: wheelchair, medical support, "
                "dietary requirements, pet travel, unaccompanied minors. "
                "What type of assistance do you need?"
            )

    # ============================================

    async def generate(self, message: str) -> str:
        """Generate response using the enhanced airline agent workflow."""
        result = self.agent.invoke(
            {
                "messages": [
                    HumanMessage(content=message),
                ]
            },
            self.runnable_config,
        )
        return result["messages"][-1].content

    async def generate_stream(self, message: str):
        """Generate streaming response with enhanced error handling."""
        try:
            async for message_chunk in self.agent.astream(
                {
                    "messages": [
                        HumanMessage(content=message),
                    ]
                },
                self.runnable_config,
                stream_mode="messages",
            ):
                if hasattr(message_chunk[0], "content") and message_chunk[0].content:
                    yield message_chunk[0].content
        except Exception:
            yield "I apologize, but I'm experiencing technical difficulties. Please try again or contact Emirates customer service at +971-4-214-4444."
