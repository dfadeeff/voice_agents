from app.conversation.manager import ConversationManager
from app.models.schemas import CallPhase
from app.services.calendar import CalendarService
from app.tools.registry import ToolRegistry

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "Requested date in YYYY-MM-DD format.",
        },
        "time_preference": {
            "type": "string",
            "description": "Preferred time: 'morning', 'afternoon', or specific like '14:00'.",
        },
        "legal_area": {
            "type": "string",
            "description": "Area of law for the consultation.",
        },
    },
    "required": ["date"],
}

BOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "slot_id": {
            "type": "integer",
            "description": "ID of the slot to book (from check_availability results).",
        },
        "caller_name": {"type": "string"},
        "caller_email": {"type": "string"},
        "caller_phone": {"type": "string"},
        "matter_type": {"type": "string"},
        "matter_description": {"type": "string"},
    },
    "required": ["slot_id", "caller_name"],
}


def _make_check_availability(calendar: CalendarService):
    async def check_availability(
        args: dict, ctx: ConversationManager
    ) -> dict:
        date = args.get("date", "")
        time_pref = args.get("time_preference", "")
        legal_area = args.get("legal_area", ctx.state.legal_area.value)

        slots = await calendar.get_available_slots(
            date=date, legal_area=legal_area, time_preference=time_pref
        )

        if slots:
            return {
                "available": True,
                "slots": [
                    {
                        "id": s["id"],
                        "date": s["date"],
                        "time": s["time"],
                        "lawyer": s["lawyer_name"],
                        "duration": s["duration_minutes"],
                    }
                    for s in slots[:3]
                ],
                "message": (
                    "Found available slots. Present them to the caller"
                    " and ask which they prefer."
                ),
            }

        alternatives = await calendar.get_next_available(
            after_date=date, legal_area=legal_area, limit=3
        )
        return {
            "available": False,
            "message": "Requested date/time is not available.",
            "alternatives": [
                {
                    "id": s["id"],
                    "date": s["date"],
                    "time": s["time"],
                    "lawyer": s["lawyer_name"],
                }
                for s in alternatives
            ],
            "instruction": "Inform the caller the slot is taken and offer these alternatives.",
        }

    return check_availability


def _make_book_consultation(calendar: CalendarService):
    async def book_consultation(
        args: dict, ctx: ConversationManager
    ) -> dict:
        unconfirmed = [
            name
            for name, e in ctx.state.entities.items()
            if name in ("name", "email", "phone") and not e.confirmed
        ]
        if unconfirmed:
            return {
                "status": "blocked",
                "message": f"Cannot book yet. These fields need confirmation: {unconfirmed}",
            }

        slot_id = args.get("slot_id")
        if not slot_id:
            return {"status": "error", "message": "slot_id is required."}

        booking = await calendar.create_booking(
            slot_id=slot_id,
            call_id=ctx.state.call_id,
            caller_name=args.get("caller_name", ""),
            caller_email=args.get("caller_email", ""),
            caller_phone=args.get("caller_phone", ""),
            matter_type=args.get("matter_type", ""),
            matter_description=args.get("matter_description", ""),
        )

        if booking is None:
            return {
                "status": "error",
                "message": "That slot is no longer available. Please check availability again.",
            }

        ctx.state.booking_confirmed = True
        ctx.state.phase = CallPhase.CONFIRMATION

        return {
            "status": "booked",
            "booking_id": booking["id"],
            "details": booking,
            "instruction": (
                "Confirm all booking details back to the caller"
                " and end the call warmly."
            ),
        }

    return book_consultation


def register_booking_tools(
    registry: ToolRegistry, calendar: CalendarService
) -> None:
    registry.register(
        name="check_availability",
        fn=_make_check_availability(calendar),
        description=(
            "Check available consultation slots for a given date."
            " Returns up to 3 options, or alternatives if the requested date is full."
        ),
        parameters=CHECK_SCHEMA,
    )
    registry.register(
        name="book_consultation",
        fn=_make_book_consultation(calendar),
        description=(
            "Book a consultation slot. All caller details (name, email, phone)"
            " must be confirmed first. Confirm ALL details back to the caller"
            " before calling this."
        ),
        parameters=BOOK_SCHEMA,
    )