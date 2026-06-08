from app.conversation.manager import ConversationManager
from app.services.calendar import CalendarService
from app.tools.registry import ToolRegistry

# A booking needs a name and at least a phone number; email is optional.
BOOKING_REQUIRED_FIELDS = ("name", "phone")

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
            "description": (
                "ID of the slot to book (must be one previously offered by check_availability)."
            ),
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
    async def check_availability(args: dict, ctx: ConversationManager) -> dict:
        date = args.get("date", "")
        time_pref = args.get("time_preference", "")
        legal_area = args.get("legal_area", ctx.state.legal_area.value)

        slots = await calendar.get_available_slots(
            date=date, legal_area=legal_area, time_preference=time_pref
        )

        if slots:
            offered = [
                {
                    "id": s["id"],
                    "date": s["date"],
                    "time": s["time"],
                    "lawyer": s["lawyer_name"],
                    "duration": s["duration_minutes"],
                }
                for s in slots[:3]
            ]
            ctx.state.offered_slot_ids = [s["id"] for s in offered]
            return {"available": True, "slots": offered}

        alternatives = await calendar.get_next_available(
            after_date=date, legal_area=legal_area, limit=3
        )
        alt_list = [
            {
                "id": s["id"],
                "date": s["date"],
                "time": s["time"],
                "lawyer": s["lawyer_name"],
            }
            for s in alternatives
        ]
        ctx.state.offered_slot_ids = [s["id"] for s in alt_list]
        return {"available": False, "alternatives": alt_list}

    return check_availability


def _make_book_consultation(calendar: CalendarService):
    async def book_consultation(args: dict, ctx: ConversationManager) -> dict:
        missing = [f for f in BOOKING_REQUIRED_FIELDS if f not in ctx.state.entities]
        unconfirmed = [
            f
            for f in BOOKING_REQUIRED_FIELDS
            if f in ctx.state.entities and not ctx.state.entities[f].confirmed
        ]
        if missing or unconfirmed:
            return {
                "status": "blocked",
                "missing_fields": missing,
                "unconfirmed_fields": unconfirmed,
            }

        slot_id = args.get("slot_id")
        if not slot_id:
            return {"status": "error", "message": "slot_id is required."}

        if ctx.state.offered_slot_ids and slot_id not in ctx.state.offered_slot_ids:
            return {
                "status": "error",
                "message": "This slot was not offered. Use check_availability first.",
            }

        slot = await calendar.get_slot_by_id(slot_id)
        if slot is None:
            return {"status": "error", "message": "Slot not found."}
        if slot.get("is_booked"):
            return {"status": "error", "message": "That slot is no longer available."}
        caller_area = ctx.state.legal_area.value
        slot_area = slot.get("legal_area")
        if caller_area != "unknown" and slot_area and slot_area != caller_area:
            return {
                "status": "error",
                "message": f"Slot is for {slot['legal_area']}, but caller needs {caller_area}.",
            }

        entities = ctx.state.entities
        booking = await calendar.create_booking(
            slot_id=slot_id,
            call_id=ctx.state.call_id,
            caller_name=args.get("caller_name", "")
            or (entities["name"].value if "name" in entities else ""),
            caller_email=args.get("caller_email", "")
            or (entities["email"].value if "email" in entities else ""),
            caller_phone=args.get("caller_phone", "")
            or (entities["phone"].value if "phone" in entities else ""),
            matter_type=args.get("matter_type", "") or ctx.state.legal_area.value,
            matter_description=args.get("matter_description", "")
            or (entities["matter_description"].value if "matter_description" in entities else ""),
        )

        if booking is None:
            return {
                "status": "error",
                "message": "That slot is no longer available.",
            }

        ctx.state.booking_confirmed = True
        ctx.state.booked_slot = booking
        ctx.advance_phase()

        return {
            "status": "booked",
            "booking_id": booking["id"],
            "details": booking,
        }

    return book_consultation


def register_booking_tools(registry: ToolRegistry, calendar: CalendarService) -> None:
    registry.register(
        name="check_availability",
        fn=_make_check_availability(calendar),
        description=(
            "Check available consultation slots for a given date. "
            "Returns up to 3 options, or alternatives if the requested date is full. "
            "Present these options to the caller and let them choose."
        ),
        parameters=CHECK_SCHEMA,
    )
    registry.register(
        name="book_consultation",
        fn=_make_book_consultation(calendar),
        description=(
            "Book a consultation slot that was previously offered by check_availability. "
            "All caller details (name, email, phone) must be confirmed first. "
            "The slot_id must be one that was offered to the caller."
        ),
        parameters=BOOK_SCHEMA,
    )
