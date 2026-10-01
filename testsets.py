"""Test sets for benchmarking Needle 3 on a Raspberry Pi 5.

Expected argument values may be plain values (exact match after normalization),
OneOf(...) for several acceptable spellings, or Text(...) for free text that is
matched leniently (equal or containment after normalization).
"""
from typing import Annotated, Literal, Optional

import needle

TODAY = "2026-10-01"  # auto_date feeds the engine today's date; relative cases depend on it


class OneOf:
    def __init__(self, *values):
        self.values = values

    def __repr__(self):
        return "OneOf" + repr(self.values)


class Text:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Text({self.value!r})"


# --------------------------------------------------------------------------- tools

Room = Literal["kitchen", "living_room", "bedroom", "bathroom"]
HHMM = Annotated[str, needle.Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
DATE = Annotated[str, needle.Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


@needle.tool
def get_weather(city: str, unit: Optional[Literal["celsius", "fahrenheit"]] = None):
    """Get the current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit; include only when the user names one.
    """
    return {"city": city, "temp": 21}


@needle.tool
def set_timer(minutes: Annotated[int, needle.Field(ge=1, le=600)]):
    """Start a countdown timer. Hours are converted to minutes.

    Args:
        minutes: Timer duration in minutes.
    """
    return {"ok": True}


@needle.tool
def set_alarm(time: HHMM):
    """Set an alarm clock for a time of day.

    Args:
        time: Alarm time in 24-hour HH:MM.
    """
    return {"ok": True}


@needle.tool
def send_message(recipient: str, message: str):
    """Send a text message to a contact.

    Args:
        recipient: Who receives the message.
        message: The message body.
    """
    return {"ok": True}


@needle.tool
def call_contact(name: str):
    """Place a phone call to a contact.

    Args:
        name: The contact to call.
    """
    return {"ok": True}


@needle.tool
def play_music(query: str):
    """Play music by song, artist, album, or genre.

    Args:
        query: What to play.
    """
    return {"ok": True}


@needle.tool
def navigate_to(destination: str,
                mode: Optional[Literal["driving", "walking", "transit", "cycling"]] = None):
    """Start turn-by-turn navigation to a destination.

    Args:
        destination: Where to go.
        mode: Travel mode; include only when stated.
    """
    return {"ok": True}


@needle.tool
def create_event(title: str, date: DATE, time: Optional[HHMM] = None):
    """Add an event to the calendar.

    Args:
        title: What the event is.
        date: Event date as YYYY-MM-DD.
        time: Start time in 24-hour HH:MM; include only when stated.
    """
    return {"ok": True}


@needle.tool
def control_lights(room: Room, action: Literal["on", "off", "dim"],
                   brightness_percent: Annotated[Optional[int], needle.Field(ge=0, le=100)] = None):
    """Turn lights on or off in a room, or dim them to a brightness percentage. Never use this for any other device.

    Args:
        room: The room to control.
        action: on, off, or dim.
        brightness_percent: Brightness from 0 to 100, only with a dim request that names a number.
    """
    return {"ok": True}


@needle.tool
def set_thermostat(temperature: Annotated[int, needle.Field(ge=10, le=30)]):
    """Set the home thermostat target temperature in degrees Celsius.

    Args:
        temperature: Target temperature in degrees Celsius.
    """
    return {"ok": True}


@needle.tool
def set_volume(level: Annotated[int, needle.Field(ge=0, le=100)]):
    """Set the speaker volume.

    Args:
        level: Volume level from 0 to 100.
    """
    return {"ok": True}


TOOLS = [get_weather, set_timer, set_alarm, send_message, call_contact, play_music,
         navigate_to, create_event, control_lights, set_thermostat, set_volume]


def C(_tool, **arguments):
    return {"name": _tool, "arguments": arguments}


# --------------------------------------------------------------------------- tool-call cases

CASES = {
    "basic": [
        ("What's the weather in Tokyo?", [C("get_weather", city="Tokyo")]),
        ("Is it raining in London right now?", [C("get_weather", city="London")]),
        ("How's the weather in Paris", [C("get_weather", city="Paris")]),
        ("Set a timer for 15 minutes", [C("set_timer", minutes=15)]),
        ("Start a 5 minute timer", [C("set_timer", minutes=5)]),
        ("Wake me up at 6:30 am", [C("set_alarm", time="06:30")]),
        ("Set an alarm for 7 pm", [C("set_alarm", time="19:00")]),
        ("Text Mom that I'll be late", [C("send_message", recipient="Mom", message=Text("I'll be late"))]),
        ("Call Sarah", [C("call_contact", name="Sarah")]),
        ("Phone Dad", [C("call_contact", name="Dad")]),
        ("Play some jazz", [C("play_music", query=Text("jazz"))]),
        ("Play Bohemian Rhapsody by Queen", [C("play_music", query=Text("Bohemian Rhapsody"))]),
        ("Take me to the airport", [C("navigate_to", destination=Text("airport"))]),
        ("Navigate to Central Park", [C("navigate_to", destination="Central Park")]),
        ("Turn on the kitchen lights", [C("control_lights", room="kitchen", action="on")]),
        ("Turn off the bedroom light", [C("control_lights", room="bedroom", action="off")]),
        ("Lights on in the living room", [C("control_lights", room="living_room", action="on")]),
        ("Set the thermostat to 22 degrees", [C("set_thermostat", temperature=22)]),
        ("Set the volume to 40", [C("set_volume", level=40)]),
        ("Add a dentist appointment on 2026-10-14",
         [C("create_event", title=Text("dentist"), date="2026-10-14")]),
    ],
    "args": [
        ("What's the weather in Chicago in fahrenheit?", [C("get_weather", city="Chicago", unit="fahrenheit")]),
        ("Berlin weather in celsius please", [C("get_weather", city="Berlin", unit="celsius")]),
        ("Set a timer for 2 hours", [C("set_timer", minutes=120)]),
        ("Timer for an hour and a half", [C("set_timer", minutes=90)]),
        ("Set a 45-minute timer", [C("set_timer", minutes=45)]),
        ("Dim the living room lights to 30%", [C("control_lights", room="living_room", action="dim", brightness_percent=30)]),
        ("Dim the bedroom lights to 10 percent", [C("control_lights", room="bedroom", action="dim", brightness_percent=10)]),
        ("Bathroom lights off", [C("control_lights", room="bathroom", action="off")]),
        ("Set the alarm for 5:45", [C("set_alarm", time="05:45")]),
        ("Alarm at 11:15 pm", [C("set_alarm", time="23:15")]),
        ("Walk me to the train station", [C("navigate_to", destination=Text("train station"), mode="walking")]),
        ("Directions to Seattle by car", [C("navigate_to", destination="Seattle", mode="driving")]),
        ("How do I get to the museum by bus", [C("navigate_to", destination=Text("museum"), mode="transit")]),
        ("Bike route to the beach", [C("navigate_to", destination=Text("beach"), mode="cycling")]),
        ("Make it 19 degrees in here", [C("set_thermostat", temperature=19)]),
        ("Set volume to zero", [C("set_volume", level=0)]),
        ("Set the volume to 75 percent", [C("set_volume", level=75)]),
        ("Add lunch with Alex on October 20 2026 at 12:30",
         [C("create_event", title=Text("lunch with Alex"), date="2026-10-20", time="12:30")]),
        ("Put team meeting on the calendar for 2026-11-03 at 9am",
         [C("create_event", title=Text("team meeting"), date="2026-11-03", time="09:00")]),
        ("Schedule a haircut tomorrow at 3pm",
         [C("create_event", title=Text("haircut"), date="2026-10-02", time="15:00")]),
    ],
    "casual": [
        ("whats the wether like in madrid", [C("get_weather", city="Madrid")]),
        ("how cold is it in oslo", [C("get_weather", city="Oslo")]),
        ("yo can u ping jake and say running 10 min late",
         [C("send_message", recipient="Jake", message=Text("running 10 min late"))]),
        ("tell emma dinner is ready", [C("send_message", recipient="Emma", message=Text("dinner is ready"))]),
        ("give john a call", [C("call_contact", name="John")]),
        ("i need a timer, like 20 mins", [C("set_timer", minutes=20)]),
        ("its freezing, bump the heat to 24", [C("set_thermostat", temperature=24)]),
        ("kill the lights in the kitchen", [C("control_lights", room="kitchen", action="off")]),
        ("can you turn off the lights in the bathroom", [C("control_lights", room="bathroom", action="off")]),
        ("crank the volume up to 80", [C("set_volume", level=80)]),
        ("get me directions to the nearest gas station", [C("navigate_to", destination=Text("gas station"))]),
        ("throw on some taylor swift", [C("play_music", query=Text("taylor swift"))]),
        ("lemme hear some lo-fi beats", [C("play_music", query=Text("lo-fi"))]),
        ("need to be up at 5 am tmrw", [C("set_alarm", time="05:00")]),
        ("plz set alarm 8:00", [C("set_alarm", time="08:00")]),
    ],
    "multi": [
        ("Turn off the kitchen lights and set the thermostat to 20",
         [C("control_lights", room="kitchen", action="off"), C("set_thermostat", temperature=20)]),
        ("Set a timer for 10 minutes and play some classical music",
         [C("set_timer", minutes=10), C("play_music", query=Text("classical"))]),
        ("What's the weather in Rome and in Milan?",
         [C("get_weather", city="Rome"), C("get_weather", city="Milan")]),
        ("Turn on the bedroom lights and the bathroom lights",
         [C("control_lights", room="bedroom", action="on"), C("control_lights", room="bathroom", action="on")]),
        ("Call Mike and set the volume to 20",
         [C("call_contact", name="Mike"), C("set_volume", level=20)]),
        ("Set an alarm for 6:00 and a timer for 30 minutes",
         [C("set_alarm", time="06:00"), C("set_timer", minutes=30)]),
        ("Dim the living room lights to 40 percent and play some jazz",
         [C("control_lights", room="living_room", action="dim", brightness_percent=40), C("play_music", query=Text("jazz"))]),
        ("Set the thermostat to 21 and turn off the bedroom lights",
         [C("set_thermostat", temperature=21), C("control_lights", room="bedroom", action="off")]),
        ("Check the weather in Seoul and navigate to Gangnam Station",
         [C("get_weather", city="Seoul"), C("navigate_to", destination=Text("Gangnam Station"))]),
        ("Text Lisa see you soon and call Tom",
         [C("send_message", recipient="Lisa", message=Text("see you soon")), C("call_contact", name="Tom")]),
    ],
    "refusal": [
        ("Order me a large pepperoni pizza", []),
        ("Book a flight to New York", []),
        ("What's the capital of France?", []),
        ("Turn on the fan in the bedroom", []),
        ("Lock the front door", []),
        ("Translate hello into Spanish", []),
        ("Take a photo", []),
        ("What's 15 times 23?", []),
        ("Tell me a joke", []),
        ("Start the dishwasher", []),
        ("Open the garage door", []),
        ("Transfer $50 to John", []),
        ("Turn on the TV", []),
        ("Read my latest email", []),
        ("Don't turn on the kitchen lights", []),
    ],
    "korean": [
        ("서울 날씨 어때?", [C("get_weather", city=OneOf("서울", "Seoul"))]),
        ("부산 날씨 알려줘", [C("get_weather", city=OneOf("부산", "Busan"))]),
        ("10분 타이머 맞춰줘", [C("set_timer", minutes=10)]),
        ("아침 7시에 알람 맞춰줘", [C("set_alarm", time="07:00")]),
        ("거실 불 켜줘", [C("control_lights", room="living_room", action="on")]),
        ("침실 불 꺼", [C("control_lights", room="bedroom", action="off")]),
        ("부엌 조명 30%로 어둡게 해줘", [C("control_lights", room="kitchen", action="dim", brightness_percent=30)]),
        ("온도 23도로 맞춰줘", [C("set_thermostat", temperature=23)]),
        ("볼륨 50으로 해줘", [C("set_volume", level=50)]),
        ("엄마한테 전화해줘", [C("call_contact", name=OneOf("엄마", "Mom"))]),
        ("재즈 틀어줘", [C("play_music", query=OneOf("재즈", "jazz"))]),
        ("피자 주문해줘", []),
    ],
}


# --------------------------------------------------------------------------- structured extraction

@needle.tool
def contact_card(name: str, email: str, phone: str):
    """Extract a contact card.

    Args:
        name: Full name of the person.
        email: Email address.
        phone: Phone number exactly as written.
    """


@needle.tool
def order_line(item: str, quantity: int, unit_price: float):
    """Extract one purchase line.

    Args:
        item: Product name.
        quantity: Number of units.
        unit_price: Price of one unit, a number without currency symbol.
    """


@needle.tool
def flight_info(airline: str, flight_number: str, origin: str, destination: str):
    """Extract flight details.

    Args:
        airline: Airline name.
        flight_number: Flight code such as KE123.
        origin: Departure city.
        destination: Arrival city.
    """


@needle.tool
def meeting_info(date: DATE, time: HHMM, location: str):
    """Extract when and where a meeting happens.

    Args:
        date: Meeting date as YYYY-MM-DD.
        time: Start time in 24-hour HH:MM.
        location: Where it happens.
    """


EXTRACTION = [
    (contact_card, "Hi, it's Jenny Park - reach me at jenny.park@example.com or 010-2345-6789 anytime.",
     dict(name="Jenny Park", email="jenny.park@example.com", phone="010-2345-6789")),
    (contact_card, "Contact: David Kim | david.kim@acme.io | +1 415 555 0132",
     dict(name="David Kim", email="david.kim@acme.io", phone="+1 415 555 0132")),
    (contact_card, "for billing questions email maria.lopez@shop.co, or phone Maria Lopez on 555-0199",
     dict(name="Maria Lopez", email="maria.lopez@shop.co", phone="555-0199")),
    (contact_card, "Sent from my phone. -- Tom Becker, tom@becker.dev, cell 020 7946 0958",
     dict(name="Tom Becker", email="tom@becker.dev", phone="020 7946 0958")),
    (order_line, "I'd like 3 bags of coffee beans at $12.50 each.",
     dict(item=Text("coffee beans"), quantity=3, unit_price=12.5)),
    (order_line, "qty 10 x USB-C cable @ 7.99",
     dict(item=Text("USB-C cable"), quantity=10, unit_price=7.99)),
    (order_line, "Please send two desk lamps; they're listed at 45 dollars a piece.",
     dict(item=Text("desk lamp"), quantity=2, unit_price=45)),
    (order_line, "Invoice line: Ergonomic chair, 1 unit, unit price 289.00",
     dict(item=Text("Ergonomic chair"), quantity=1, unit_price=289)),
    (flight_info, "Your Korean Air flight KE017 from Seoul to Los Angeles departs at 14:20.",
     dict(airline="Korean Air", flight_number="KE017", origin="Seoul", destination="Los Angeles")),
    (flight_info, "Booking confirmed: Lufthansa LH 400, Frankfurt -> New York.",
     dict(airline="Lufthansa", flight_number=OneOf("LH 400", "LH400"), origin="Frankfurt", destination="New York")),
    (flight_info, "we're flying delta DL89 out of atlanta and landing in tokyo",
     dict(airline="Delta", flight_number="DL89", origin="Atlanta", destination="Tokyo")),
    (flight_info, "Itinerary - Air France AF11 | Paris CDG to New York JFK",
     dict(airline="Air France", flight_number="AF11", origin=Text("Paris"), destination=Text("New York"))),
    (meeting_info, "Let's meet on 2026-10-08 at 14:00 in Conference Room B.",
     dict(date="2026-10-08", time="14:00", location=Text("Conference Room B"))),
    (meeting_info, "Kickoff is October 12, 2026, 9:30 am, at the Gangnam office.",
     dict(date="2026-10-12", time="09:30", location=Text("Gangnam office"))),
    (meeting_info, "reminder: sync moved to nov 2 2026 4pm, starbucks on main st",
     dict(date="2026-11-02", time="16:00", location=Text("starbucks"))),
    (meeting_info, "The review happens 2026-12-01 at 10:15 via Zoom.",
     dict(date="2026-12-01", time="10:15", location=Text("Zoom"))),
]


# --------------------------------------------------------------------------- embedding routing

ROUTES = {
    "weather": ["what's the weather today", "will it rain tomorrow", "temperature outside"],
    "music": ["play a song", "put on some music", "next track please"],
    "lights": ["turn on the lights", "switch off the lamp", "make the room brighter"],
    "timer": ["set a timer", "count down five minutes", "remind me in ten minutes"],
    "messaging": ["send a text", "message my friend", "reply to the chat"],
    "navigation": ["directions to the mall", "how do I get home", "find the nearest station"],
    "calendar": ["what's on my schedule", "add a meeting", "book an appointment for friday"],
    "shopping": ["buy more milk", "add eggs to my cart", "order a new phone case"],
}

ROUTE_QUERIES = [
    ("is it going to be sunny this weekend", "weather"),
    ("how hot will it get in the afternoon", "weather"),
    ("do I need an umbrella", "weather"),
    ("I want to listen to rock", "music"),
    ("skip this song", "music"),
    ("shuffle my playlist", "music"),
    ("it's too dark in here", "lights"),
    ("lamp off please", "lights"),
    ("dim the bedroom", "lights"),
    ("start a countdown for the eggs", "timer"),
    ("ping me in 20 minutes", "timer"),
    ("stopwatch for 3 minutes", "timer"),
    ("tell Sam I'm running late", "messaging"),
    ("text my wife", "messaging"),
    ("write back to the group", "messaging"),
    ("route to the airport", "navigation"),
    ("which way to the train station", "navigation"),
    ("navigate to work", "navigation"),
    ("am I free on Thursday", "calendar"),
    ("move my dentist appointment", "calendar"),
    ("put lunch with Joe on the calendar", "calendar"),
    ("we're out of bread", "shopping"),
    ("purchase batteries", "shopping"),
    ("reorder dog food", "shopping"),
]


# --------------------------------------------------------------------------- tool-count scaling

_FILLER_NOUNS = [
    "stock_price", "news_headlines", "recipe", "translation", "flight_status", "package_tracking",
    "currency_rate", "sports_score", "movie_showtimes", "restaurant", "parking_spot", "ev_charger",
    "air_quality", "pollen_count", "tide_times", "sunrise_time", "podcast_episode", "audiobook",
    "dictionary_definition", "unit_conversion", "horoscope", "lottery_result", "traffic_report",
    "bus_arrival", "bike_share", "taxi_fare", "hotel_room", "museum_hours", "pharmacy_hours",
    "gym_class", "library_book", "parcel_locker", "recycling_day", "school_lunch", "tv_listing",
    "radio_station", "concert_ticket", "wine_pairing", "plant_care", "pet_feeding",
]


def filler_tools(n):
    """n distinct JSON-schema tools that never match any test query."""
    out = []
    for noun in _FILLER_NOUNS[:n]:
        out.append({
            "name": f"lookup_{noun}",
            "description": f"Look up the {noun.replace('_', ' ')} for the user.",
            "parameters": {"type": "object",
                           "properties": {"subject": {"type": "string",
                                                      "description": f"What {noun.replace('_', ' ')} to look up."}},
                           "required": ["subject"]},
        })
    return out
