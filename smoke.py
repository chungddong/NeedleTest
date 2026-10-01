import time, json, needle
@needle.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 27, "sky": "clear"}
t=time.perf_counter()
agent = needle.Needle(tools=[get_weather])
print("init s", time.perf_counter()-t)
for q in ["what's it like in Lagos right now?", "weather in Seoul please"]:
    t=time.perf_counter(); r = agent.complete(q); dt=time.perf_counter()-t
    print(f"{dt*1000:.1f} ms", json.dumps(r, ensure_ascii=False)[:800])
    agent.reset()
