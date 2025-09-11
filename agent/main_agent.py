from langgraph.prebuilt import create_react_agent


# fake tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_property_value(MOF: str, property: str):
    """This tool returns the value of a specific property for a given MOF.
    property can be "surface_area" or "pore_volume"
    """
    if property == "surface_area":
        return "1500 m2/g"
    elif property == "pore_volume":
        return "0.8 cm3/g"
    else:
        return "Property not found."
    
def create_my_agent():
    # Create the agent with the tool
    agent = create_react_agent(
        model="gpt-4o",
        tools=[get_weather, get_property_value],
        prompt="You are a helpful assistant"
    )

    return agent

