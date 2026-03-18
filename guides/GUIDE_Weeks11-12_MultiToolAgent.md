# Guide: Multi-Tool Agent (Weeks 11-12)

## Big Picture
Build an AI agent that can use multiple tools to accomplish complex tasks through reasoning.

**Why?** Move from passive QA to active problem-solving. Agents interpret goals and decide which tools/steps to use.

**Key Skills:**
- Tool definition and integration
- Reasoning loops (REACT: Reasoning + Acting)
- Error handling and recovery
- Tool chaining (one tool output → another tool input)
- Evaluating agent performance

---

## Concept 1: Agents vs Chains

**Chain:** Predetermined sequence
```
Input → Step1 → Step2 → Step3 → Output
(Fixed order, known path)
```

**Agent:** Decides best path
```
Input → Reason ("what do I need?") → 
→ Choose Tool → Execute → Check Result → 
→ Continue or Done? → Output
(Adaptive, learns from failures)
```

**Example:**
```
User: "Book a flight to Paris and get restaurant recommendations"

Chain: Always does: Search Flight → Book Flight → Search Restaurants

Agent: 
1. Thinks: "I need to book flight AND get restaurants"
2. Chooses: "First book flight with flight_booking tool"
3. Executes: Finds flight, books it
4. Checks: Success, move to next goal
5. Chooses: "Now search restaurants with restaurant_search tool"
6. Executes: Finds top restaurants
7. Checks: Done!
```

---

## Concept 2: Tool Definition

**What:** Defining what tools an agent can use.

```python
from langchain.tools import Tool

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

def get_news(topic: str) -> str:
    """Get latest news for a topic."""
    return f"News about {topic}: [headlines...]"

# Wrap as LangChain tools
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Get current weather for a location. Input: city name"
)

news_tool = Tool(
    name="get_news",
    func=get_news,
    description="Get latest news. Input: topic"
)

tools = [weather_tool, news_tool]
```

**Key Elements:**
- `name`: How agent refers to tool
- `func`: Actual function being called
- `description`: What tool does + input format

---

## Concept 3: REACT (Reasoning + Acting)

**What:** Thinking loop that agents use.

```
Thought: "What's the current problem?"
→ Action: "Which tool should I use?"
→ Action Input: "What parameters?"
→ Observation: "What did the tool return?"
→ Thought: "Do I need more info or am I done?"
```

**Example Flow:**
```
Thought: User wants restaurant recommendation in Paris
Action: restaurant_search_tool
Action Input: "Paris, 4-star, French cuisine"
Observation: Returns list of 3 restaurants

Thought: I have good options, should I search for reviews?
Action: get_restaurant_reviews
Action Input: "Restaurant A"
Observation: Reviews show 4.8 stars

Thought: I have enough info, time to respond
Final Answer: "I recommend Restaurant A, highly rated, French cuisine..."
```

---

## Concept 4: Creating an Agent

**What:** Using LangChain to build reasoning agent.

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

# Setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
tools = [weather_tool, news_tool, calculator_tool]

# Create agent with REACT loop
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # Shows thinking process!
)

# Use agent
response = agent.run("What's the weather in NYC and latest tech news?")
```

**Output:**
```
Thought: Need weather and news
Action: get_weather
Action Input: "NYC"
Observation: Sunny, 72°F

Thought: Got weather, now get news
Action: get_news
Action Input: "technology"
Observation: [3 news items...]

Final Answer: "NYC is sunny and 72°F. Latest tech news: ..."
```

---

## Concept 5: Tool Chaining

**What:** Using output of one tool as input to another.

```python
def calculate_trip_cost(distance_km: float) -> float:
    """Calculate trip cost based on distance."""
    return distance_km * 0.5

def get_distance(location1: str, location2: str) -> float:
    """Get distance between two locations."""
    # Calls external API
    return 150.0

# Agent flow:
# User: "What's the cost to drive from NYC to Boston?"
# Step 1: get_distance("NYC", "Boston") → 215 km
# Step 2: calculate_trip_cost(215) → $107.50
```

```python
# Agent automatically chains:
tools = [
    Tool("get_distance", get_distance, "..."),
    Tool("calculate_trip_cost", calculate_trip_cost, "...")
]

agent.run("What's driving cost NYC to Boston?")
# Agent: Calls get_distance, sees 215, then calls calculate_trip_cost(215)
```

---

## Concept 6: Error Handling

**What:** What happens when tool fails.

```python
def retry_agent(query: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            result = agent.run(query)
            return result
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                return "Sorry, couldn't complete that task"
            # Continue to retry

# In agent prompt, add:
prompt = """
If a tool returns an error:
1. Acknowledge the error
2. Try an alternative approach
3. If all fail, tell user you can't help
"""
```

---

## Concept 7: Tool Limitations

**What:** Constraints to prevent misuse.

```python
# Without limits - dangerous!
agent_tools = [delete_database_tool, send_email_tool]
# Agent might delete data trying to solve problem!

# With limits - safer
agent_tools = [search_database_tool, read_email_tool]
# Read-only, no destructive actions

# In system prompt add:
prompt = """
You can ONLY use these tools: search, read, reason
You CANNOT: delete data, modify records, or send communications
If user asks for restricted action, explain why you can't.
"""
```

---

## Concept 8: Agent Memory

**What:** Agents that remember context across interactions.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    verbose=True
)

# Turn 1
agent.run("My name is Alice. Find me restaurants in Paris.")
memory saves: "User is Alice"

# Turn 2
agent.run("What was my previous request?")
# Agent remembers: "You asked for Paris restaurants"
```

---

## Concept 9: Agent Evaluation

**What:** Measuring if agent works well.

```python
# Create test suite
test_cases = [
    {
        "query": "What's weather in NYC and stock price of AAPL?",
        "expected_tools": ["weather_tool", "stock_tool"],
        "expected_output_contains": ["sunny", "AAPL"]
    },
    ...
]

# Run tests
correct = 0
for test in test_cases:
    result = agent.run(test["query"])
    if test["expected_output_contains"][0] in result:
        correct += 1

success_rate = correct / len(test_cases)
print(f"Agent success rate: {success_rate * 100}%")
```

---

## Concept 10: Production Considerations

**What:** Deploy agents responsibly.

```python
# Rate limiting
from time import sleep
from functools import wraps

def rate_limit(calls_per_minute: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep(60 / calls_per_minute)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(10)
def call_agent(query):
    return agent.run(query)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

# Validation
def validate_input(query: str) -> bool:
    if len(query) > 1000:
        return False
    if "malicious" in query.lower():
        return False
    return True
```

---

## Challenge Approach

### Challenge 1-3: Tool Creation
- Define 3-5 useful tools
- Each with clear name, description, function
- Test each tool independently

### Challenge 4-6: Basic Agent
- Create ReAct agent with tools
- Test with 3 queries
- Observe thinking/reasoning process

### Challenge 7-9: Tool Chaining & Error Handling
- Create tasks requiring multiple tools
- Test error recovery
- Add fallback handling

### Challenge 10-12: Evaluation & Deployment
- Create test suite with expected outcomes
- Measure success rate
- Document agent behavior and limitations
- Prepare deployment guide

---

## Key Takeaways

✅ **Agents decide their path** (unlike chains with fixed steps)

✅ **REACT loop = reasoning + action** (transparent thinking process)

✅ **Tools extend agent capabilities** (connect to any service/API)

✅ **Tool chaining enables complex tasks** (output of one → input of next)

✅ **Error handling critical** (agents must recover gracefully)

✅ **Evaluation matters** (test thoroughly before deployment)
