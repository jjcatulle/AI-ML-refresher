# Guide: Multi-Tool Agent (Weeks 11-12)

## Beginner Start Here
You do not need prior agent framework experience to complete this week.

### What is an agent?
An agent is a component that receives a goal, chooses actions, and uses tools.

### Terms you must know first
- `Tool`: callable function/API the agent can use.
- `Orchestration`: order and rules for tool/agent execution.
- `State`: shared context carried across steps.
- `Reflection loop`: retry with feedback when output quality is low.
- `Fallback`: backup path when a tool fails.

### Modules used
- `langchain` or custom orchestrator logic.
- `requests` for external APIs.
- Optional storage/logging modules.

### How to study this guide
1. Build one tool and call it manually.
2. Add second tool and route logic.
3. Add critic/retry logic.
4. Log decisions and errors for debugging.

## Big Picture
Build an AI agent that can use multiple tools to accomplish complex tasks through reasoning.

**Why?** Move from passive QA to active problem-solving. Agents interpret goals and decide which tools/steps to use.

**Key Skills:**
- Tool definition and integration
- Reasoning loops (REACT: Reasoning + Acting)
- Error handling and recovery
- Tool chaining (one tool output → another tool input)
- Evaluating agent performance

## 💼 Real-World Use Cases
- **Customer support automation:** Agents decide whether to answer, search knowledge base, or escalate.
- **DevOps assistants:** Agents run diagnostics, restart services, and report status.
- **Research assistants:** Agents fetch data, summarize papers, and generate citations.

---

## 🛠️ Recommended Tools/APIs for Weeks 11-12

For Multi-Tool Agents, you'll create or connect to different tools. Choose a combination:

### Tool Option 1: Math Tools ✅ **EASIEST & RECOMMENDED**
- **What:** Calculator functions that agents use
- **Implementation:**
  ```python
  def add(a: float, b: float) -> float:
      """Add two numbers"""
      return a + b
  
  def multiply(a: float, b: float) -> float:
      """Multiply two numbers"""
      return a * b
  
  def power(base: float, exponent: float) -> float:
      """Raise base to exponent power"""
      return base ** exponent
  ```
- **Use case:** Agent solves math problems (e.g., "What's 2^10 * 5?")
- **Why:** Simple to understand, focuses on agent logic not external APIs.

### Tool Option 2: Web Search + Weather APIs 🌐
- **What:** Connect to free public APIs
- **Popular services:**
  - Weather API: `https://open-meteo.com/` (free, no key needed)
  - Wikipedia search: `pip install wikipedia`
  - Currency conversion: `fixer.io` (free tier available)
- **Example:**
  ```python
  import requests
  
  def get_weather(city: str) -> str:
      """Get current weather"""
      response = requests.get(
          f"https://api.open-meteo.com/v1/forecast?latitude=40&longitude=74"
      )
      return response.json()
  
  def search_wikipedia(query: str) -> str:
      """Search Wikipedia"""
      import wikipedia
      return wikipedia.summary(query)
  ```
- **Setup:** `pip install requests wikipedia`
- **Why:** Real APIs, good agent practice, practical tools.

### Tool Option 3: Database Query Tools 🗄️
- **What:** Allow agent to query a database
- **Setup (SQLite - simple):**
  ```python
  import sqlite3
  
  def query_database(sql: str) -> list:
      """Run SQL query"""
      conn = sqlite3.connect('company.db')
      cursor = conn.cursor()
      cursor.execute(sql)
      return cursor.fetchall()
  
  # Agent example query:
  # "How many employees are in sales?"
  # Agent translates to: SELECT COUNT(*) FROM employees WHERE dept='Sales'
  ```
- **Why:** Real-world skill, agents learn when to query vs when to search.

### Tool Option 4: Code Execution Tool 🐍
- **What:** Agent can run Python code to solve problems
- **Implementation:**
  ```python
  def code_executor(code: str) -> str:
      """Execute Python code and return result"""
      import subprocess
      result = subprocess.run(
          ['python', '-c', code],
          capture_output=True,
          text=True
      )
      return result.stdout
  
  # Agent might decide:
  # "I need to analyze this data, I'll write Python to process it"
  ```
- **Security warning:** Only use for trusted code in development!
- **Why:** Agents become problem-solvers beyond predefined tools.

### Tool Option 5: Language Translation Tools 🌍
- **What:** Agent translates between languages
- **Implementation:**
  ```python
  from transformers import pipeline
  
  def translate_text(text: str, source: str, target: str) -> str:
      """Translate text between languages"""
      translator = pipeline(
          "translation",
          model=f"Helsinki-NLP/opus-mt-{source}-{target}"
      )
      return translator(text)[0]["translation_text"]
  ```
- **Setup:** `pip install transformers torch`
- **Why:** Multi-lingual agents, realistic feature.

### Tool Option 6: Document Search Tool 📄
- **What:** Agent searches through documents (RAG-style)
- **Implementation:**
  ```python
  from sentence_transformers import SentenceTransformer, util
  import os
  
  def search_documents(query: str, folder: str) -> list:
      """Find relevant documents"""
      model = SentenceTransformer('all-MiniLM-L6-v2')
      
      # Load all documents
      docs = []
      for file in os.listdir(folder):
          with open(f'{folder}/{file}') as f:
              docs.append(f.read())
      
      # Find most similar
      query_embedding = model.encode(query)
      doc_embeddings = model.encode(docs)
      similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)
      
      top_idx = similarities.argsort(reverse=True)[0]
      return docs[top_idx]
  ```
- **Why:** Combines agent reasoning + document understanding.

### Tool Option 7: Data Analysis Tool 📊
- **What:** Agent analyzes CSV/data files
- **Implementation:**
  ```python
  import pandas as pd
  
  def analyze_data(file: str, operation: str) -> str:
      """Analyze CSV data"""
      df = pd.read_csv(file)
      
      if operation == "summary":
          return str(df.describe())
      elif operation == "correlations":
          return str(df.corr())
      elif operation == "missing":
          return str(df.isnull().sum())
  ```
- **Why:** Agents handle data questions (e.g., "What's the average price?").

---

## 🚀 Getting Started Recommendation

**Step 1:** Start with **Option 1** (Math tools)
- Focus on agent logic, not tool complexity
- Quick to debug
- Build confidence

**Step 2:** Add **Option 2** (Web APIs)
- Learn HTTP requests
- Real-world APIs

**Step 3:** Combine with **Option 6** (Document search)
- Now agent has knowledge + reasoning
- Getting closer to production

---

## 🔗 Tool Chaining Example

```
User: "Find the weather in Paris and translate it to Spanish"

Agent reasoning:
1. Need to get Paris weather
2. Then translate result to Spanish

Actions:
1. Call get_weather("Paris") → "Sunny, 72°F"
2. Call translate_text("Sunny, 72°F", "en", "es") → "Soleado, 72°F"
3. Return result to user
```

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
