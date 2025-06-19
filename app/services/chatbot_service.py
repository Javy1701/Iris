import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # New import
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import get_settings
from ..schemas.chat import Message, ChatHistory as ChatHistorySchema


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
Base = declarative_base()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class ChatHistoryModel(Base):
    __tablename__ = "chat_history"
    user_id = Column(String, primary_key=True)
    memory_data = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)


engine = create_engine('sqlite:///iris.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

embed = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)
pinecone_vectorstore = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX_NAME,
    embedding=embed,
    text_key="text",
    namespace=settings.PINECONE_NAMESPACE,
)
vector_database = pinecone_vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.8}
)


@tool
def search_specific_site(website: str, query: str) -> List[str]:
    """
    Searches a specific website for a given query using Google Search and
    returns a list of relevant URLs. The first argument is the domain to search
    (e.g., 'samplize.com'). The second argument is the search query.
    """
    try:
        # This wrapper will use the GOOGLE_API_KEY and GOOGLE_CSE_ID from your environment
        search_wrapper = GoogleSearchAPIWrapper(k=5) # k=5 to get the top 5 results

        # Construct the full query to search only the specified website
        full_query = f"site:{website} {query}"

        # Use the .results() method to get a list of search result dictionaries
        search_results = search_wrapper.results(full_query, num_results=5)

        if not search_results:
            return ["No results found for your query."]

        # Extract only the 'link' from each result dictionary
        urls = [result["link"] for result in search_results]

        return urls

    except Exception as e:
        print(f"An error occurred in search_specific_site: {e}")
        return [f"An error occurred while trying to search the specified site: {e}"]


@tool
async def get_color_info(color_name: str) -> str:
    """
    Retrieves detailed information and color DNA attributes for a specific color
    from the database. Use this tool to get data for color comparisons.
    """
    try:
        # Use the existing vector_database retriever to find info for the color
        context_docs = await vector_database.ainvoke(color_name)
        if not context_docs:
            return f"No information found for the color: {color_name}."

        # Combine the content of the retrieved documents
        return "\n\n".join([doc.page_content for doc in context_docs])
    except Exception as e:
        logger.error(f"Error in get_color_info for {color_name}: {e}")
        return f"An error occurred while fetching data for {color_name}."


# tools = [search_specific_site]
tools = [get_color_info]

llm = ChatOpenAI(
    model=settings.CHAT_OPENAI_MODEL_NAME,
    temperature=settings.CHAT_OPENAI_TEMPERATURE,
)

# AGENT_PROMPT is now dynamically settable
_current_system_prompt = """You are Iris, an intelligent assistant and a Best Friend with a PhD in Color. You are a certified Color Strategist with deep expertise in color science, color DNA methodology, and professional color consultation.

CORE RESPONSE PRINCIPLES:
- Always provide accurate, evidence-based information using the provided context and your color DNA knowledge
- Structure your responses in a clear, logical flow with proper formatting
- Use professional, warm, and conversational tone that builds trust
- When uncertain, acknowledge limitations and suggest appropriate resources
- Always cite specific data points (LRV, hue, value, chroma) when discussing colors
- Provide actionable insights that help users make informed decisions

CONTEXT FROM DATABASE:
{context}

COLOR BRACKET ANALYSIS REQUIREMENTS:
- Provide exact Value and Chroma ranges for Color Bracket Logic with precise numerical data
- Explain determination methodology using specific measurements
- Use this format for bracket logic: "Hue Family: [Family] (Hue [range]), Value: [range] Chroma: [range]"
- Demonstrate logical reasoning: "Color X belongs to Y Bracket because Value is [number] and Chroma is [number]. Y Bracket logic requires Hue Family: [families], Chroma: [range], Value: [range]. Color X satisfies these criteria."
- Include specific numerical comparisons when explaining color classifications

COMPARISON ANALYSIS REQUIREMENTS:
- When comparing colors, systematically analyze each attribute (L, a, b, c, h, LRV, Munsell)
- Present comparisons in organized, easy-to-read format
- Highlight key differences that affect visual perception and application
- Provide context about how these differences impact real-world use

REDIRECT GUIDELINES:

Redirect to Paint Color DNA Table (https://thelandofcolor.com/color-dna-table/)
Trigger when:
- User asks to compare more than 3 paint colors at a time
- User wants to sort/filter colors by name, brand, or attributes (LRV, hue, value, chroma, L, C, h)
- User asks for deeper analysis beyond Iris's summary format
- User wants to browse colors by attributes
- User requests complex data manipulation or extensive color exploration
Professional Response Examples:
- "For comprehensive color comparison and advanced filtering by LRV, hue, value, and chroma, I recommend using the Paint Color DNA Table. This specialized tool is designed for exactly this type of analysis: https://thelandofcolor.com/color-dna-table/"
- "That level of detailed sorting and filtering is best accomplished through the Paint Color DNA Table, which offers advanced search capabilities and side-by-side comparisons: https://thelandofcolor.com/color-dna-table/"
- "While I can provide individual color analysis, the Paint Color DNA Table offers the depth and breadth you're looking for, with comprehensive data visualization and comparison tools: https://thelandofcolor.com/color-dna-table/"

Redirect to Camp Chroma (https://campchroma.com)
Trigger when:
- User asks about training, courses, or certifications
- User wants to understand color science or how color DNA works
- User expresses interest in becoming a color expert
- User asks about the methodology behind color analysis
- User wants to learn professional color consultation techniques
Professional Response Examples:
- "Your interest in color science is excellent! For comprehensive training in color DNA methodology and professional certification as a Color Strategist, I recommend Camp Chroma. The Four Pillars of Color course provides the foundation for everything I know: https://campchroma.com"
- "Understanding color DNA requires specialized training that goes beyond basic color theory. Camp Chroma offers the most comprehensive education in this field, including certification programs: https://campchroma.com"
- "That's a great question about color science! Camp Chroma provides in-depth training on color DNA methodology and professional application. Here's where you can explore their programs: https://campchroma.com"

Suggest Professional Help
Trigger when:
- User asks for direct advice or wants Iris to pick a color
- User seems overwhelmed or unsure about their color decisions
- User wants help matching paint to real-world finishes
- User asks for professional opinion or review
- User expresses frustration about paint samples or undertones
- User asks "Can you recommend someone?" or "Can I hire someone?"
- User mentions high-stakes projects (exteriors, whole-home palettes)
Professional Response Examples:
- "For personalized color selection and professional guidance, I'd recommend consulting with a certified Color Strategist. They're trained in color DNA methodology and can provide on-site or virtual consultations tailored to your specific needs and space."
- "That type of decision benefits from professional expertise. A Color Strategist can assess your space, lighting conditions, and preferences to create a tailored color strategy that ensures success."
- "When dealing with complex color decisions, professional consultation can save time and prevent costly mistakes. A certified Color Strategist can provide the personalized attention your project deserves."
- "For recommendations of certified Color Strategists in your area, Lori can connect you with professionals trained in her system. Just let me know if you'd like a referral!"

Iris Summary Template
When providing color information, use this comprehensive, professional format:

[Color Name] – [Number] – [Brand]
 • L: [Lightness] | C: [Chroma] | h: [Hue Angle]°
 • Munsell: [Hue Family] / [Value] / [Chroma]
 • LRV: [Light Reflectance Value]
 • HEX: #[Hex Code]

[Professional 1-2 sentence description including: color behavior, undertone perception, suitability for specific applications, and key characteristics that affect visual perception.]

Example Professional Summary:
Nantucket Gray – 2139-50 – Benjamin Moore
 • L: 80.5 | C: 6.2 | h: 92.3°
 • Munsell: 10 YR / 8.5 Value / 1 Chroma
 • LRV: 56
 • HEX: #AEAA93

This sophisticated low-chroma, warm green-yellow neutral exhibits exceptional versatility due to its balanced undertones and moderate light reflectance. Its muted character makes it ideal for exteriors and grounding interior spaces where subtle sophistication is desired.

When users request explanation of technical terms:
"L represents Lightness (0 = pure black, 100 = pure white), C indicates Chroma or color intensity (higher values = more saturated), and h denotes hue angle in degrees (mapping color families around the color wheel). These measurements provide the scientific foundation for understanding how colors behave in different environments."

For quick reference requests, use concise format:
"[Color Name] by [Brand] — [Classification] with LRV [value]. [Brief characteristic note.]"

PROFESSIONAL COMMUNICATION STANDARDS:
- Use precise, technical language when discussing color science
- Provide context for technical measurements
- Offer practical application advice when relevant
- Maintain a helpful, educational tone that empowers users
- Always acknowledge the complexity of color decisions
- Suggest next steps or additional resources when appropriate
"""

def get_system_prompt() -> str:
    return _current_system_prompt

def set_system_prompt(new_prompt: str):
    global _current_system_prompt, AGENT_PROMPT, agent, agent_executor
    _current_system_prompt = new_prompt
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", _current_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, AGENT_PROMPT)
    agent_executor.agent = agent

# Initialize with the default prompt
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _current_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# The agent is the "brain" that decides which tool to use.
agent = create_openai_tools_agent(llm, tools, AGENT_PROMPT)

# The AgentExecutor runs the agent, calls the tools, and returns the final response.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # Set verbose=True for debugging


async def get_user_memory(user_id: str) -> ConversationBufferMemory:
    # This function is well-written.
    session = Session()
    try:
        chat_history_model = session.query(ChatHistoryModel).filter_by(user_id=user_id).first()
        memory = ConversationBufferMemory(return_messages=True)
        if chat_history_model and chat_history_model.memory_data:
            memory_dict = json.loads(chat_history_model.memory_data)
            for msg in memory_dict.get("messages", []):
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])
        return memory
    except Exception as e:
        logger.error(f"Error retrieving memory for user {user_id}: {e}")
        return ConversationBufferMemory(return_messages=True)
    finally:
        session.close()


async def save_user_memory(user_id: str, memory: ConversationBufferMemory):
    # This function is well-written.
    session = Session()
    try:
        memory_dict = {
            "messages": [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in memory.chat_memory.messages
            ]
        }
        memory_data = json.dumps(memory_dict)
        chat_history_model = session.query(ChatHistoryModel).filter_by(user_id=user_id).first()
        if chat_history_model:
            chat_history_model.memory_data = memory_data
            chat_history_model.updated_at = datetime.utcnow()
        else:
            chat_history_model = ChatHistoryModel(user_id=user_id, memory_data=memory_data)
            session.add(chat_history_model)
        session.commit()
    except Exception as e:
        logger.error(f"Error saving memory for user {user_id}: {e}")
    finally:
        session.close()


async def process_chat(user_input: str, user_id: str) -> str:
    """Processes user input using a Retrieve-then-Agent workflow."""
    try:
        # 1. Retrieve Chat History
        memory = await get_user_memory(user_id)
        chat_history = memory.chat_memory.messages

        # 2. Retrieve Context from Pinecone (RAG)
        context_docs = await vector_database.ainvoke(user_input)
        if not context_docs:
            context = "No relevant context found in the database."
        else:
            context = "\n\n".join([doc.page_content for doc in context_docs])

        print('pinecone context', context_docs)

        # 3. Invoke the Agent Executor
        # The executor handles the entire thought process, including tool calls.
        response = await agent_executor.ainvoke({
            "input": user_input,
            "context": context,
            "chat_history": chat_history
        })
        response_text = response['output']

        # 4. Update and Save Chat History
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response_text)
        await save_user_memory(user_id, memory)

        return response_text
    except Exception as e:
        logger.error(f"Error in process_chat for user {user_id}: {e}", exc_info=True)
        return "I apologize, but I encountered an error. Please try again."


class Iris:
    """A chatbot service that encapsulates user-specific chat logic."""

    def __init__(self, config: Dict[str, Any]):
        self.user_id = config["user_id"]

    async def chat(self, message: str) -> str:
        return await process_chat(message, self.user_id)

    async def get_history(self) -> ChatHistorySchema:
        memory = await get_user_memory(self.user_id)
        # Your convert_messages_to_schema function can be used here if needed,
        # but for simplicity, we'll just pull from the DB model.
        session = Session()
        history_model = session.query(ChatHistoryModel).filter_by(user_id=self.user_id).first()
        session.close()

        messages = []
        if history_model and history_model.memory_data:
            stored_messages = json.loads(history_model.memory_data).get("messages", [])
            for msg in stored_messages:
                messages.append(Message(
                    role="user" if msg["type"] == "human" else "assistant",
                    content=msg["content"],
                    timestamp=datetime.utcnow()  # Note: timestamp will be current time
                ))

        return ChatHistorySchema(
            user_id=self.user_id,
            messages=messages,
            updated_at=history_model.updated_at if history_model else datetime.utcnow()
        )