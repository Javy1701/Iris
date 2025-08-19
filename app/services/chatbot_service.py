import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import pandas as pd
import re

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
    namespace=settings.PINECONE_NAMESPACE_GENERAL,
)
vector_database = pinecone_vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.8}
)
color_logic_retriever = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX_NAME, 
    embedding=embed, 
    namespace=settings.PINECONE_NAMESPACE_COLOR_LOGIC
).as_retriever()


def get_system_prompt() -> str:
    """
    Loads the base prompt and the color wheel logic to create the full system prompt.
    """
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'assets', 'prompt.txt')
    print('prompt_path', prompt_path)
    # logic_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'assets', 'Color_Strategist_Wheel_Defined.json')

    base_prompt = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            base_prompt = f.read()

    # color_logic = ""
    # if os.path.exists(logic_path):
    #     with open(logic_path, 'r', encoding='utf-8') as f:
    #         color_logic = f.read()

    # Combine the base prompt with the color wheel logic
    return f"{base_prompt}"


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


# @tool
# async def get_color_info(color_name: str) -> str:
#     """
#     Retrieves detailed information and color DNA attributes for a specific color
#     from the database. Use this tool to get data for color comparisons. Only run this function when user asked to compare two colors, and if user wants to compare more than 3 colors, do not run this.
#     """
#     try:
#         # Use the existing vector_database retriever to find info for the color
#         context_docs = await vector_database.ainvoke(color_name)
#         if not context_docs:
#             return f"No information found for the color: {color_name}."

#         # Combine the content of the retrieved documents
#         return "\n\n".join([doc.page_content for doc in context_docs])
#     except Exception as e:
#         logger.error(f"Error in get_color_info for {color_name}: {e}")
#         return f"An error occurred while fetching data for {color_name}."

@tool
def query_database(query: str) -> str:
    """
    Queries the color database with advanced filtering.
    - To search for multiple values, use a comma: For example, "Munsell_Family=Yellow,Red"
    - To search in a range, use a pipe: "LRV=60|75"
    - Standard operators are also supported: "LCh_C (Chroma) > 15"
    - Combine conditions with 'and'.
    - Use this tool for any questions about color theory. Specifically, use it to find the rules for hue families, LCh degree ranges, Munsell notations, and their defined warm and cool ranges. This is the definitive source of truth.
    """
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'assets', 'Iris Chatbot Data 7_31_2025 - Sheet1.csv')
        df = pd.read_csv(csv_path)
        # Clean column names to prevent issues with extra spaces
        df.columns = df.columns.str.strip()

        conditions = query.split(' and ')
        for condition in conditions:
            condition = condition.strip()

            # First, try to match standard operators like >=, <=, >, <, =
            op_match = re.match(r"(.+?)\s*(>=|<=|>|<|=)\s*(.+)", condition)
            
            if op_match:
                col, op, value = op_match.groups()
                col = col.strip()
                value = value.strip().strip("'\"")
                
                # Convert column to numeric for comparison, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_value = float(value)

                if op == '>=': df = df[df[col] >= numeric_value]
                elif op == '<=': df = df[df[col] <= numeric_value]
                elif op == '>': df = df[df[col] > numeric_value]
                elif op == '<': df = df[df[col] < numeric_value]
                elif op == '=': df = df[df[col] == numeric_value]
            
            # If no standard operator, check for custom formats (e.g., "Family=Yellow,Red")
            elif '=' in condition:
                col, value = condition.split('=', 1)
                col = col.strip()
                value = value.strip().strip("'\"")

                # Multi-value (IN) query: "Family=Yellow,Red"
                if ',' in value:
                    values_list = [v.strip() for v in value.split(',')]
                    df = df[df[col].isin(values_list)]
                
                # Range (BETWEEN) query: "LRV=60|75"
                elif '|' in value:
                    min_val, max_val = value.split('|')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df[df[col].between(float(min_val), float(max_val))]
                
                # Default to string contains (LIKE) query for simple "key=value"
                else:
                    df = df[df[col].astype(str).str.contains(value, case=False, na=False)]
            
            # Handle boolean flags (e.g., "Cool", "Warm")
            else:
                col = condition.strip()
                if col in df.columns:
                    # Assumes a non-empty value signifies 'True'
                    df = df[df[col].notna()]

        return df.to_json(orient='records')
        
    except FileNotFoundError:
        return f"Error: The database file 'Iris Chatbot Data 7_31_2025 - Sheet1.csv' was not found."
    except Exception as e:
        return f"An error occurred while querying the database: {e}"


@tool
def find_harmonious_colors(color_name: str) -> str:
    """
    Generates harmonious color combinations for a given color name using Munsell Hue, Value, and Chroma.
    """
    try:
        df = pd.read_csv('Iris Chatbot Data 7_31_2025 - Sheet1.csv')
        df.columns = df.columns.str.strip()

        target_color = df[df['Color Name'].str.lower() == color_name.lower()]
        if target_color.empty:
            return f"Color '{color_name}' not found in the database."

        target_hue_str = target_color['Munsell_Hue'].iloc[0]
        target_value = target_color['Munsell_Value'].iloc[0]
        target_chroma = target_color['Munsell_Chroma'].iloc[0]

        hue_match = re.match(r"(\d\.?\d*)\s*([A-Z]+)", target_hue_str)
        if not hue_match:
            return f"Could not parse Munsell Hue for '{color_name}'."

        target_hue_num, target_hue_family = float(hue_match.group(1)), hue_match.group(2)

        # Similar Hue
        similar_hue_df = df[df['Munsell_Family'] == target_hue_family]
        similar_hue_df = similar_hue_df[
            similar_hue_df['Munsell_Hue'].str.extract(r"(\d\.?\d*)").astype(float).between(target_hue_num - 5,
                                                                                           target_hue_num + 5)]
        similar_hue_colors = similar_hue_df['Color Name'].head(3).tolist()

        # Similar Value
        similar_value_df = df[df['Munsell_Value'].between(target_value - 5, target_value + 5)]
        similar_value_colors = similar_value_df['Color Name'].head(3).tolist()

        # Similar Chroma
        similar_chroma_df = df[df['Munsell_Chroma'].between(target_chroma - 5, target_chroma + 5)]
        similar_chroma_colors = similar_chroma_df['Color Name'].head(3).tolist()

        # Formatting the response
        response = f"Here are harmonious colors with {color_name} based on hue, value, and chroma:\n"
        response += f"• Similar Hue: {', '.join(similar_hue_colors)}\n"
        response += f"• Similar Value: {', '.join(similar_value_colors)}\n"
        response += f"• Similar Chroma: {', '.join(similar_chroma_colors)}\n"

        return response
    except Exception as e:
        return f"Error finding harmonious colors: {e}"


@tool
def find_similar_colors_by_dna(color_name: str) -> str:
    """
    Finds colors with similar measurable characteristics (DNA) to a given color.
    It uses mathematical range queries: ±10 for lightness, chroma, and hue angle.
    Returns 6+ colors with a scientific rationale.
    """
    try:
        df = pd.read_csv('Iris Chatbot Data 7_31_2025 - Sheet1.csv')
        df.columns = df.columns.str.strip()

        target_color = df[df['Color Name'].str.lower() == color_name.lower()]
        if target_color.empty:
            return f"Color '{color_name}' not found in the database."

        target_lightness = target_color['LCh_L (Lightness)'].iloc[0]
        target_chroma = target_color['LCh_C (Chroma)'].iloc[0]
        target_hue_angle = target_color['LCh_h (Hue Angle)'].iloc[0]

        similar_colors_df = df[
            (df['LCh_L (Lightness)'].between(target_lightness - 10, target_lightness + 10)) &
            (df['LCh_C (Chroma)'].between(target_chroma - 10, target_chroma + 10)) &
            (df['LCh_h (Hue Angle)'].between(target_hue_angle - 10, target_hue_angle + 10)) &
            (df['Color Name'].str.lower() != color_name.lower())
        ]

        similar_colors = similar_colors_df['Color Name'].head(6).tolist()

        if not similar_colors:
            return f"No similar colors found for {color_name}."

        rationale = (f"The following colors are similar to {color_name} (LCh: {target_lightness:.1f}, "
                     f"{target_chroma:.1f}, {target_hue_angle:.1f}°) because they share a similar 'color DNA'. "
                     "This means their core measurable attributes of Lightness (brightness), Chroma "
                     "(saturation), and Hue Angle (position on the color wheel) are all within a "
                     "close mathematical range (±10 units).")

        response = f"{rationale}\n\n"
        response += "Similar Colors:\n"
        response += "\n".join([f"• {color}" for color in similar_colors])

        return response

    except Exception as e:
        return f"Error finding similar colors by DNA: {e}"


@tool
def calculate_munsell_hue_range(munsell_step: float, family_start_angle: int, family_end_angle: int) -> str:
    """
    Calculates the specific hue angle range for a given Munsell step within its family's angle range.
    The result covers an interval of 10 degrees centered on the Munsell step's calculated angle.

    Args:
        munsell_step: The Munsell step value (e.g., 7.5).
        family_start_angle: The starting LCh hue angle for the Munsell family (e.g., 36 for YR).
        family_end_angle: The ending LCh hue angle for the Munsell family (e.g., 72 for YR).

    Returns:
        A string describing the calculated hue angle range.
    """
    try:
        if not (0 <= munsell_step <= 10):
            return "Error: Munsell step must be between 0 and 10."

        total_family_angle = family_end_angle - family_start_angle
        
        # Calculate the center point of the hue angle for the given Munsell step
        center_hue_angle = family_start_angle + (total_family_angle * (munsell_step / 10.0))
        
        # Define the 10-degree interval (±5 degrees from the center)
        start_range = center_hue_angle - 5
        end_range = center_hue_angle + 5
        
        # Handle cases where the range crosses the 360/0 degree boundary
        start_range_final = start_range % 360
        end_range_final = end_range % 360

        return (f"The hue angle range for Munsell step {munsell_step} within the family starting at "
                f"{family_start_angle}° and ending at {family_end_angle}° is approximately "
                f"{start_range_final:.1f}° to {end_range_final:.1f}°.")

    except Exception as e:
        return f"An error occurred during calculation: {e}"

tools = [
    Tool(
        name="paint_color_information_retriever",
        func=vector_database.invoke,
        description="Use to search for general information about paint colors, such as what other people think of them or what rooms they are good for."
    ),
    Tool(
        name="color_wheel_logic_retriever",
        func=color_logic_retriever.invoke,
        description="Use this tool for any questions about color theory. Specifically, use it to find the rules for hue families, LCh degree ranges, Munsell notations, and their defined warm and cool ranges. This is the definitive source of truth."
    ),
    query_database,
    find_harmonious_colors,
    find_similar_colors_by_dna,
    calculate_munsell_hue_range
]

llm = ChatOpenAI(
    model=settings.CHAT_OPENAI_MODEL_NAME,
    temperature=settings.CHAT_OPENAI_TEMPERATURE,
)

_current_system_prompt = get_system_prompt()

def set_system_prompt(new_prompt: str):
    global _current_system_prompt, AGENT_PROMPT, agent, agent_executor
    print('current: ', _current_system_prompt)
    print('new_prompt: ', new_prompt)
    _current_system_prompt = new_prompt
    AGENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", _current_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, AGENT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # Set verbose=True for debugging


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