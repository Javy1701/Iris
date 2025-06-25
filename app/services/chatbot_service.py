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


def get_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'prompt.txt')
    prompt_path = os.path.abspath(prompt_path)
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    return _current_system_prompt


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
    from the database. Use this tool to get data for color comparisons. Only run this function when user asked to compare two colors, and if user wants to compare more than 3 colors, do not run this.
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