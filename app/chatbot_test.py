import json
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import get_settings
from schemas.chat import Message, ChatHistory as ChatHistorySchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize SQLite database
Base = declarative_base()


class ChatHistoryModel(Base):
    """SQLAlchemy model for storing chat history in the database."""
    __tablename__ = "chat_history"

    user_id = Column(String, primary_key=True)
    memory_data = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)


# Create database engine and session
engine = create_engine('sqlite:///E:/Iris/iris.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Initialize embeddings
embed = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL_NAME
)

# Initialize vector store
pinecone_vectorstore = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX_NAME,
    embedding=embed,
    text_key="text"
)

# Initialize LLM
llm = ChatOpenAI(
    model=settings.CHAT_OPENAI_MODEL_NAME,
    temperature=settings.CHAT_OPENAI_TEMPERATURE,
)


def convert_messages_to_schema(messages: List[Any]) -> List[Message]:
    """Convert LangChain messages to our Message schema."""
    schema_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "system"

        schema_messages.append(Message(
            role=role,
            content=msg.content,
            timestamp=datetime.utcnow()
        ))
    return schema_messages


async def get_user_memory(user_id: str) -> ConversationBufferMemory:
    """Retrieve user's conversation memory from SQLite."""
    try:
        session = Session()
        chat_history = session.query(ChatHistoryModel).filter_by(user_id=user_id).first()
        
        if chat_history and chat_history.memory_data:
            memory_dict = json.loads(chat_history.memory_data)
            memory = ConversationBufferMemory(return_messages=True)
            # Load the messages directly into chat_memory
            for msg in memory_dict.get("messages", []):
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])
            session.close()
            return memory
            
        session.close()
        return ConversationBufferMemory(return_messages=True)
    except Exception as e:
        logger.error(f"Error retrieving memory for user {user_id}: {str(e)}")
        return ConversationBufferMemory(return_messages=True)


async def save_user_memory(user_id: str, memory: ConversationBufferMemory):
    """Save user's conversation memory to SQLite."""
    try:
        session = Session()
        
        # Convert messages to a serializable format
        memory_dict = {
            "messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in memory.chat_memory.messages
            ]
        }
        
        memory_data = json.dumps(memory_dict)
        
        chat_history = session.query(ChatHistoryModel).filter_by(user_id=user_id).first()
        if chat_history:
            chat_history.memory_data = memory_data
            chat_history.updated_at = datetime.utcnow()
        else:
            chat_history = ChatHistoryModel(
                user_id=user_id,
                memory_data=memory_data
            )
            session.add(chat_history)
            
        session.commit()
        session.close()
    except Exception as e:
        logger.error(f"Error saving memory for user {user_id}: {str(e)}")


async def get_chat_history(user_id: str) -> ChatHistorySchema:
    """Get chat history in schema format."""
    try:
        memory = await get_user_memory(user_id)
        messages = convert_messages_to_schema(memory.chat_memory.messages)
        return ChatHistorySchema(
            user_id=user_id,
            messages=messages,
            updated_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return ChatHistorySchema(user_id=user_id)


# System prompts
CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question which can be understood \
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are an intelligent assistant that helps users by answering their questions based on the provided context and conversation history.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.
Keep your answers concise, informative, and conversational. If the question is not related to the provided context, you can use your general knowledge to answer.
Always maintain a helpful and friendly tone.

Context:
{context}

Chat History:
{chat_history}

Question: {input}

Answer:"""

# Initialize prompts
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", "{input}"),
])

# Initialize chains
contextualize_chain = contextualize_prompt | llm
qa_chain = qa_prompt | llm

# Initialize retriever
vector_retriever = pinecone_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


async def process_chat(user_input: Dict[str, Any], user_id: str) -> str:
    """Process user input with chat history and context awareness."""
    try:
        # Get user's memory
        memory = await get_user_memory(user_id)

        # Get chat history
        chat_history = memory.chat_memory.messages

        # Rephrase question if there's chat history
        if chat_history:
            question = await contextualize_chain.ainvoke({
                "chat_history": chat_history,
                "input": user_input["input"]
            })
            # Extract content from AIMessage
            question = question.content if isinstance(question, AIMessage) else str(question)
        else:
            question = user_input["input"]

        # Retrieve relevant context
        context = await vector_retriever.ainvoke(question)
        print('question: ', question)
        print('context: ', context)

        # Generate response
        response = await qa_chain.ainvoke({
            "input": user_input["input"],
            "context": context,
            "chat_history": chat_history
        })

        # Extract content from AIMessage
        response_text = response.content if isinstance(response, AIMessage) else str(response)

        # Update memory
        memory.chat_memory.add_user_message(user_input["input"])
        memory.chat_memory.add_ai_message(response_text)

        await save_user_memory(user_id, memory)

        return response_text
    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again."


class Iris:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_id = config["user_id"]

    async def chat(self, message: str) -> str:
        """Process a chat message and return a response."""
        try:
            response = await process_chat(
                {"input": message},
                self.user_id
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    async def get_history(self) -> ChatHistorySchema:
        """Get chat history for the current user."""
        return await get_chat_history(self.user_id)

chat_test = Iris({'user_id': "crystal1701"})
response = asyncio.run(chat_test.chat('Do you know my name?'))
pass
