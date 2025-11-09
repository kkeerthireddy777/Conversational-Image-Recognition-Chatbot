from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
from config import Config

class ConversationState(TypedDict):
    """State for conversation graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    image_context: str

class ConversationalLLM:
    def __init__(self):
        """Initialize LLM with LangGraph memory"""
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Initialize memory saver
        self.memory = MemorySaver()
        
        # System prompt template
        self.system_prompt = """You are an intelligent image analysis assistant. You have access to detailed information about an image including object detection data and image descriptions.

Current Image Analysis:
{image_context}

Instructions:
- Provide natural, conversational responses
- Use the object detection data to answer spatial questions (where, how many, what position)
- Use the image caption for overall scene understanding
- Be precise when answering "where" questions using the position information
- If asked about objects not detected, politely say they're not visible in the image
- Keep responses concise but informative
- Maintain conversation context from previous messages"""
        
        # Build conversation graph
        self.graph = self._build_graph()
        self.thread_id = "conversation_1"
        
    def _build_graph(self):
        """Build LangGraph for conversation with memory"""
        
        def chatbot_node(state: ConversationState):
            """Process messages through LLM"""
            # Get image context
            image_context = state.get("image_context", "No image context available.")
            
            # Build messages with system prompt
            system_msg = SystemMessage(
                content=self.system_prompt.format(image_context=image_context)
            )
            
            # Get conversation history (limit to last 10 messages for context window)
            messages = state["messages"][-Config.MAX_CONVERSATION_HISTORY:]
            
            # Combine system message with conversation
            full_messages = [system_msg] + list(messages)
            
            # Get LLM response
            response = self.llm.invoke(full_messages)
            
            return {"messages": [response]}
        
        # Create graph
        workflow = StateGraph(ConversationState)
        workflow.add_node("chatbot", chatbot_node)
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)
    
    def generate_response(self, user_query: str, image_context: str) -> str:
        """
        Generate conversational response using image context and chat history
        """
        # Create user message
        user_message = HumanMessage(content=user_query)
        
        # Invoke graph with memory
        config = {"configurable": {"thread_id": self.thread_id}}
        
        result = self.graph.invoke(
            {
                "messages": [user_message],
                "image_context": image_context
            },
            config=config
        )
        
        # Extract AI response
        ai_response = result["messages"][-1].content
        return ai_response
    
    def reset_memory(self):
        """Clear conversation history by creating new thread"""
        import uuid
        self.thread_id = f"conversation_{uuid.uuid4().hex[:8]}"
    
    def get_conversation_history(self) -> list:
        """Get current conversation history"""
        config = {"configurable": {"thread_id": self.thread_id}}
        
        try:
            # Get state from memory
            state = self.graph.get_state(config)
            messages = state.values.get("messages", [])
            
            # Format messages
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
        except:
            return []