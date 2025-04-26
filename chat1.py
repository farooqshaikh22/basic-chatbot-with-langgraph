import os
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", api_key=api_key)

## define state

class State(TypedDict):
    messages: List[Dict[str, str]]
    
## initialize stategraph

graph_builder = StateGraph(State)

## define chatbot function

def chatbot(state: State):
    
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response.content})
    return {"messages": state["messages"]}

## Add nodes and edges

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

## compile the graph

graph = graph_builder.compile()

## stream update

def stream_graph_update(user_input: str):
    
    query = {"messages": [{"role": "user", "content": user_input}]}
    
    for event in graph.stream(query):
        # print("event", event)
        for value in event.values():
            # print("value", value)
            print("Assistant:", value["messages"][-1]["content"])
            
## Run chatbot in loop

if __name__ == "__main__":
    
    while True:
        
        try:
            user_input = input("User: ")
            
            if user_input.lower() in ["bye","exit", "quit", "q"]:
                print("goodbye")
                break
            
            stream_graph_update(user_input)
            
        except Exception as e:
            print(f"An error occured : {e}")
            break
