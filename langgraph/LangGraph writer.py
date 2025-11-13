#!/usr/bin/env python
# coding: utf-8

# # Lesson 6: Essay Writer

# In[ ]:


from dotenv import load_dotenv

_ = load_dotenv()


# In[ ]:

import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from pydantic.v1 import BaseModel
from IPython.display import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import warnings
from tavily import TavilyClient
from langchain_openai import ChatOpenAI


warnings.filterwarnings("ignore")

# memory = SqliteSaver.from_conn_string(":memory:")


# In[ ]:


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# In[ ]:




# Setup model

genai.configure(api_key=os.environ["OPENAI_API_KEY"])
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# In[ ]:


PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an SEO Optimized marketing blog. \
Write such an outline for the user provided topic. Give an outline of the blog along with any relevant notes \
or instructions for the headings (H1, H2, etc)."""


# In[ ]:


WRITER_PROMPT = """You are a marketing copy expert tasked with writing excellent SEO optimized blog with proper headings.\
Generate the best possible blog for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""


# In[ ]:


REFLECTION_PROMPT = """You are an expert marketing blog writer grading an marketing blog submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, SEO optimization, keyword selections headings etc."""


# In[ ]:


RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following marketing blog. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""


# In[ ]:


RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


# In[ ]:

# 
# from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]


# In[ ]:


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# In[ ]:


def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


# In[ ]:


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[ ]:


def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }


# In[ ]:


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


# In[ ]:


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[ ]:


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# In[ ]:


builder = StateGraph(AgentState)


# In[ ]:


builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)


# In[ ]:


builder.set_entry_point("planner")


builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

# In[ ]:


builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)


# In[ ]:



# In[ ]:


# graph = builder.compile(checkpointer=memory)


# In[ ]:


# from IPython.display import Image

# Image(graph.get_graph().draw_png())


# In[ ]:


if __name__ == '__main__':
    try:
        with SqliteSaver.from_conn_string(":memory:") as memory:
            graph = builder.compile(checkpointer=memory)
            # Image(graph.get_graph().draw_png())
            # builder.to_graph().draw(format="png").write("graph.png")
            
            
            # # Visualize the graph structure
            # try:
            #     import graphviz
            #     # Create graph visualization
            #     dot = graphviz.Digraph(comment='Essay Writer Flow')
            #     dot.attr(rankdir='LR')  # Left to right layout
                
            #     # Add nodes
            #     for node in ["planner", "research_plan", "generate", "reflect", "research_critique"]:
            #         dot.node(node)
                
            #     # Add edges
            #     dot.edge("planner", "research_plan")
            #     dot.edge("research_plan", "generate")
            #     dot.edge("reflect", "research_critique")
            #     dot.edge("research_critique", "generate")
            #     dot.edge("generate", "reflect")
                
            #     # Save the graph
            #     dot.render("essay_writer_graph", format="png", cleanup=True)
            #     print("Graph visualization saved as 'essay_writer_graph.png'")
            # except Exception as graph_error:
            #     print(f"Could not create graph visualization: {str(graph_error)}")
            
            
            thread = {"configurable": {"thread_id": "1"}}
            input_data = {
                'task': "What is the difference between LangChain and LangSmith?",
                'max_revisions': 2,
                'revision_number': 1,
                'content': [],
                'plan': "",
                'draft': "",
                'critique': ""
            }
            

            for step in graph.stream(input_data, thread):
                for key, value in step.items():
                    print(f"\n{key}:")
                    print(value)
                            
    except Exception as e:
        print(f"Error occurred: {str(e)}")


