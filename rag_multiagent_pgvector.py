from phi.agent import Agent
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.model.ollama import Ollama
from phi.agent import Agent
import ollama
from typing import List


import os
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

def get_embedding(text: str) -> List[float]:
        response = ollama.embeddings(model="nomic-embed-text",prompt=text)
        return response['embedding']

get_embedding.dimensions = 768  # Replace 768 with the actual dimension size if different

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name= "recipes", db_url=db_url,embedder=get_embedding)
)

knowledge_base.load(recreate=False)

agent = Agent(
    model=Ollama(id="llama3.2"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
