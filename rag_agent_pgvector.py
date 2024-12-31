from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector
from phi.model.ollama import Ollama
from phi.agent import Agent
from typing import List
from phi.embedder.ollama import OllamaEmbedder

# Database connection URL
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Load the knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        embedder=OllamaEmbedder(model="nomic-embed-text", dimensions=768)
    )
)
knowledge_base.load(recreate=False)

# Create the RAG agent
agent = Agent(
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,  # Add the knowledge base to the agent
    show_tool_calls=True,
    markdown=True,
)

# Query the agent
agent.print_response("How do I make 'Khao Niew Dam Piek Maphrao Awn' ", stream=True)