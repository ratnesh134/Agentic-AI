import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize LLM (Groq + LLaMA3)
llm = ChatOpenAI(
    openai_api_key=os.environ.get("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    model="llama3-8b-8192"
)

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant that will help generate a research paper.
Answer the user query and wrap the output in this format and provide no other text:
{format_instructions}"""
        ),
        ("human", "{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

# Build chain
chain = prompt | llm | parser

user_query = input("Enter your query : ")

# Invoke
response = chain.invoke({"query": user_query})
print("\n Response \n")
print(response)

