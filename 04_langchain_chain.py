import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful backend engineering tutor."),
    ("user", "{question}")
])

chain = prompt | llm

response = chain.invoke({"question": "What's the difference between SQL and NoSQL?"})
print(response.content)