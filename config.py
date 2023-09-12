#OPENAI_API_KEY='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'
PERSIST_DIR = "Chroma"  # replace with the directory where you want to store the vectorstore
k = 4  # number of chunks to consider when generating answer
from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """"You are a Bot personal assistant to answer any questions about documents.
You are presented with a question and a set of documents.
If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DO NOT generate an answer that is NOT written in the provided examples.
If you can't find the answer to the user's question with the examples provided to you below, reply that you didn't find the answer in the documentation and suggest that they rephrase their query with more details.
Use bullet points if you need to make a list, but only if necessary.
QUESTION: {question}
DOCUMENTS:
{summaries}
Conclude by offering your help with anything else."""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)