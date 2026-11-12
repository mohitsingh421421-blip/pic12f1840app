import gradio as gr
from getpass import getpass
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate   # ← corrected import

api_key = getpass("Enter your Groq API key: ")

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0.1
)

# Better prompt for technical accuracy
prompt_template = """You are a precise technical assistant for microcontroller datasheets.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I could not find this information in the datasheet."

Context:
{context}

Question: {question}

Instructions:
- Be clear and accurate
- Use bullet points when listing features or steps
- Mention exact values, pin numbers, or register names when available
- Keep the answer focused and professional

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

def chat_function(message, history):
    if not message.strip():
        return "Please type a question."
    
    result = qa_chain.invoke({"query": message})
    answer = result["result"]
    
    # Format sources
    sources = []
    for i, doc in enumerate(result["source_documents"], 1):
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "Unknown")
        source_name = source.split("/")[-1] if "/" in source else source
        page_num = int(page) + 1 if isinstance(page, int) else page
        sources.append(f"{i}. {source_name} — Page {page_num}")
    
    sources_text = "\n".join(sources) if sources else "No sources found."
    
    full_response = f"{answer}\n\n**Sources:**\n{sources_text}"
    return full_response

demo = gr.ChatInterface(
    fn=chat_function,
    title="PIC12F1840 Datasheet Assistant",
    description="Technical answers with sources from the datasheet",
    examples=[
        "What is the operating voltage range of the PIC12F1840?",
        "How many ADC channels does it have and what is the resolution?",
        "Explain the DAC module",
        "What are the functions of pin RA2?"
    ]
)

demo.launch(share=True)
