from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st
from dotenv import load_dotenv
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"]="Enter your Hugging Face API token here"
# or
# load_dotenv()  # If you have a .env file with the token
prompt = PromptTemplate(
    input_variables=["text","category"],
    template="You are a helpful and expert summarizer. Given the following text, produce a {category} summary. Ensure the summary reflects the core style and intent of the {category} summarization type. Focus on clarity, coherence, and accuracy and make sure to cover every content present in the following text:\n{text}",
)
output_parser = StrOutputParser()
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    device_map="auto",          
    low_cpu_mem_usage=True       
)

tp = pipeline("summarization", model=model, tokenizer=tokenizer) 

model = HuggingFacePipeline(pipeline=tp)
chain = prompt | model | output_parser

def summarize_text(text: str, sumar) -> str:
    """Summarize the given text using a Hugging Face model."""
    return chain.invoke({"text": text,"category": sumar}) 

st.title("Text Summarization App")
summarization_options = ['narrative', 'informative', 'executive', 'abstract', 'thematic', 'critical', 'descriptive', 'synoptic']
summarization_type = st.selectbox("Select summarization type:", summarization_options)
text_input = st.text_area("Enter text to summarize:", height=300)   
if st.button("Summarize"):
    if text_input:
        summary = summarize_text(text_input,sumar=summarization_type)    
        st.subheader("Summary:")
        st.write(summary)