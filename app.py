import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.docstore.document import Document

# Page configuration
st.set_page_config(
    page_title="Text Summarization",
    page_icon="📚",
    layout="wide"
)

# Sidebar for API key
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    st.caption("Get your API key from Groq platform")

# Main title
st.title("🤖 Langchain: Summarize Text From YT or Website")
st.markdown("This app uses Langchain and Groq to summarize content from YouTube videos or websites.")

# Create columns for input section
st.markdown("### Summarize URL")
source_type = st.radio("Select Source Type:", ["YouTube", "Website"])
url = st.text_input("Enter URL:")
summarize_button = st.button("Summarize", type="primary")

def initialize_llm():
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        return None
    
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="qwen-2.5-32b",
        temperature=0.3
    )

def get_youtube_id(url):
    """Extract video ID from YouTube URL"""
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
    return None

def load_youtube_content(url):
    try:
        video_id = get_youtube_id(url)
        if not video_id:
            st.error("Could not extract YouTube video ID from the URL")
            return None

        # Get the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript text
        full_transcript = " ".join([entry['text'] for entry in transcript])
        
        # Create a Document object
        doc = Document(
            page_content=full_transcript,
            metadata={"source": url, "type": "youtube"}
        )
        
        return [doc]
    
    except Exception as e:
        st.error(f"Error loading YouTube content: {str(e)}")
        if "No transcript" in str(e):
            st.error("This video doesn't have English subtitles/transcript available.")
        return None

def load_and_process_content(url, source_type):
    try:
        if source_type == "YouTube":
            documents = load_youtube_content(url)
        else:
            loader = WebBaseLoader(url)
            documents = loader.load()
        
        if not documents:
            return None
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        return splits
    except Exception as e:
        st.error(f"Error loading content: {str(e)}")
        return None

def create_summary_chain(llm):
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides clear and concise summaries."),
        ("user", """Please provide a comprehensive summary of the following content. 
         Include the main points and key takeaways in a well-structured format.
         
         Content: {context}
         
         Summary:""")
    ])
    
    # Create the summary chain
    return create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )

def validate_url(url, source_type):
    if not url:
        st.warning("Please enter a URL first.")
        return False
        
    if source_type == "YouTube":
        if "youtube.com" not in url and "youtu.be" not in url:
            st.error("Please enter a valid YouTube URL.")
            return False
        if not get_youtube_id(url):
            st.error("Could not extract video ID from the URL. Please check the URL format.")
            return False
    
    return True

def main():
    if summarize_button:
        if not validate_url(url, source_type):
            return
            
        with st.spinner("Processing content..."):
            llm = initialize_llm()
            
            if llm:
                # Load and process content
                documents = load_and_process_content(url, source_type)
                
                if documents:
                    # Create and run the summary chain
                    summary_chain = create_summary_chain(llm)
                    
                    try:
                        summary = summary_chain.invoke({
                            "context": documents
                        })
                        
                        # Display results
                        st.success("Summary generated successfully!")
                        st.markdown("### Summary")
                        st.write(summary)
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    main()
