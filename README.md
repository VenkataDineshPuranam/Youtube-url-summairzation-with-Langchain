# Youtube Videos website Summarization with Langchain

A Streamlit-based web application that uses Langchain and Groq to generate summaries from YouTube videos and web pages. The app leverages the powerful Qwen 2.5 32B model to create comprehensive summaries of content.

## Features

- Summarize content from:
  - YouTube videos (using video transcripts)
  - Web pages
- Clean and intuitive user interface
- Configurable API settings
- Intelligent text splitting for better processing
- Error handling and user feedback

## Prerequisites

Before running this application, make sure you have:

- Python 3.7+
- A Groq API key (get it from the Groq platform)
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd text-summarization-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Required Packages

Create a `requirements.txt` file with the following dependencies:

```
streamlit
langchain-groq
langchain-core
langchain-community
youtube-transcript-api
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your Groq API key in the sidebar settings.

3. Choose your content source:
   - YouTube: Enter a valid YouTube URL
   - Website: Enter any web page URL

4. Click "Summarize" to generate the summary

## How It Works

1. **Content Loading**: The app supports two types of content sources:
   - YouTube videos: Extracts transcripts using the YouTube Transcript API
   - Websites: Uses Langchain's WebBaseLoader to extract content

2. **Text Processing**: 
   - Content is split into manageable chunks using RecursiveCharacterTextSplitter
   - Chunks are processed with 1000-character size and 200-character overlap

3. **Summary Generation**:
   - Uses the Qwen 2.5 32B model through Groq's API
   - Implements a custom prompt template for better summarization
   - Processes the content using Langchain's document chain

## Error Handling

The application includes comprehensive error handling for:
- Invalid URLs
- Missing API keys
- Unavailable YouTube transcripts
- Content loading failures
- API communication issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Disclaimer

This application requires a Groq API key to function. Make sure you comply with Groq's terms of service and API usage guidelines.
