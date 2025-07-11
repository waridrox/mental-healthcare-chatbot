# Mental Healthcare Chatbot

This repository contains a healthcare chatbot application built using LangChain and Streamlit, focusing on mental health support. The chatbot utilizes advanced features including PDF search using a vector database (FAISS) to provide relevant responses.

https://github.com/user-attachments/assets/da3bf6fb-4b69-4584-8957-ab0db82d548d

## Overview

The healthcare chatbot leverages LangChain, a framework for integrating retrieval and generation models, to offer personalized mental health assistance. It includes capabilities for querying information within PDF documents stored in a vector database.

## Features

- **PDF Query Search**: Utilizes FAISS and RAPTOR for efficient querying of information within PDF documents.
- **Mental Health Support**: Provides conversational AI capabilities tailored to mental health queries and support.
- **Streamlit Integration**: User-friendly interface powered by Streamlit for seamless interaction.

## Getting Started

1. **Installation**: Clone this repository and install dependencies specified in `requirements.txt`.

   ```bash
   git clone https://github.com/waridrox/mental-healthcare-chatbot.git
   cd mental-healthcare-chatbot
   create a python or conda env
   pip install -r requirements.txt
   ```

2. **Run the Chatbot**: Execute the Streamlit application.

   ```bash
   streamlit run app.py
   ```

3. **Explore PDF Search**: Use the chatbot interface to query mental health-related topics and test the PDF search functionality.
