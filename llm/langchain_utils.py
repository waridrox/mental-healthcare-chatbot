import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_embeddings(provider: str = "huggingface"):
    """
    Retrieve embeddings model based on the specified provider.

    Args:
        provider (str): The provider for the embeddings. Defaults to 'huggingface'.

    Returns:
        Embeddings model instance.
    """
    if provider.lower() == "huggingface":
        try:
            model_name = st.secrets.get("EMBEDDING_MODEL_NAME") or os.environ.get(
                "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
            )
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize HuggingFace embeddings: {e}")
    else:
        raise ValueError(f"Unsupported provider for embeddings: {provider}")

    return embeddings


def get_llm(provider: str = "openrouter"):
    """
    Retrieve a language model instance.

    Args:
        provider (str): The LLM provider. Defaults to 'openrouter'.

    Returns:
        Language model instance.
    """
    if provider.lower() == "openrouter":
        try:
            api_key = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
            model_name = os.environ.get("OPENROUTER_MODEL_NAME") or st.secrets.get(
                "OPENROUTER_MODEL_NAME", "openai/gpt-5.2"
            )

            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment or secrets")

            model = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                model=model_name,
                temperature=0.7,
                max_tokens=1024,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501", # Optional. Site URL for rankings on openrouter.ai.
                    "X-OpenRouter-Title": "Mental HealthCare Chatbot", # Optional. Site title for rankings on openrouter.ai.
                }
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenRouter model: {e}")
    elif provider.lower() == "openai":
        try:
            # Try environment variables first, then secrets
            api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
            model_name = os.environ.get("OPENAI_MODEL_NAME") or st.secrets.get(
                "OPENAI_MODEL_NAME", "gpt-4o-mini"
            )

            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or secrets")

            model = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.7,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI model: {e}")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return model


def create_conversational_chain(retriever: FAISS):
    """
    Create the conversational retrieval chain.

    Args:
        retriever (FAISS): The vector database retriever.

    Returns:
        create_retrieval_chain: The conversational retrieval chain.
    """
    language_model = get_llm(provider="openrouter")

    contextualize_q_system_prompt = "Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        language_model, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \

    {context}.
    Do not include, "According to the context" in the final output
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(language_model, qa_prompt)
    conversation_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    return conversation_chain
