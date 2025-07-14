# app_utils/streamlit_utils.py
"""
Streamlit UI helpers for the mental-healthcare-chatbot app.

Public API expected by app.py:
 - initialize_session_state()
 - display_chat_interface(conversation_chain)
 - handle_user_query(get_conversation_chain, user_query)
"""

from typing import Any, List
import streamlit as st
import traceback

from langchain_core.messages import HumanMessage, AIMessage


# -----------------------
# Session-state helpers
# -----------------------
def _ensure_session_state_internal() -> None:
    """Internal initializer for session_state keys used by the chat UI."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "processing" not in st.session_state:
        st.session_state["processing"] = False


def initialize_session_state() -> None:
    """
    Public initializer expected by app.py.
    Call this at app startup to ensure session_state keys exist.
    """
    _ensure_session_state_internal()


# -----------------------
# History sanitization
# -----------------------
def _sanitize_history(raw_history: List[Any]) -> List[Any]:
    """
    Ensure history contains only HumanMessage / AIMessage objects.
    Convert stray strings / dicts to AIMessage to avoid malformed payloads.
    """
    sanitized: List[Any] = []
    for item in raw_history:
        if isinstance(item, (HumanMessage, AIMessage)):
            sanitized.append(item)
        elif isinstance(item, str):
            sanitized.append(AIMessage(content=item))
        elif isinstance(item, dict):
            content = item.get("content") if isinstance(item, dict) else None
            role = item.get("role") if isinstance(item, dict) else None
            if content and role and str(role).lower() in ("user", "human"):
                sanitized.append(HumanMessage(content=content))
            elif content:
                sanitized.append(AIMessage(content=content))
            else:
                sanitized.append(AIMessage(content=str(item)))
        else:
            sanitized.append(AIMessage(content=str(item)))
    return sanitized


# -----------------------
# Response extraction
# -----------------------
def _extract_answer_from_response(response: Any) -> str:
    """
    Normalize different chain response shapes to a text answer.
    """
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("answer", "output", "text", "response", "result", "final_answer"):
            if key in response and response[key] is not None:
                return response[key] if isinstance(response[key], str) else str(response[key])
        if "output" in response and isinstance(response["output"], dict):
            for subkey in ("text", "answer"):
                if subkey in response["output"] and response["output"][subkey]:
                    return (
                        response["output"][subkey]
                        if isinstance(response["output"][subkey], str)
                        else str(response["output"][subkey])
                    )
        return str(response)

    try:
        return str(response)
    except Exception:
        return "​(failed to stringify response)"


# -----------------------
# Main chain handler
# -----------------------
def handle_user_query(get_conversation_chain, user_query: str) -> str:
    """
    Send the user_query to the conversation chain safely and return the assistant text.
    - get_conversation_chain: object with .invoke(...) compatible with your app
    - user_query: text from the user
    """
    _ensure_session_state_internal()

    # sanitize to avoid mixed-type chat_history
    st.session_state["chat_history"] = _sanitize_history(st.session_state["chat_history"])

    try:
        response = get_conversation_chain.invoke(
            {"input": user_query, "chat_history": st.session_state["chat_history"]}
        )

        answer_text = _extract_answer_from_response(response)

        # store both sides as proper Message objects
        st.session_state["chat_history"].extend(
            [HumanMessage(content=user_query), AIMessage(content=answer_text)]
        )

        st.session_state["last_error"] = None
        return answer_text

    except Exception as exc:
        tb = traceback.format_exc()
        st.session_state["last_error"] = tb
        st.error("An error occurred while invoking the conversation chain. See debug output below.")
        st.text_area("Traceback (debug)", tb, height=300)
        # Re-raise so platform logs capture the full stack (optional)
        raise


# -----------------------
# Streamlit UI
# -----------------------
def display_chat_interface(conversation_chain) -> None:
    """
    Render the Streamlit chat UI. conversation_chain is forwarded to handle_user_query.
    """
    _ensure_session_state_internal()

    st.header("Mental Healthcare Chatbot")

    # Show last error if present
    if st.session_state.get("last_error"):
        with st.expander("Last error (click to expand)"):
            st.text_area("Last error", st.session_state["last_error"], height=200)

    # Render conversation
    st.session_state["chat_history"] = _sanitize_history(st.session_state["chat_history"])

    for msg in st.session_state["chat_history"]:
        try:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
            else:
                st.markdown(f"**Message:** {str(msg)}")
        except Exception:
            st.markdown("**(failed to render a message)**")

    # -------------------------
    # Input area: use chat_input for better Enter key handling
    # -------------------------
    user_input = st.chat_input("Type your message here...")

    if user_input:
        if st.session_state.get("processing"):
            st.warning("Still processing previous request...")
        else:
            st.session_state["processing"] = True
            try:
                # Add user message immediately for better UX
                st.session_state["chat_history"].append(HumanMessage(content=user_input))
                
                # Show user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get response with spinner
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = conversation_chain.invoke(
                            {"input": user_input, "chat_history": st.session_state["chat_history"][:-1]}
                        )
                        answer_text = _extract_answer_from_response(response)
                        st.markdown(answer_text)
                
                # Add assistant response to history
                st.session_state["chat_history"].append(AIMessage(content=answer_text))
                st.session_state["last_error"] = None
                
            except Exception as exc:
                tb = traceback.format_exc()
                st.session_state["last_error"] = tb
                st.error("An error occurred. See debug output below.")
                st.text_area("Traceback (debug)", tb, height=300)
            finally:
                st.session_state["processing"] = False
                st.rerun()