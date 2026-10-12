"""
Streamlit App for Rocket Failure QA Chatbot
Clean UI with chat interface, sidebar, and visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import STREAMLIT_CONFIG, LOG_LEVEL, LOG_FORMAT
from chatbot import RocketFailureChatBot
import logging

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(**STREAMLIT_CONFIG)

# Set style
sns.set_style("whitegrid")


@st.cache_resource
def initialize_chatbot():
    """Initialize and cache the chatbot."""
    return RocketFailureChatBot(load_artifacts=True)


def display_sidebar(chatbot, df):
    """Display sidebar with dataset stats and filters."""
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        
        # Basic stats
        st.metric("Total Incidents", len(df))
        st.metric("Date Range", f"{df['date'].min()[:4]} - {df['date'].max()[:4]}")
        st.metric("Countries", df['country'].nunique())
        st.metric("Vehicles", df['launch_vehicle'].nunique())
        
        st.divider()
        
        # Top failure types chart
        st.header("🔥 Top Failure Types")
        failure_counts = df['failure_type'].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        failure_counts.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Count')
        ax.set_title('Top 8 Failure Types')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.divider()
        
        # Decade filter
        st.header("📅 Filter by Decade")
        decades = df['decade'].unique().tolist()
        selected_decade = st.selectbox("Select Decade", ["All"] + sorted(decades))
        
        st.divider()
        
        # Severity filter
        st.header("⚠️ Filter by Severity")
        severities = df['severity'].unique().tolist()
        severity_order = ['Minor', 'Partial Failure', 'Major', 'Catastrophic']
        severities_sorted = [s for s in severity_order if s in severities]
        selected_severity = st.selectbox("Select Severity", ["All"] + severities_sorted)
        
        return selected_decade, selected_severity


def display_chat_history():
    """Display chat history with user/bot bubbles."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display intent badge for bot messages
            if message["role"] == "assistant" and "intent" in message:
                intent_badges = {
                    'FILTER_QUERY': '🔍 FILTER',
                    'FACTUAL_LOOKUP': '📖 FACTUAL',
                    'COUNT_QUERY': '🔢 COUNT',
                    'COMPARISON_QUERY': '⚖️ COMPARE',
                    'TREND_QUERY': '📈 TREND',
                    'STATISTICAL_QUERY': '📊 STATS',
                    'DEFINITION_QUERY': '📝 DEFINE'
                }
                badge = intent_badges.get(message["intent"], message["intent"])
                st.caption(f"Intent: {badge}")
            
            # Show source records if available
            if message["role"] == "assistant" and "num_results" in message and message["num_results"] > 0:
                with st.expander("Show source records"):
                    if "results" in message and message["results"]:
                        for i, result in enumerate(message["results"][:3], 1):
                            st.write(f"**{i}. {result['incident_id']}** - {result['launch_vehicle']} ({result['date']})")
                            st.write(f"   Failure: {result['failure_type']}")
                            st.write(f"   Score: {result.get('retrieval_score', 0):.4f}")
                            st.divider()


def export_chat_history():
    """Export chat history to CSV."""
    if "messages" not in st.session_state or not st.session_state.messages:
        return None
    
    # Extract chat data
    chat_data = []
    for msg in st.session_state.messages:
        chat_data.append({
            'role': msg['role'],
            'content': msg['content'],
            'intent': msg.get('intent', ''),
            'num_results': msg.get('num_results', 0)
        })
    
    return pd.DataFrame(chat_data)


def main():
    """Main Streamlit app."""
    # Initialize chatbot
    chatbot = initialize_chatbot()
    df = chatbot.df
    
    # Display title
    st.title("🚀 Rocket Failure QA Chatbot")
    st.markdown("Ask questions about rocket failure incidents (1957-2023)")
    
    # Display sidebar
    selected_decade, selected_severity = display_sidebar(chatbot, df)
    
    # Apply filters if selected
    filtered_df = df.copy()
    if selected_decade != "All":
        filtered_df = filtered_df[filtered_df['decade'] == selected_decade]
    if selected_severity != "All":
        filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
    
    # Update chatbot's dataframe if filters applied
    if selected_decade != "All" or selected_severity != "All":
        chatbot.df = filtered_df
        st.info(f"Filtered to {len(filtered_df)} records")
    else:
        chatbot.df = df
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about rocket failures..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    response = chatbot.chat(prompt)
                    answer = response['answer']
                    intent = response['intent']
                    num_results = response['num_results']
                    results = response.get('results', [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display intent badge
                    intent_badges = {
                        'FILTER_QUERY': '🔍 FILTER',
                        'FACTUAL_LOOKUP': '📖 FACTUAL',
                        'COUNT_QUERY': '🔢 COUNT',
                        'COMPARISON_QUERY': '⚖️ COMPARE',
                        'TREND_QUERY': '📈 TREND',
                        'STATISTICAL_QUERY': '📊 STATS',
                        'DEFINITION_QUERY': '📝 DEFINE'
                    }
                    badge = intent_badges.get(intent, intent)
                    st.caption(f"Intent: {badge}")
                    
                    # Show source records
                    if num_results > 0:
                        with st.expander("Show source records"):
                            for i, result in enumerate(results[:3], 1):
                                st.write(f"**{i}. {result['incident_id']}** - {result['launch_vehicle']} ({result['date']})")
                                st.write(f"   Failure: {result['failure_type']}")
                                st.write(f"   Score: {result.get('retrieval_score', 0):.4f}")
                                st.divider()
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "intent": intent,
                        "num_results": num_results,
                        "results": results
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing your query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Export button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        chat_df = export_chat_history()
        if chat_df is not None and not chat_df.empty:
            csv = chat_df.to_csv(index=False)
            st.download_button(
                label="Export Chat",
                data=csv,
                file_name="chat_history.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
