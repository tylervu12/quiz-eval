import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from collections import Counter
import json # For handling JSON strings if tags are stored as such

# --- Streamlit App Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Fitness Quiz AI Dashboard", layout="wide")

# Load environment variables (especially for local development)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- Supabase Client Initialization ---
@st.cache_resource # Cache the client across reruns
def init_supabase_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        # Let the main app flow handle the error display after page config
        return None
    try:
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return client
    except Exception as e:
        # Log error to console, main app flow will show user error.
        print(f"Error initializing Supabase client: {e}") 
        return None

supabase_client = init_supabase_client()

# --- Data Fetching ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_data(_client: Client) -> pd.DataFrame:
    if not _client:
        return pd.DataFrame() # Handled by the main flow
    try:
        # Check if created_at exists, if not, try to order by id or skip ordering
        # This is a defensive check; ideally, the schema should be correct.
        try:
            # Attempt to get column info - this is a bit Supabase specific and might not be
            # directly available or easy with the client lib without more complex introspection.
            # A simpler approach for now is to try and catch the specific error for ordering.
            # For now, we assume 'created_at' SHOULD exist as per schema design.
            response = _client.table("chatbot_runs").select("*, created_at").limit(1).execute() # Test presence
            order_column = "created_at"
        except Exception as e:
            # Check if error indicates created_at doesn't exist
            if "column chatbot_runs.created_at does not exist" in str(e).lower():
                st.warning("Column 'created_at' not found for ordering. Data will not be time-sorted. Please check table schema.")
                order_column = None # Cannot order by it
            else:
                raise # Re-raise other errors
        
        if order_column:
            response = _client.table("chatbot_runs").select("*").order(order_column, desc=True).execute()
        else:
            response = _client.table("chatbot_runs").select("*").execute() # Fetch without ordering

        if response.data:
            df = pd.DataFrame(response.data)
            if 'tags' in df.columns and df['tags'].iloc[0] is not None and isinstance(df['tags'].iloc[0], str):
                try:
                    df['tags'] = df['tags'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except Exception as e:
                    st.warning(f"Could not parse some 'tags' column entries as JSON list: {e}. Tags might not be processed correctly.")
            return df
        else:
            # st.warning("No data fetched from Supabase table 'chatbot_runs'.") # Handled by main flow
            return pd.DataFrame()
    except Exception as e:
        # st.error(f"Error fetching data from Supabase: {e}") # Handled by main flow
        print(f"Error fetching data from Supabase: {e}")
        return pd.DataFrame()

# --- Streamlit App Main Flow ---
st.title("🏆 AI Fitness Quiz Dashboard")

if not supabase_client:
    st.error("Supabase client could not be initialized. Please check SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
    st.stop()

df_all_runs = fetch_data(supabase_client)

if df_all_runs.empty:
    st.warning("No data available to display. Ensure your Lambda is logging runs or check Supabase connection.")
    st.stop()

# --- 1. Overall Out-of-Scope Percentage ---
st.header("🎯 Out-of-Scope Analysis (All Runs)")
st.caption("Shows the percentage of all incoming requests that were flagged by the AI as being out-of-scope for fitness/wellness advice (e.g., nonsensical, jokes, or unrelated topics). Subsequent charts only analyze in-scope runs.")
if 'is_out_of_scope' in df_all_runs.columns:
    out_of_scope_counts = df_all_runs['is_out_of_scope'].value_counts(normalize=True).mul(100).reset_index()
    out_of_scope_counts.columns = ['is_out_of_scope', 'percentage']
    out_of_scope_counts['is_out_of_scope'] = out_of_scope_counts['is_out_of_scope'].astype(str) # For consistent labeling

    fig_out_of_scope = px.pie(out_of_scope_counts, names='is_out_of_scope', values='percentage',
                              title='Percentage of Out-of-Scope Requests',
                              labels={'is_out_of_scope': 'Is Out of Scope?', 'percentage': 'Percentage (%)'})
    fig_out_of_scope.update_layout(
        legend=dict(
            orientation="v", # Or "h" for horizontal
            yanchor="top",
            y=0.99, # Adjust this value (0 to 1) to move legend up/down
            xanchor="right",
            x=0.99  # Adjust this value (0 to 1) to move legend left/right
        )
    )
    st.plotly_chart(fig_out_of_scope, use_container_width=True)
else:
    st.warning("Column 'is_out_of_scope' not found in the data.")

# --- Filter data for subsequent charts (is_out_of_scope == FALSE) ---
if 'is_out_of_scope' in df_all_runs.columns:
    df_in_scope = df_all_runs[df_all_runs['is_out_of_scope'] == False].copy() # Use .copy() to avoid SettingWithCopyWarning
    st.metric(label="Total In-Scope Runs Analyzed", value=len(df_in_scope))
    if df_in_scope.empty:
        st.info("No 'in-scope' runs available for detailed analysis. All subsequent charts will be empty.")
else:
    st.warning("Cannot filter for in-scope data as 'is_out_of_scope' column is missing. Subsequent charts might be affected.")
    df_in_scope = df_all_runs.copy() # Proceed with all data if column is missing, with a warning


# Placeholder for next charts
st.header("📊 Detailed Analysis (In-Scope Runs Only)")
if df_in_scope.empty:
    st.info("No data from in-scope runs to display for detailed analysis.")
else:
    # --- 2.1. Distribution of Individual Tags ---
    st.subheader("🏷️ Distribution of Individual Tags")
    if 'tags' in df_in_scope.columns:
        # Handle cases where tags might be None or not a list properly
        all_tags = []
        for item_tags in df_in_scope['tags'].dropna(): # dropna to skip None/NaN values
            if isinstance(item_tags, list):
                if item_tags: # If the list is not empty
                    all_tags.extend(item_tags)
                else: # If the list is empty, count as "No Tags Assigned"
                    all_tags.append("No Tags Assigned")
            # If tags are not lists (e.g., None after parsing, or other types), 
            # you might decide to log a warning or handle them as "No Tags Assigned" as well.
            # For now, we only process lists.

        if not all_tags and not df_in_scope['tags'].dropna().empty: 
            # This case means there were rows with tags, but all were empty lists or non-list Nones
            # and we didn't convert them to "No Tags Assigned" effectively in the loop above for this specific counter scenario
            # A more robust way if all are empty lists that didn't get "No Tags Assigned":
            if all(isinstance(t, list) and not t for t in df_in_scope['tags'].dropna()):
                 all_tags = ["No Tags Assigned"] * len(df_in_scope['tags'].dropna())
        elif not all_tags and df_in_scope['tags'].dropna().empty:
            # All tags are None/NaN
             all_tags = ["No Tags Assigned"] * len(df_in_scope)


        if all_tags:
            tag_counts = Counter(all_tags)
            tag_counts_df = pd.DataFrame.from_dict(tag_counts, orient='index', columns=['count']).reset_index()
            tag_counts_df = tag_counts_df.rename(columns={'index': 'tag'})
            tag_counts_df = tag_counts_df.sort_values(by='count', ascending=False)

            fig_individual_tags = px.bar(tag_counts_df, x='tag', y='count', 
                                         title='Frequency of Each Tag (In-Scope Runs)',
                                         labels={'tag': 'Tag', 'count': 'Number of Occurrences'})
            fig_individual_tags.update_layout(
                height=600, # Increase chart height
                xaxis_tickangle=-45,
                xaxis_title_font_size=16, # Increase x-axis title font size
                yaxis_title_font_size=16, # Increase y-axis title font size
                xaxis_tickfont_size=12,   # Increase x-axis tick label font size
                yaxis_tickfont_size=12    # Increase y-axis tick label font size
            )
            st.plotly_chart(fig_individual_tags, use_container_width=True)
        else:
            st.info("No tags found or all tags were empty/None in in-scope data to display for individual tag distribution.")
    else:
        st.warning("Column 'tags' not found in the in-scope data.")

    # --- 2.2. Distribution of Tag Combinations ---
    st.subheader("🔗 Distribution of Tag Combinations")
    if 'tags' in df_in_scope.columns:
        # Convert list of tags to a sorted, comma-separated string to make them hashable for counting
        # Handle None or empty lists explicitly
        def format_tag_combination(tag_list):
            if isinstance(tag_list, list):
                if not tag_list: # Empty list
                    return "No Tags Assigned"
                return ", ".join(sorted(tag_list))
            return "No Tags Assigned" # Default for None or non-list types

        df_in_scope.loc[:, 'tag_combination'] = df_in_scope['tags'].apply(format_tag_combination)
        
        combination_counts = df_in_scope['tag_combination'].value_counts().reset_index()
        combination_counts.columns = ['combination', 'count']
        combination_counts = combination_counts.sort_values(by='count', ascending=False)

        fig_tag_combinations = px.bar(combination_counts, x='combination', y='count', 
                                      title='Frequency of Tag Combinations (In-Scope Runs)',
                                      labels={'combination': 'Tag Combination', 'count': 'Number of Occurrences'})
        fig_tag_combinations.update_layout(
            height=700, # Increase chart height (can be different from the other chart)
            xaxis_tickangle=-45,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            xaxis_tickfont_size=12,
            yaxis_tickfont_size=12,
            # Potentially add bottom margin if x-axis labels are long and get cut off
            margin=dict(b=150) # b is bottom margin; adjust as needed
        )
        st.plotly_chart(fig_tag_combinations, use_container_width=True)
    else:
        st.warning("Column 'tags' not found for tag combination distribution.")

    # --- 3. Distribution of Custom Context Relevance Score ---
    st.subheader("💡 Context Relevance Score Distribution")
    st.caption("Assesses if the retrieved RAG context can address the user's request. Scores: (1) Cannot address, (2) Partially addresses, (3) Directly addresses.")
    if 'custom_context_relevance_score' in df_in_scope.columns:
        # Create the categorical column, converting to string first, then filling NaNs
        df_in_scope.loc[:, 'context_relevance_cat'] = df_in_scope['custom_context_relevance_score'].astype(str).replace('nan', 'Not Scored/Invalid')
        # If original scores were integers and became e.g. "1.0", clean that up for display
        df_in_scope.loc[:, 'context_relevance_cat'] = df_in_scope['context_relevance_cat'].replace({'1.0': '1', '2.0': '2', '3.0': '3'})

        context_relevance_counts = df_in_scope['context_relevance_cat'].value_counts().reset_index()
        context_relevance_counts.columns = ['score', 'count']
        
        score_order = ["1", "2", "3", "Not Scored/Invalid"]
        context_relevance_counts['score_order_cat'] = pd.Categorical(
            context_relevance_counts['score'], 
            categories=[s for s in score_order if s in context_relevance_counts['score'].unique()], 
            ordered=True
        )
        context_relevance_counts = context_relevance_counts.sort_values('score_order_cat')

        fig_context_relevance = px.pie(context_relevance_counts, names='score', values='count',
                                       title='Distribution of Context Relevance Scores (In-Scope Runs)',
                                       labels={'score': 'Context Relevance Score', 'count': 'Number of Runs'})
        fig_context_relevance.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_context_relevance, use_container_width=True)
    else:
        st.warning("Column 'custom_context_relevance_score' not found.")

    # --- 4. Percentage for Custom Hallucination Score ---
    st.subheader("👻 Hallucination Score Distribution")
    st.caption("Assesses if the recommendation is faithful to the retrieved RAG context. Scores: (0) Hallucination detected / Not supported by context, (1) Faithful / Supported by context.")
    if 'custom_hallucination_score' in df_in_scope.columns:
        # Create the categorical column, converting to string first, then filling NaNs
        df_in_scope.loc[:, 'hallucination_cat'] = df_in_scope['custom_hallucination_score'].astype(str).replace('nan', 'Not Scored/Invalid')
        # If original scores were integers and became e.g. "0.0", "1.0", clean that up for display
        df_in_scope.loc[:, 'hallucination_cat'] = df_in_scope['hallucination_cat'].replace({'0.0': '0', '1.0': '1'})
        
        hallucination_counts = df_in_scope['hallucination_cat'].value_counts(normalize=True).mul(100).reset_index()
        hallucination_counts.columns = ['score', 'percentage']
        
        score_order_hallu = ["0", "1", "Not Scored/Invalid"]
        hallucination_counts['score_order_cat'] = pd.Categorical(
            hallucination_counts['score'], 
            categories=[s for s in score_order_hallu if s in hallucination_counts['score'].unique()], 
            ordered=True
        )
        hallucination_counts = hallucination_counts.sort_values('score_order_cat')

        fig_hallucination = px.pie(hallucination_counts, names='score', values='percentage', 
                                     title='Distribution of Hallucination Scores (In-Scope Runs)',
                                     labels={'score': 'Hallucination Score', 'percentage': 'Percentage (%)'})
        fig_hallucination.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_hallucination, use_container_width=True)
    else:
        st.warning("Column 'custom_hallucination_score' not found.")

    # --- 5. Distribution of Similar Chunk Percentage ---
    st.subheader("🔗 Similar Chunk Percentage (RAG Quality)")
    st.caption("Shows the distribution of runs based on the percentage of retrieved RAG chunks that had a similarity score of 0.5 or higher with the user's input. Higher percentages suggest more relevant chunks were retrieved overall for that run.")
    if 'similar_chunk_pct' in df_in_scope.columns:
        # Ensure the column is numeric, coercing errors
        df_in_scope.loc[:, 'similar_chunk_pct_numeric'] = pd.to_numeric(df_in_scope['similar_chunk_pct'], errors='coerce')
        
        fig_similar_chunk_pct = px.histogram(df_in_scope.dropna(subset=['similar_chunk_pct_numeric']), 
                                             x='similar_chunk_pct_numeric', nbins=20,
                                             title='Distribution of Similar Chunk Percentage (In-Scope Runs)',
                                             labels={'similar_chunk_pct_numeric': 'Percentage of Chunks with Score >= 0.5'})
        st.plotly_chart(fig_similar_chunk_pct, use_container_width=True)
    else:
        st.warning("Column 'similar_chunk_pct' not found.")

    # --- 6. Distribution of Missing Context Flag ---
    st.subheader("❓ Missing Context Flag Distribution")
    st.caption("Indicates if context was deemed missing. This is True if no RAG documents were retrieved, or if all retrieved documents had a similarity score below 0.4 with the user's input.")
    if 'missing_context' in df_in_scope.columns:
        missing_context_counts = df_in_scope['missing_context'].value_counts(normalize=True).mul(100).reset_index()
        missing_context_counts.columns = ['missing_context_flag', 'percentage']
        # Ensure boolean values are strings for consistent labeling in pie chart
        missing_context_counts['missing_context_flag'] = missing_context_counts['missing_context_flag'].astype(str)
        
        fig_missing_context = px.pie(missing_context_counts, names='missing_context_flag', values='percentage',
                                   title='Proportion of Runs Flagged with Missing Context (In-Scope Runs)',
                                   labels={'missing_context_flag': 'Missing Context?', 'percentage': 'Percentage (%)'})
        fig_missing_context.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_missing_context, use_container_width=True)
    else:
        st.warning("Column 'missing_context' not found.")

    st.markdown("--- End of Dashboard ---")

# To run this dashboard:
# 1. Save this code as dashboard.py
# 2. Make sure your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY
# 3. Open your terminal in the same directory and run: streamlit run dashboard.py 