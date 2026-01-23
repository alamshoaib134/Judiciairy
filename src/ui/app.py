"""
Streamlit UI for JudicAIry Legal RAG Assistant
A beautiful interface for exploring Supreme Court opinions
"""

import streamlit as st
import requests
from typing import Optional, Dict, Any, List

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="JudicAIry",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sophisticated legal aesthetic
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #d4af37 !important;
    }
    
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        border-bottom: 2px solid #d4af37;
        padding-bottom: 0.5rem;
    }
    
    /* Body text */
    p, li, label, .stMarkdown {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #e8e8e8 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
        border-right: 1px solid #d4af37;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #d4af37 !important;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #1a1a2e !important;
        color: #e8e8e8 !important;
        border: 1px solid #d4af37 !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #f4cf47 !important;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.3) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #d4af37 0%, #b8962f 100%) !important;
        color: #1a1a2e !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #f4cf47 0%, #d4af37 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Cards/containers */
    .source-card {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .source-card:hover {
        border-color: #d4af37;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.2);
    }
    
    .source-card h4 {
        color: #d4af37 !important;
        font-family: 'Cormorant Garamond', serif !important;
        margin-bottom: 0.5rem;
    }
    
    .source-card .citation {
        color: #8b8b8b;
        font-style: italic;
        font-size: 0.9rem;
    }
    
    .source-card .excerpt {
        color: #c8c8c8;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-left: 3px solid #d4af37;
        border-radius: 0 8px 8px 0;
    }
    
    /* Answer box */
    .answer-box {
        background: rgba(26, 26, 46, 0.9);
        border: 2px solid #d4af37;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .answer-box h3 {
        color: #d4af37 !important;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.1) 0%, rgba(26, 26, 46, 0.8) 100%);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-card .number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #d4af37;
        font-family: 'Cormorant Garamond', serif;
    }
    
    .stat-card .label {
        color: #8b8b8b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
        border-radius: 8px !important;
        color: #d4af37 !important;
    }
    
    /* Metrics override */
    [data-testid="stMetricValue"] {
        color: #d4af37 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(212, 175, 55, 0.3) !important;
    }
    
    /* Scale balance icon animation */
    @keyframes balance {
        0%, 100% { transform: rotate(-5deg); }
        50% { transform: rotate(5deg); }
    }
    
    .balance-icon {
        display: inline-block;
        animation: balance 3s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unavailable", "message": "API is not running"}


def query_api(question: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Send a query to the API"""
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query failed: {e}")
        return None


def search_api(query: str, top_k: int = 10) -> Optional[List[Dict[str, Any]]]:
    """Search the API"""
    try:
        response = requests.post(
            f"{API_URL}/api/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Search failed: {e}")
        return None


def get_stats() -> Optional[Dict[str, Any]]:
    """Get API stats"""
    try:
        response = requests.get(f"{API_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def render_source_card(source: Dict[str, Any], index: int):
    """Render a source card"""
    st.markdown(f"""
    <div class="source-card">
        <h4>📜 {source.get('case_name', 'Unknown Case')}</h4>
        <p class="citation">{source.get('citation', 'Citation unavailable')} • {source.get('section', 'Body').title()}</p>
        <div class="excerpt">
            "{source.get('excerpt', 'No excerpt available')}"
        </div>
        <p style="color: #8b8b8b; font-size: 0.8rem; margin-top: 0.5rem;">
            Relevance Score: {source.get('score', 0):.3f}
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="font-size: 1.8rem;">⚖️ JudicAIry</h2>', unsafe_allow_html=True)
        st.markdown("*Legal Intelligence at Your Fingertips*")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🔍 Ask a Question", "📚 Search Cases", "📊 Statistics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Number of Sources", 1, 10, 5)
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔌 API Status")
        health = check_api_health()
        if health.get("status") == "healthy":
            st.success(f"✅ Connected ({health.get('documents_indexed', 0):,} docs)")
        elif health.get("status") == "degraded":
            st.warning(f"⚠️ Degraded: {health.get('message', 'Unknown issue')}")
        else:
            st.error("❌ API Unavailable")
            st.info("Start the API with:\n`uvicorn src.api.main:app`")
        
        st.markdown("---")
        st.markdown("""
        <p style="font-size: 0.8rem; color: #8b8b8b;">
        Built with 🏛️ for legal research.<br>
        Powered by U.S. Supreme Court opinions.
        </p>
        """, unsafe_allow_html=True)
    
    # Main content
    if "Ask a Question" in page:
        render_query_page(top_k)
    elif "Search Cases" in page:
        render_search_page(top_k)
    elif "Statistics" in page:
        render_stats_page()


def render_query_page(top_k: int):
    """Render the main Q&A page"""
    st.markdown('<h1><span class="balance-icon">⚖️</span> JudicAIry</h1>', unsafe_allow_html=True)
    st.markdown("### Ask questions about U.S. Supreme Court jurisprudence")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What are the key precedents for free speech limitations?",
            "How has the Court interpreted the Fourth Amendment regarding digital privacy?",
            "What is the standard for proving employment discrimination?",
            "How does qualified immunity apply to police officers?",
            "What are the requirements for standing in federal court?"
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state.question = ex
    
    # Query input
    question = st.text_area(
        "Your Legal Question",
        value=st.session_state.get("question", ""),
        height=100,
        placeholder="Enter your question about Supreme Court law...",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 Ask", use_container_width=True)
    
    # Process query
    if submit and question:
        with st.spinner("Researching Supreme Court opinions..."):
            result = query_api(question, top_k)
        
        if result:
            # Answer
            st.markdown(f"""
            <div class="answer-box">
                <h3>📋 Answer</h3>
                <div style="color: #e8e8e8; line-height: 1.8;">
                    {result.get('answer', 'No answer available')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            sources = result.get('sources', [])
            if sources:
                st.markdown("### 📚 Sources")
                for i, source in enumerate(sources):
                    render_source_card(source, i)
            
            # Metadata
            with st.expander("🔧 Query Metadata"):
                metadata = result.get('metadata', {})
                cols = st.columns(3)
                cols[0].metric("Model", metadata.get('model', 'N/A'))
                cols[1].metric("Retrieved", metadata.get('retrieval_count', 0))
                cols[2].metric("Reranked", metadata.get('rerank_count', 0))


def render_search_page(top_k: int):
    """Render the search page"""
    st.markdown('<h1>📚 Search Supreme Court Opinions</h1>', unsafe_allow_html=True)
    st.markdown("Browse and explore the indexed case law database")
    
    query = st.text_input(
        "Search Query",
        placeholder="Enter keywords or case name...",
        key="search_input"
    )
    
    if st.button("🔍 Search", use_container_width=False) and query:
        with st.spinner("Searching..."):
            results = search_api(query, top_k * 2)
        
        if results:
            st.markdown(f"### Found {len(results)} relevant excerpts")
            
            for result in results:
                with st.expander(
                    f"📜 {result.get('case_name', 'Unknown')} — {result.get('section', 'Body').title()} (Score: {result.get('score', 0):.3f})"
                ):
                    st.markdown(f"**Citation:** {result.get('citation', 'N/A')}")
                    st.markdown("**Content:**")
                    st.markdown(f"> {result.get('content', 'No content')}")
        else:
            st.info("No results found. Try different keywords.")


def render_stats_page():
    """Render the statistics page"""
    st.markdown('<h1>📊 Database Statistics</h1>', unsafe_allow_html=True)
    
    stats = get_stats()
    
    if stats:
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{stats.get('total_documents', 0):,}</div>
                <div class="label">Indexed Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{stats.get('collection_name', 'N/A')}</div>
                <div class="label">Collection</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            section_dist = stats.get('section_distribution_sample', {})
            st.markdown(f"""
            <div class="stat-card">
                <div class="number">{len(section_dist)}</div>
                <div class="label">Section Types</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section distribution
        if section_dist:
            st.markdown("### Section Distribution (Sample)")
            st.bar_chart(section_dist)
    else:
        st.warning("Could not retrieve statistics. Is the API running?")
    
    st.markdown("---")
    st.markdown("### About the Data")
    st.markdown("""
    **JudicAIry** indexes U.S. Supreme Court opinions from multiple sources:
    
    - **CaseSumm Dataset** — 25,600+ opinions with official summaries (1815-present)
    - **Supreme Court Database (SCDB)** — Rich metadata for case analysis
    - **Official supremecourt.gov** — Latest slip opinions
    
    The system uses semantic search with hybrid retrieval (vector + BM25) 
    and cross-encoder reranking for high-precision legal research.
    """)


if __name__ == "__main__":
    main()
