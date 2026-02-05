import pandas as pd
import datetime as dt
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
from collections import defaultdict, Counter
from mlxtend.frequent_patterns import apriori, association_rules

# Set page configuration
st.set_page_config(
    page_title="RFM Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for animations
st.markdown("""
<style>
    .nav-link {
        padding: 12px 20px;
        margin: 8px 0;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        display: block;
        text-decoration: none;
        color: inherit;
    }
    
    .nav-link:hover {
        background: rgba(255, 255, 255, 0.2);
        padding-left: 30px;
        transform: scale(1.02);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .nav-link.active {
        background: rgba(255, 255, 255, 0.25);
        font-weight: bold;
        border-left: 4px solid #ff4b4b;
    }
    
    .main-content {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .nav-icon {
        display: inline-block;
        margin-right: 8px;
        transition: transform 0.3s ease;
    }
    
    .nav-link:hover .nav-icon {
        transform: scale(1.2);
    }
</style>
""", unsafe_allow_html=True)

# Add click sound JavaScript
st.markdown("""
<script>
    const audio = new Audio("data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//NQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV6urq6urq6urq6urq6urq6urq6urq6urq6v////////////////////////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAAAAAAAAAAAECIjk4EAAAAAAAAAAAAAAAAAAAAA//MUZAAAAAGkAAAAAAAAA0gAAAAATEFN//MUZAMAAAGkAAAAAAAAA0gAAAAARTMu//MUZAYAAAGkAAAAAAAAA0gAAAAAOTku//MUZAkAAAGkAAAAAAAAA0gAAAAANVVV");
    
    function playClickSound() {
        audio.currentTime = 0;
        audio.play().catch(e => {});
    }
    
    document.addEventListener('DOMContentLoaded', function() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', playClickSound);
        });
    });
</script>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "RFM Analysis"

def change_page(page):
    st.session_state.current_page = page

# Enhanced navigation function
def show_navigation():
    st.sidebar.title("📱 Navigation")
    
    nav_items = {
        "RFM Analysis": "📊",
        "Dashboard": "📈",
        "Customers": "👥",
        "Revenue": "💰",
        "ML Analysis": "🤖"
    }
    
    for page, icon in nav_items.items():
        # Using columns for better click detection
        col1, col2 = st.sidebar.columns([1, 0.1])
        with col1:
            if st.button(
                f"{icon} {page}",
                key=f"nav_{page}",
                help=f"Navigate to {page}",
                use_container_width=True,
                on_click=change_page,
                args=(page,)
            ):
                pass
        
        # Add the styling for the active state
        if st.session_state.current_page == page:
            st.sidebar.markdown(f"""
                <style>
                    div[data-testid="stButton"] button[kind="secondary"] {{
                        background-color: rgba(255, 255, 255, 0.25);
                        border-left: 4px solid #ff4b4b;
                        font-weight: bold;
                    }}
                </style>
                """, unsafe_allow_html=True)

# Dashboard page
def show_dashboard():
    st.title("📊 RFM Analysis Dashboard")
    
    # Load the data
    df = pd.read_csv('rfm_data.csv')
    
    # Convert PurchaseDate to datetime
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    
    # Define reference date for recency calculation
    reference_date = dt.datetime(2023, 7, 1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'PurchaseDate': lambda x: (reference_date - x.max()).days,
        'OrderID': 'count',
        'TransactionAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Filter out non-positive monetary values
    rfm = rfm[rfm['Monetary'] > 0]
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(rfm):,}",
            delta=f"{len(rfm[rfm['Recency'] <= 30])} active customers"
        )
    
    with col2:
        avg_monetary = rfm['Monetary'].mean()
        st.metric(
            label="Average Customer Value",
            value=f"${avg_monetary:,.2f}",
            delta=f"{((rfm['Monetary'] > avg_monetary).mean() * 100):.1f}% above average"
        )
    
    with col3:
        avg_recency = rfm['Recency'].mean()
        st.metric(
            label="Average Recency",
            value=f"{avg_recency:.1f} days",
            delta=f"{((rfm['Recency'] <= avg_recency).mean() * 100):.1f}% recent customers"
        )
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Distribution by RFM Score")
        # Calculate RFM score
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, ['1', '2', '3', '4'])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, ['4', '3', '2', '1'])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, ['4', '3', '2', '1'])
        rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)
        
        fig = px.histogram(rfm, x='RFM_Score', nbins=20,
                          title='Distribution of Customer RFM Scores')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Value vs Recency")
        fig = px.scatter(rfm, x='Recency', y='Monetary',
                        title='Customer Value vs Recency',
                        color='RFM_Score')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segments Analysis
    st.subheader("Customer Segments Analysis")
    # Define RFM segments
    def rfm_segment(df):
        if df['RFM_Score'] >= 9:
            return 'Champions'
        elif df['RFM_Score'] >= 8:
            return 'Loyal Customers'
        elif df['RFM_Score'] >= 7:
            return 'Potential Loyalists'
        elif df['RFM_Score'] >= 6:
            return 'Recent Customers'
        elif df['RFM_Score'] >= 5:
            return 'Promising'
        elif df['RFM_Score'] >= 4:
            return 'Need Attention'
        elif df['RFM_Score'] >= 3:
            return 'At Risk'
        else:
            return 'Lost'
    
    rfm['Customer_Segment'] = rfm.apply(rfm_segment, axis=1)
    segments = rfm['Customer_Segment'].value_counts()
    fig = px.pie(values=segments.values, names=segments.index,
                 title='Distribution of Customer Segments')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Customers Table
    st.subheader("Top 10 Customers by Value")
    top_customers = rfm.nlargest(10, 'Monetary')[['CustomerID', 'Monetary', 'Recency', 'Customer_Segment']]
    st.dataframe(top_customers)

# Main RFM Analysis page
def show_rfm_analysis():
    # Define a function to get translations based on the selected language
    def get_translations(language):
        translations = {
            'English': {
                'title': 'RFM Analysis',
                'theme': 'Theme Settings',
                'refresh_rate': 'Data Refresh Rate',
                'notifications': 'Notification Preferences',
                'language': 'Language Settings'
            },
            'Spanish': {
                'title': 'RFM Analysis',
                'theme': 'Configuración de Tema',
                'refresh_rate': 'Frecuencia de Actualización de Datos',
                'notifications': 'Preferencias de Notificación',
                'language': 'Configuración de Idioma'
            },
            'French': {
                'title': 'RFM Analysis',
                'theme': 'Paramètres de Thème',
                'refresh_rate': 'Fréquence de Rafraîchissement des Données',
                'notifications': 'Préférences de Notification',
                'language': 'Paramètres de Langue'
            },
            'German': {
                'title': 'RFM Analysis',
                'theme': 'Thema Einstellungen',
                'refresh_rate': 'Datenaktualisierungsrate',
                'notifications': 'Benachrichtigungseinstellungen',
                'language': 'Spracheinstellungen'
            },
            'Hindi': {
                'title': 'RFM Analysis',
                'theme': 'थीम सेटिंग्स',
                'refresh_rate': 'डेटा रीफ्रेश दर',
                'notifications': 'सूचना प्राथमिकताएँ',
                'language': 'भाषा सेटिंग्स'
            },
            'Punjabi': {
                'title': 'RFM Analysis',
                'theme': 'ਥੀਮ ਸੈਟਿੰਗਜ਼',
                'refresh_rate': 'ਡਾਟਾ ਰੀਫ੍ਰੈਸ਼ ਦਰ',
                'notifications': 'ਸੂਚਨਾ ਪ੍ਰਾਥਮਿਕਤਾਵਾਂ',
                'language': 'ਭਾਸ਼ਾ ਸੈਟਿੰਗਜ਼'
            }
        }
        return translations.get(language, translations['English'])

    # Load data
    file_path = 'rfm_data.csv'  # Change this to the actual path if necessary
    data = pd.read_csv(file_path)

    # Convert PurchaseDate to datetime
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])

    # Define reference date for recency calculation
    reference_date = dt.datetime(2023, 7, 1)

    # Calculate RFM metrics
    rfm = data.groupby('CustomerID').agg({
        'PurchaseDate': lambda x: (reference_date - x.max()).days,
        'OrderID': 'count',
        'TransactionAmount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Filter out non-positive monetary values
    rfm = rfm[rfm['Monetary'] > 0]

    # Define RFM score thresholds
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, ['1', '2', '3', '4'])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, ['4', '3', '2', '1'])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, ['4', '3', '2', '1'])

    # Concatenate RFM score to a single RFM segment
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1).astype(int)

    # Define RFM segments
    def rfm_segment(df):
        if df['RFM_Score'] >= 9:
            return 'Champions'
        elif df['RFM_Score'] >= 8:
            return 'Loyal Customers'
        elif df['RFM_Score'] >= 7:
            return 'Potential Loyalists'
        elif df['RFM_Score'] >= 6:
            return 'Recent Customers'
        elif df['RFM_Score'] >= 5:
            return 'Promising'
        elif df['RFM_Score'] >= 4:
            return 'Need Attention'
        elif df['RFM_Score'] >= 3:
            return 'At Risk'
        else:
            return 'Lost'

    rfm['RFM_Segment'] = rfm.apply(rfm_segment, axis=1)

    # Count of customers in each segment
    segment_counts = rfm['RFM_Segment'].value_counts().reset_index()
    segment_counts.columns = ['RFM_Segment', 'Count']

    # Streamlit Dashboard

    # Add custom CSS with animations
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        * {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
            animation: fadeIn 1.5s ease-out;
        }

        .header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            font-size: 3.2em;
            color: white;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            animation: fadeIn 2s ease-out;
        }

        .header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1em;
            margin: 0.5rem 0;
        }

        .header img {
            width: 70px;
            margin: 1rem 0;
            animation: pulse 2s infinite;
        }

        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            padding: 1rem;
            animation: fadeIn 1s ease-out;
        }

        .metric {
            background: linear-gradient(135deg, #F0F7FF 0%, #E5F0FF 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .metric:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
        }

        .metric h3 {
            color: #2575fc;
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }

        .metric p {
            font-size: 2.2em;
            color: #6a11cb;
            font-weight: 700;
            margin: 0;
        }

        .stButton>button {
            background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
            color: white !important;
            padding: 0.8rem 2rem;
            border: none;
            border-radius: 10px;
            font-weight: 500;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            width: auto;
            margin: 1rem auto;
            display: block;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            background: linear-gradient(120deg, #5a0cb1 0%, #1565ec 100%);
        }

        .plot-container {
            background: linear-gradient(135deg, #F5F8FF 0%, #EDF2FF 100%);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin: 2rem 0;
            animation: fadeIn 1.5s ease-out;
        }

        .segment {
            background: linear-gradient(135deg, #F0F7FF 0%, #E5F0FF 100%);
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            text-align: center;
        }

        .segment h3 {
            color: #6a11cb;
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .segment p {
            color: #4a5568;
            font-size: 1.1em;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #6a11cb 0%, #2575fc 100%);
        }

        .css-1d391kg .stSelectbox label {
            color: white !important;
            font-weight: 500;
        }

        .stSelectbox select {
            background: white;
            border-radius: 8px;
            border: none;
            color: #2575fc !important;
            font-weight: 500;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(120deg, #5a0cb1 0%, #1565ec 100%);
        }

        /* Custom styling for data preview table */
        .dataframe {
            font-family: 'Poppins', sans-serif !important;
            width: 100% !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
            margin: 2rem 0 !important;
            animation: fadeIn 1s ease-out !important;
            border: 1px solid #e0e0e0 !important;
            background-color: #F8FAFF !important;
        }

        .dataframe thead {
            background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%) !important;
        }

        .dataframe thead th {
            padding: 1rem !important;
            font-weight: 600 !important;
            text-align: left !important;
            font-size: 1.1em !important;
            border: none !important;
            color: #e8f4ff !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        .dataframe tbody tr {
            transition: all 0.3s ease !important;
            background-color: #F0F7FF !important;
        }

        .dataframe tbody tr:nth-child(even) {
            background-color: #E5F0FF !important;
        }

        .dataframe tbody tr:hover {
            background-color: #D1E5FF !important;
            transform: translateX(5px) !important;
        }

        .dataframe tbody td {
            padding: 0.8rem 1rem !important;
            border: none !important;
            font-size: 1em !important;
            color: #1e293b !important;
            border-bottom: 1px solid #e0e0e0 !important;
        }

        /* Style for the data preview container */
        .data-preview-container {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            margin: 2rem 0 !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        }

        .data-preview-header {
            color: #1e293b !important;
            font-size: 1.8em !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            text-align: center !important;
        }

        .data-preview-description {
            color: #334155 !important;
            font-size: 1.1em !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }

        /* Enhanced search box styling */
        .search-container {
            margin: 2rem 0 !important;
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%) !important;
            padding: 2rem !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15) !important;
            border: 1px solid #C7D2FE !important;
        }

        /* Search input label styling */
        .search-container .stTextInput label {
            color: #FF6B6B !important;
            font-weight: 600 !important;
            font-size: 1.3em !important;
            font-family: 'Poppins', sans-serif !important;
            margin-bottom: 1rem !important;
            text-transform: none !important;
            letter-spacing: 0.5px !important;
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            padding: 0.5rem 0 !important;
            display: block !important;
            position: relative !important;
        }

        /* Search input label emoji styling */
        .search-container .stTextInput label span {
            color: #FF6B6B !important;
            font-size: 1.4em !important;
            margin-right: 0.8rem !important;
            vertical-align: middle !important;
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }

        /* Search input field styling */
        .search-container .stTextInput input {
            font-family: 'Poppins', sans-serif !important;
            font-size: 1.1em !important;
            padding: 1.2rem 1.5rem !important;
            border-radius: 12px !important;
            border: 2px solid #FFB4AC !important;
            background-color: #FFF0EE !important;
            color: #E53E3E !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin-top: 0.5rem !important;
            box-shadow: 0 2px 10px rgba(255, 107, 107, 0.1) !important;
        }

        /* Search input hover state */
        .search-container .stTextInput input:hover {
            border-color: #FF6B6B !important;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.2) !important;
            background-color: #ffffff !important;
        }

        /* Search input focus state */
        .search-container .stTextInput input:focus {
            border-color: #FF6B6B !important;
            box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.3) !important;
            outline: none !important;
            background-color: #ffffff !important;
        }

        /* Search input placeholder */
        .search-container .stTextInput input::placeholder {
            color: #FF8E53 !important;
            opacity: 0.8 !important;
            font-size: 1em !important;
        }

        /* Force color for the search label */
        .search-container [data-testid="stTextInput"] label p {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-weight: 600 !important;
            display: inline-block !important;
            position: relative !important;
        }

        /* Style for the search icon */
        .search-container .stTextInput .st-emotion-cache-1gulkj5 {
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 1.2em !important;
        }

        /* Add a subtle animation to the search container */
        .search-container {
            animation: fadeInUp 0.5s ease-out !important;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Add a subtle transition effect to the input */
        .search-container .stTextInput input {
            transition: all 0.3s ease-in-out !important;
        }

        /* Enhanced pagination styling */
        .pagination-container {
            background: #f0f4f8 !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05) !important;
            margin: 1rem 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }

        .pagination-container .stButton>button {
            background: #2c3e50 !important;
            color: #e2e8f0 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
        }

        .pagination-container .stButton>button:hover:not([disabled]) {
            background: #3498db !important;
            color: #f0f4f8 !important;
            transform: translateY(-2px) !important;
        }

        .pagination-container .stButton>button:disabled {
            background: #94a3b8 !important;
            color: #475569 !important;
            cursor: not-allowed !important;
        }

        .page-info {
            font-family: 'Poppins', sans-serif !important;
            color: #1e293b !important;
            font-size: 1.1em !important;
            font-weight: 500 !important;
            text-align: center !important;
        }

        /* Stats cards styling */
        .stats-container {
            background: linear-gradient(135deg, #F0F7FF 0%, #E5F0FF 100%) !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
            margin: 1.5rem 0 !important;
        }

        .stats-container .stMetric {
            background: linear-gradient(135deg, #F5F8FF 0%, #EDF2FF 100%) !important;
            border: 1px solid #e2e8f0 !important;
            transition: all 0.3s ease !important;
        }

        .stats-container .stMetric:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
        }

        /* Metric styling for stats container */
        .stats-container .stMetric label {
            color: #1a202c !important;  /* Dark color for label */
            font-size: 1.1em !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
        }

        .stats-container .stMetric [data-testid="stMetricValue"] {
            color: #000000 !important;  /* Black color for the value */
            font-size: 1.8em !important;
            font-weight: 700 !important;
            font-family: 'Poppins', sans-serif !important;
        }

        .stats-container .stMetric [data-testid="stMetricDelta"] {
            color: red !important;  /* Red color for any delta values */
            font-weight: 600 !important;
        }

        /* General metric styling */
        .stMetric {
            background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%) !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            border: 1px solid #cbd5e0 !important;
            margin: 0.5rem !important;
        }

        .stMetric label {
            color: #1a202c !important;  /* Dark color for label */
            font-size: 1.1em !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: #000000 !important;  /* Black color for the value */
            font-size: 1.8em !important;
            font-weight: 700 !important;
            font-family: 'Poppins', sans-serif !important;
        }

        .stMetric [data-testid="stMetricDelta"] {
            color: red !important;  /* Red color for any delta values */
            font-weight: 600 !important;
        }

        /* Emoji icon styling */
        .stats-container .stMetric label span {
            font-size: 1.2em !important;
            color: #1a202c !important;  /* Dark color for emoji */
        }

        /* Additional styling for metric containers */
        [data-testid="stMetricValue"] > div {
            color: #000000 !important;  /* Ensure nested divs also have black text */
        }

        [data-testid="stMetricLabel"] {
            color: #1a202c !important;  /* Dark color for all metric labels */
        }

        /* Ensure all metric text is visible */
        .stMetric div {
            color: #000000 !important;  /* Force all div text in metrics to be black */
        }

        .stMetric span {
            color: #1a202c !important;  /* Force all span text in metrics to be dark */
        }

        /* Toggle button styling */
        .toggle-button-container {
            text-align: center !important;
            margin: 2rem 0 !important;
        }

        .toggle-button-container .stButton>button {
            background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%) !important;
            color: #e2e8f0 !important;
            padding: 0.8rem 2rem !important;
            font-size: 1.1em !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
        }

        .toggle-button-container .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
            background: linear-gradient(120deg, #34495e 0%, #2980b9 100%) !important;
            color: #f0f4f8 !important;
        }

        /* Navbar Styling */
        .navbar {
            background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%);
            padding: 1rem 2rem;
            margin-bottom: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-items {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-item {
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1.1em;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 8px;
        }

        .nav-item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
        }

        .nav-logo {
            font-size: 1.5em;
            font-weight: 700;
            color: white;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Top Performing Segments styling */
        .top-segments-header {
            color: #000000 !important;
            font-size: 1.8em !important;
            font-weight: 600 !important;
            margin: 1.5rem 0 !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }

        /* Update dark theme background color for better visibility */
        .stApp {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        }

        .segment, .data-preview-container, .stats-container, .header, .navbar, .nav-item, .nav-logo {
            color: #1e293b;
        }

        .stMetric [data-testid="stMetricDelta"] {
            color: red !important;  /* Red color for any delta values */
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize translations based on the selected language
    language = st.sidebar.selectbox("Language:", ('English', 'Spanish', 'French', 'German', 'Hindi', 'Punjabi'))
    translations = get_translations(language)

    # Update the navbar to cover the full width and ensure links are functional
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">
            📊 <span style='color: #32CD32;'>RFM Analysis</span>
        </div>
        <div class="nav-items">
            <a href="#dashboard" class="nav-item">
                <i class="fas fa-chart-line"></i> Dashboard
            </a>
            <a href="#customers" class="nav-item">
                <i class="fas fa-users"></i> Customers
            </a>
            <a href="#revenue" class="nav-item">
                <i class="fas fa-dollar-sign"></i> Revenue
            </a>
            <a href="#settings" class="nav-item">
                <i class="fas fa-cog"></i> Settings
            </a>
            <button class="translate-button nav-item" onclick="toggleTranslate()" fdprocessedid="3fmhjf">
                🌐
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ensure Font Awesome is loaded for icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """, unsafe_allow_html=True)

    # Header with enhanced styling
    st.markdown("""
    <div class='header'>
        <h1>✨ <span style='color: #32CD32;'>RFM Analysis</span></h1>
        <img src='https://img.icons8.com/fluency/96/000000/customer-insight.png'/>
        <p>RFM Analysis Dashboard</p>
        <p>Analyze customer segments based on Recency, Frequency, and Monetary values</p>
    </div>
    """, unsafe_allow_html=True)

    # Data Preview Button with Toggle and Enhanced Display
    if 'data_preview' not in st.session_state:
        st.session_state.data_preview = False
        st.session_state.page_number = 0
        st.session_state.rows_per_page = 10
        st.session_state.search_term = ""

    st.markdown("""
        <div class='data-preview-container'>
            <h3 class='data-preview-header'>Interactive Data Preview</h3>
            <p class='data-preview-description'>Explore your customer transaction data with enhanced visualization and filtering options.</p>
        </div>
    """, unsafe_allow_html=True)

    # Wrap the toggle button in a centered container
    st.markdown("<div class='toggle-button-container'>", unsafe_allow_html=True)
    if st.button('📊 Toggle Data Preview', key='toggle_preview'):
        st.session_state.data_preview = not st.session_state.data_preview
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.data_preview:
        # Search functionality with enhanced styling
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        search = st.text_input('🔍 Search in data:', key='search_input')
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Filter data based on search term
        if search:
            filtered_data = data[data.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
        else:
            filtered_data = data

        # Pagination with enhanced styling
        total_pages = len(filtered_data) // st.session_state.rows_per_page
        
        st.markdown("<div class='pagination-container'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button('◀ Previous', disabled=st.session_state.page_number == 0):
                st.session_state.page_number -= 1
        
        with col2:
            st.markdown(f"<p class='page-info'>Page {st.session_state.page_number + 1} of {total_pages + 1}</p>", unsafe_allow_html=True)
        
        with col3:
            if st.button('Next ▶', disabled=st.session_state.page_number >= total_pages):
                st.session_state.page_number += 1
        st.markdown("</div>", unsafe_allow_html=True)

        # Display paginated data
        start_idx = st.session_state.page_number * st.session_state.rows_per_page
        end_idx = start_idx + st.session_state.rows_per_page
        
        # Show data with styling
        st.markdown("<div class='data-preview-container'>", unsafe_allow_html=True)
        st.dataframe(
            filtered_data.iloc[start_idx:end_idx],
            height=400
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Display data statistics with enhanced styling
        st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", len(filtered_data))
        with col2:
            st.metric("👥 Unique Customers", filtered_data['CustomerID'].nunique())
        with col3:
            st.metric("📅 Date Range", f"{filtered_data['PurchaseDate'].min().strftime('%Y-%m-%d')} to {filtered_data['PurchaseDate'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("💰 Total Revenue", f"${filtered_data['TransactionAmount'].sum():,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Metrics
    total_customers = rfm['CustomerID'].nunique()
    avg_recency = int(rfm['Recency'].mean())
    avg_frequency = int(rfm['Frequency'].mean())
    avg_monetary = int(rfm['Monetary'].mean())

    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric'>
            <h3>Total Customers</h3>
            <p>{total_customers}</p>
        </div>
        <div class='metric'>
            <h3>Average Recency</h3>
            <p>{avg_recency}</p>
        </div>
        <div class='metric'>
            <h3>Average Frequency</h3>
            <p>{avg_frequency}</p>
        </div>
        <div class='metric'>
            <h3>Average Monetary Value</h3>
            <p>{avg_monetary}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dropdown for analysis type
    st.sidebar.title("Analysis Options")
    analysis_type = st.sidebar.selectbox("Choose Analysis Type:", [
        "Customer Segmentation Overview",
        "Purchase Pattern Analysis",
        "Customer Value Distribution",
        "Segment Performance Metrics",
        "Customer Loyalty Trends",
        "Revenue Impact Analysis"
    ])

    # Add settings section below analysis options in the sidebar
    st.sidebar.title("Settings")

    # Theme selection
    st.sidebar.subheader("Select Mode")
    theme = st.sidebar.radio("Mode:", ('Light', 'Dark'))

    # Update dark theme background color to Pistachio
    new_dark_theme = {
        'background': 'linear-gradient(135deg, #93c572 0%, #a2d149 100%)',  # Pistachio
        'text_color': '#1e293b'
    }

    # Slightly darken the light theme background color
    light_theme = {
        'background': 'linear-gradient(135deg, #e5e5c4 0%, #e0d68c 100%)',  # Slightly darker beige
        'text_color': '#1e293b'
    }

    # Apply selected theme
    selected_theme = new_dark_theme if theme == 'Dark' else light_theme

    # Update CSS dynamically
    st.markdown(f"""
        <style>
        .stApp {{
            background: {selected_theme['background']};
        }}
        .segment, .data-preview-container, .stats-container, .header, .navbar, .nav-item, .nav-logo {{
            color: {selected_theme['text_color']};
        }}
        </style>
    """, unsafe_allow_html=True)

    # Update page content based on selected language
    translations = get_translations(language)

    # Update all text elements with translations
    # st.markdown(f"""
    # <div class='header'>
    #     <h1>✨ {translations['title']}</h1>
    #     <p>{translations['theme']}</p>
    #     <p>{translations['refresh_rate']}</p>
    #     <p>{translations['notifications']}</p>
    #     <p>{translations['language']}</p>
    # </div>
    # """, unsafe_allow_html=True)

    # Update navbar with translations
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">
            📊 <span style='color: #32CD32;'>RFM Analysis</span>
        </div>
        <div class="nav-items">
            <a href="#dashboard" class="nav-item">
                <i class="fas fa-chart-line"></i> {translations['theme']}
            </a>
            <a href="#customers" class="nav-item">
                <i class="fas fa-users"></i> {translations['refresh_rate']}
            </a>
            <a href="#revenue" class="nav-item">
                <i class="fas fa-dollar-sign"></i> {translations['notifications']}
            </a>
            <a href="#settings" class="nav-item">
                <i class="fas fa-cog"></i> {translations['language']}
            </a>
            <button class="translate-button nav-item" onclick="toggleTranslate()" fdprocessedid="3fmhjf">
                🌐
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Update all graph layouts with black text and better colors
    def update_graph_layout(fig):
        fig.update_layout(
            font=dict(color='black', size=12, family='Poppins'),
            title_font=dict(color='black', size=24),
            plot_bgcolor='rgba(240, 247, 255, 0.5)',
            paper_bgcolor='rgba(240, 247, 255, 0.5)',
            xaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='rgba(0, 0, 0, 0.1)'
            ),
            yaxis=dict(
                title_font=dict(color='black', size=14),
                tickfont=dict(color='black', size=12),
                gridcolor='rgba(0, 0, 0, 0.1)'
            ),
            legend=dict(
                font=dict(color='black', size=12),
                bgcolor='rgba(240, 247, 255, 0.5)'
            )
        )
        return fig

    # Plot based on selection
    if analysis_type == "Customer Segmentation Overview":
        st.markdown("""
            <div class='segment'>
                <h3>Customer Segmentation Overview</h3>
                <p>Discover key insights about your customer segments and their behavior patterns.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Bar chart of segment counts
        fig_bar = px.bar(
            segment_counts, 
            x='RFM_Segment', 
            y='Count',
            color='RFM_Segment',
            color_discrete_sequence=px.colors.qualitative.Bold,
            title='Customer Distribution Across Segments'
        )
        fig_bar = update_graph_layout(fig_bar)
        
        st.plotly_chart(fig_bar, use_container_width=True)

        # Segment-wise metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            champions_pct = len(rfm[rfm['RFM_Segment'] == 'Champions']) / len(rfm) * 100
            st.metric("🏆 Champions", f"{champions_pct:.1f}%", "High Value")
        with col2:
            loyal_pct = len(rfm[rfm['RFM_Segment'] == 'Loyal Customers']) / len(rfm) * 100
            st.metric("💎 Loyal Customers", f"{loyal_pct:.1f}%", "Stable")
        with col3:
            at_risk_pct = len(rfm[rfm['RFM_Segment'].isin(['At Risk', 'Lost'])]) / len(rfm) * 100
            st.metric("⚠ At Risk", f"{at_risk_pct:.1f}%", "Needs Attention")

    elif analysis_type == "Purchase Pattern Analysis":
        st.markdown("""
            <div class='segment'>
                <h3>Purchase Pattern Analysis</h3>
                <p>Understand customer buying behaviors and identify trends in purchase frequency.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Purchase frequency distribution
        purchase_freq = pd.DataFrame(rfm['Frequency'].value_counts()).reset_index()
        purchase_freq.columns = ['Purchase Count', 'Number of Customers']
        
        fig_freq = px.bar(
            purchase_freq,
            x='Purchase Count',
            y='Number of Customers',
            title='Purchase Frequency Distribution',
            color='Number of Customers',
            color_continuous_scale='Viridis'
        )
        fig_freq = update_graph_layout(fig_freq)
        
        st.plotly_chart(fig_freq, use_container_width=True)

        # Purchase timing analysis
        data['Month'] = data['PurchaseDate'].dt.month
        monthly_purchases = data.groupby('Month')['OrderID'].count().reset_index()
        
        fig_monthly = px.line(
            monthly_purchases,
            x='Month',
            y='OrderID',
            title='Monthly Purchase Trends',
            markers=True
        )
        fig_monthly.update_traces(line_color='#1f77b4')
        fig_monthly = update_graph_layout(fig_monthly)
        
        st.plotly_chart(fig_monthly, use_container_width=True)

    elif analysis_type == "Customer Value Distribution":
        st.markdown("""
            <div class='segment'>
                <h3>Customer Value Distribution</h3>
                <p>Analyze monetary value patterns and customer spending behavior.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Monetary value distribution
        fig_monetary = px.histogram(
            rfm,
            x='Monetary',
            nbins=30,
            title='Customer Spending Distribution',
            color_discrete_sequence=['#2575fc']
        )
        fig_monetary = update_graph_layout(fig_monetary)
        
        st.plotly_chart(fig_monetary, use_container_width=True)

        # Value segments - Fix for the pie chart error
        value_segments = pd.qcut(rfm['Monetary'], q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
        value_dist = pd.DataFrame(value_segments.value_counts())
        value_dist.reset_index(inplace=True)
        value_dist.columns = ['Category', 'Count']
        
        fig_value = px.pie(
            value_dist,
            values='Count',
            names='Category',
            title='Customer Value Segments',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_value.update_traces(textfont_color='black')
        fig_value = update_graph_layout(fig_value)
        
        st.plotly_chart(fig_value, use_container_width=True)

    elif analysis_type == "Segment Performance Metrics":
        st.markdown("""
            <div class='segment'>
                <h3>Segment Performance Metrics</h3>
                <p>Track key performance indicators across different customer segments.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Segment performance metrics
        segment_metrics = rfm.groupby('RFM_Segment').agg({
            'Monetary': ['mean', 'sum'],
            'Frequency': 'mean',
            'Recency': 'mean'
        }).round(2)
        
        segment_metrics.columns = ['Avg Spend', 'Total Revenue', 'Avg Frequency', 'Avg Recency']
        segment_metrics = segment_metrics.reset_index()
        
        # Revenue contribution
        fig_revenue = px.bar(
            segment_metrics,
            x='RFM_Segment',
            y='Total Revenue',
            title='Revenue Contribution by Segment',
            color='RFM_Segment',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_revenue = update_graph_layout(fig_revenue)
        
        st.plotly_chart(fig_revenue, use_container_width=True)

    elif analysis_type == "Customer Loyalty Trends":
        st.markdown("""
            <div class='segment'>
                <h3>Customer Loyalty Trends</h3>
                <p>Monitor customer retention and loyalty patterns over time.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Recency distribution
        fig_recency = px.histogram(
            rfm,
            x='Recency',
            nbins=30,
            title='Customer Recency Distribution',
            color_discrete_sequence=['#6a11cb']
        )
        fig_recency = update_graph_layout(fig_recency)
        
        st.plotly_chart(fig_recency, use_container_width=True)

        # Loyalty score calculation
        rfm['Loyalty_Score'] = (rfm['Frequency'] * 0.5 + rfm['Monetary'] * 0.3 + (100 - rfm['Recency']) * 0.2)
        
        fig_loyalty = px.box(
            rfm,
            x='RFM_Segment',
            y='Loyalty_Score',
            title='Loyalty Score by Segment',
            color='RFM_Segment',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_loyalty = update_graph_layout(fig_loyalty)
        
        st.plotly_chart(fig_loyalty, use_container_width=True)

    elif analysis_type == "Revenue Impact Analysis":
        st.markdown("""
            <div class='segment'>
                <h3>Revenue Impact Analysis</h3>
                <p>Analyze revenue patterns and identify high-impact customer segments.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Revenue trends
        monthly_revenue = data.groupby(data['PurchaseDate'].dt.strftime('%Y-%m'))['TransactionAmount'].sum().reset_index()
        
        fig_revenue_trend = px.line(
            monthly_revenue,
            x='PurchaseDate',
            y='TransactionAmount',
            title='Monthly Revenue Trends',
            markers=True
        )
        fig_revenue_trend.update_traces(line_color='#1f77b4')
        fig_revenue_trend = update_graph_layout(fig_revenue_trend)
        
        st.plotly_chart(fig_revenue_trend, use_container_width=True)

        # Segment revenue contribution
        segment_revenue = rfm.groupby('RFM_Segment')['Monetary'].sum().reset_index()
        segment_revenue['Percentage'] = (segment_revenue['Monetary'] / segment_revenue['Monetary'].sum() * 100).round(1)
        
        fig_revenue_pie = px.pie(
            segment_revenue,
            values='Monetary',
            names='RFM_Segment',
            title='Revenue Contribution by Segment',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_revenue_pie.update_traces(textfont_color='black')
        fig_revenue_pie = update_graph_layout(fig_revenue_pie)
        
        st.plotly_chart(fig_revenue_pie, use_container_width=True)

        # Top customer segments with updated styling
        top_segments = segment_revenue.nlargest(3, 'Monetary')
        st.markdown('<h3 class="top-segments-header">🌟 Top Performing Segments</h3>', unsafe_allow_html=True)
        for _, row in top_segments.iterrows():
            st.metric(
                row['RFM_Segment'],
                f"${row['Monetary']:,.2f}",
                f"{row['Percentage']}% of total revenue"
            )

    # Concluding Lines
    st.markdown("""
    <div class='segment'>
        <h3>Thank you for using the RFM Analysis Dashboard</h3>
        <p>We hope this analysis helps you understand your customer segments better and make informed business decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Add anchors for each section to make links functional
    st.markdown("<a id='dashboard'></a>", unsafe_allow_html=True)
    # Dashboard content here

    st.markdown("<a id='customers'></a>", unsafe_allow_html=True)
    # Customers content here

    st.markdown("<a id='revenue'></a>", unsafe_allow_html=True)
    # Revenue content here

    st.markdown("<a id='settings'></a>", unsafe_allow_html=True)
    # Settings content here

    st.markdown("""
    <div id="google_translate_element"></div>
    <script>
      // Initialize Google Translate
      function googleTranslateElementInit() {
        new google.translate.TranslateElement(
          {
            pageLanguage: "en",
            includedLanguages: "en,hi,es,fr,de,it,ja,zh-CN",
            layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
            autoDisplay: false,
          },
          "google_translate_element"
        );
      }

      // Show or hide the Google Translate options
      function toggleTranslate() {
        const translateElement = document.getElementById(
          "google_translate_element"
        );
        if (
          translateElement.style.display === "none" ||
          translateElement.style.display === ""
        ) {
          translateElement.style.display = "block";
        } else {
          translateElement.style.display = "none";
        }
      }
    </script>
    <script type="text/javascript" src="//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"></script>
    """, unsafe_allow_html=True)

# Customers Analysis page
def show_customers_analysis():
    st.title("👥 Customer Analysis")
    
    # Load the data
    df = pd.read_csv('rfm_data.csv')
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    
    # Calculate customer metrics
    customer_metrics = df.groupby('CustomerID').agg({
        'OrderID': 'count',
        'TransactionAmount': 'sum',
        'PurchaseDate': lambda x: (df['PurchaseDate'].max() - x.max()).days
    }).reset_index()
    
    customer_metrics.columns = ['CustomerID', 'Total_Orders', 'Total_Spent', 'Days_Since_Last_Purchase']
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(customer_metrics):,}",
            delta=f"{len(customer_metrics[customer_metrics['Days_Since_Last_Purchase'] <= 30])} active customers"
        )
    
    with col2:
        avg_orders = customer_metrics['Total_Orders'].mean()
        st.metric(
            label="Average Orders per Customer",
            value=f"{avg_orders:.1f}",
            delta=f"{((customer_metrics['Total_Orders'] > avg_orders).mean() * 100):.1f}% above average"
        )
    
    with col3:
        avg_spent = customer_metrics['Total_Spent'].mean()
        st.metric(
            label="Average Customer Value",
            value=f"${avg_spent:,.2f}",
            delta=f"{((customer_metrics['Total_Spent'] > avg_spent).mean() * 100):.1f}% above average"
        )
    
    # Customer Segments Analysis
    st.subheader("Customer Segments Analysis")
    
    # Define customer segments based on spending
    customer_metrics['Segment'] = pd.qcut(
        customer_metrics['Total_Spent'],
        q=4,
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segment distribution
        segment_counts = customer_metrics['Segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Distribution by Segment'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average value by segment
        segment_avg = customer_metrics.groupby('Segment')['Total_Spent'].mean()
        fig = px.bar(
            x=segment_avg.index,
            y=segment_avg.values,
            title='Average Customer Value by Segment'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Activity Timeline
    st.subheader("Customer Activity Timeline")
    
    # Monthly customer activity
    monthly_activity = df.groupby(df['PurchaseDate'].dt.strftime('%Y-%m')).agg({
        'CustomerID': 'nunique',
        'OrderID': 'count',
        'TransactionAmount': 'sum'
    }).reset_index()
    
    monthly_activity.columns = ['Month', 'Active_Customers', 'Total_Orders', 'Total_Revenue']
    
    fig = px.line(
        monthly_activity,
        x='Month',
        y=['Active_Customers', 'Total_Orders'],
        title='Monthly Customer Activity',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Customers Table
    st.subheader("Top 10 Customers")
    top_customers = customer_metrics.nlargest(10, 'Total_Spent')
    st.dataframe(top_customers)

# Revenue Analysis page
def show_revenue_analysis():
    st.title("💰 Revenue Analysis")
    
    # Load the data
    df = pd.read_csv('rfm_data.csv')
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    
    # Calculate revenue metrics
    revenue_metrics = df.groupby(df['PurchaseDate'].dt.strftime('%Y-%m')).agg({
        'TransactionAmount': ['sum', 'mean', 'count']
    }).reset_index()
    
    revenue_metrics.columns = ['Month', 'Total_Revenue', 'Average_Order_Value', 'Number_of_Orders']
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_revenue = df['TransactionAmount'].sum()
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.2f}",
            delta=f"{((revenue_metrics['Total_Revenue'].iloc[-1] > revenue_metrics['Total_Revenue'].iloc[-2]).astype(int) * 100)}% vs last month"
        )
    
    with col2:
        avg_order_value = df['TransactionAmount'].mean()
        st.metric(
            label="Average Order Value",
            value=f"${avg_order_value:,.2f}",
            delta=f"{((revenue_metrics['Average_Order_Value'].iloc[-1] > revenue_metrics['Average_Order_Value'].iloc[-2]).astype(int) * 100)}% vs last month"
        )
    
    with col3:
        total_orders = len(df)
        st.metric(
            label="Total Orders",
            value=f"{total_orders:,}",
            delta=f"{((revenue_metrics['Number_of_Orders'].iloc[-1] > revenue_metrics['Number_of_Orders'].iloc[-2]).astype(int) * 100)}% vs last month"
        )
    
    # Revenue Trends
    st.subheader("Revenue Trends")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly revenue trend
        fig = px.line(
            revenue_metrics,
            x='Month',
            y='Total_Revenue',
            title='Monthly Revenue Trend',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average order value trend
        fig = px.line(
            revenue_metrics,
            x='Month',
            y='Average_Order_Value',
            title='Average Order Value Trend',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue Distribution
    st.subheader("Revenue Distribution")
    
    # Daily revenue distribution
    daily_revenue = df.groupby(df['PurchaseDate'].dt.date)['TransactionAmount'].sum().reset_index()
    
    fig = px.histogram(
        daily_revenue,
        x='TransactionAmount',
        nbins=30,
        title='Daily Revenue Distribution'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Revenue Days
    st.subheader("Top 10 Revenue Days")
    top_days = daily_revenue.nlargest(10, 'TransactionAmount')
    st.dataframe(top_days)

# ML Analysis page
def show_ml_analysis():
    st.title("🤖 Machine Learning Analysis")
    
    # Add custom CSS with animations
    st.markdown("""
        <style>
        .ml-container {
            animation: fadeIn 0.8s ease-out;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        
        .ml-header {
            color: #4B0082;
            margin-bottom: 1rem;
            border-bottom: 2px solid #4B0082;
            padding-bottom: 0.5rem;
        }
        
        .info-box {
            background: rgba(75, 0, 130, 0.05);
            border-left: 4px solid #4B0082;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load data
    file_path = 'rfm_data.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'rfm_data.csv' exists in the current directory.")
        return
    
    # Convert PurchaseDate to datetime
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'])
    
    # Define reference date for recency calculation
    reference_date = dt.datetime(2023, 7, 1)
    
    # Calculate RFM metrics
    rfm = data.groupby('CustomerID').agg({
        'PurchaseDate': lambda x: (reference_date - x.max()).days,
        'OrderID': 'count',
        'TransactionAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Filter out non-positive monetary values
    rfm = rfm[rfm['Monetary'] > 0]
    
    # Add additional features for ML
    customer_data = data.groupby('CustomerID').agg({
        'PurchaseDate': [
            lambda x: (x.max() - x.min()).days if len(x) > 1 else 0,  # Customer tenure
            'count'  # Total transactions
        ],
        'TransactionAmount': [
            'mean',  # Average order value
            'std',   # Variability in spending
            'sum'    # Total spending
        ],
        'ProductInformation': [
            lambda x: x.nunique(),  # Product variety
            'count'  # Total products purchased
        ]
    }).reset_index()
    
    # Flatten the column names
    customer_data.columns = ['CustomerID', 'Tenure', 'TransactionCount', 
                            'AvgOrderValue', 'SpendingStd', 'TotalSpending',
                            'ProductVariety', 'TotalProducts']
    
    # Replace NaN values in SpendingStd with 0 (for customers with only one transaction)
    customer_data['SpendingStd'].fillna(0, inplace=True)
    
    # Merge with RFM data
    ml_data = pd.merge(rfm, customer_data, on='CustomerID')
    
    # Create tabs for different ML analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Advanced Segmentation", 
        "Churn Prediction", 
        "Customer Lifetime Value", 
        "Next Purchase Prediction",
        "Time Series Forecasting",
        "Enhanced Segmentation",
        "Customer Behavior Modeling"
    ])
    
    with tab1:
        st.markdown('<h3 class="ml-header">Advanced Customer Segmentation with K-Means</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        K-Means clustering provides a more data-driven approach to customer segmentation compared to rule-based RFM segmentation.
        This can reveal natural groupings in your customer base that might not be apparent with traditional methods.
        </div>
        """, unsafe_allow_html=True)
        
        # Features for clustering
        cluster_features = ['Recency', 'Frequency', 'Monetary', 'Tenure', 'ProductVariety']
        
        # Allow user to select features
        selected_features = st.multiselect(
            "Select features for clustering:",
            options=cluster_features,
            default=cluster_features
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for clustering.")
        else:
            # Number of clusters
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=5)
            
            # Prepare data for clustering
            X = ml_data[selected_features].copy()
            
            # Scale the data
            scaler = StandardSca
