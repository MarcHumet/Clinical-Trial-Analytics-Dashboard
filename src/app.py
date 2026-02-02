import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Clinical Trial Analytics Dashboard", layout="wide")

# Title and description
st.title("🏥 Clinical Trial Analytics Dashboard")
st.markdown("---")
st.markdown("Welcome to the Clinical Trial Analytics Dashboard. Here you can explore and analyze clinical trial data.")

# Database connection function
@st.cache_resource
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'mysql'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'user'),
            password=os.getenv('DB_PASSWORD', 'pass'),
            database=os.getenv('DB_NAME', 'clinicaltrials')
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Data Overview", "Analysis", "Search Studies"])

if page == "Home":
    st.subheader("Welcome to the Clinical Trial Analytics Dashboard")
    st.write("""
    This dashboard provides:
    - **Data Overview**: View comprehensive statistics about clinical trials
    - **Analysis**: Perform detailed analysis on trial data
    - **Search Studies**: Find and explore specific clinical trials
    """)

elif page == "Data Overview":
    st.subheader("Data Overview")
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) as total_studies FROM studies LIMIT 1")
            result = cursor.fetchone()
            st.metric("Total Studies", result['total_studies'] if result else 0)
            
            # Display sample data
            cursor.execute("SELECT * FROM studies LIMIT 10")
            df = pd.DataFrame(cursor.fetchall())
            st.dataframe(df, use_container_width=True)
            cursor.close()
        except Error as e:
            st.error(f"Error fetching data: {e}")
        finally:
            conn.close()
    else:
        st.error("Could not connect to the database")

elif page == "Analysis":
    st.subheader("Analysis")
    st.write("Analysis tools and visualizations will appear here.")
    # Add your analysis logic here

elif page == "Search Studies":
    st.subheader("Search Clinical Studies")
    search_query = st.text_input("Enter a search term:")
    
    if search_query:
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                query = f"""
                SELECT * FROM studies 
                WHERE title LIKE %s OR description LIKE %s 
                LIMIT 50
                """
                cursor.execute(query, (f"%{search_query}%", f"%{search_query}%"))
                results = cursor.fetchall()
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No studies found matching your query.")
                
                cursor.close()
            except Error as e:
                st.error(f"Error searching studies: {e}")
            finally:
                conn.close()
        else:
            st.error("Could not connect to the database")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 Clinical Trial Analytics Dashboard")
