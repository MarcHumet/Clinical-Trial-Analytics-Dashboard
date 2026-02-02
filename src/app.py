import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import subprocess
import shlex
import time
import requests
import importlib.util
import json

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

# Database connection with retry for the seed check
def get_db_connection_with_retry(retries=3, delay=1):
    for attempt in range(retries):
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
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None
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

    st.markdown("---")
    st.subheader("Initialize / Seed Database")
    st.write("Use the button below to run the database population script. The app will download data from the ClinicalTrials.gov API and populate the database.")
    st.info("Data is downloaded from the API and can take up to 3 minutes to complete.")

    # Check if database is empty
    def is_db_empty():
        try:
            conn = get_db_connection_with_retry(retries=3, delay=0.5)
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT COUNT(*) as total FROM studies")
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                return (result['total'] if result else 0) == 0
            return True  # Assume empty if can't connect
        except Exception as e:
            return True  # Assume empty if can't check

    db_empty = is_db_empty()

    if not db_empty:
        st.success("✓ Database is already populated with clinical trial data.")
        st.info("To refresh the data, you may need to delete the existing data first or use the CLI tool.")
    else:
        st.warning("Database is empty. Click the button below to download clinical trial data from the ClinicalTrials.gov API (~10000 studies across ~100 pages).")
        
        # Download up to 100 pages (~10000 studies) if DB is empty
        seed_mode = "api"
        seed_count = 100

    def run_seed_in_app(max_pages: int = 100, page_size: int = 100):
        api_url = os.getenv('API_URL', 'https://clinicaltrials.gov/api/v2/studies')
        all_studies = []
        page_token = None
        page = 0

        progress_bar = st.progress(0)
        status = st.empty()
        elapsed_place = st.empty()

        start_time = time.time()

        while True:
            params = {"pageSize": page_size}
            if page_token:
                params["pageToken"] = page_token

            try:
                resp = requests.get(api_url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                status.error(f"Error fetching page {page+1}: {e}")
                break

            studies = payload.get("studies") or payload.get("data") or []
            all_studies.extend(studies)

            page += 1
            status.info(f"Fetched page {page} — {len(studies)}  (total {len(all_studies)} studies)")

            # Update progress bar and elapsed time
            progress = min(page / max_pages, 1.0)
            progress_bar.progress(progress)
            elapsed = time.time() - start_time
            if page > 0:
                est_total = elapsed / page * max_pages
                eta = max(0, est_total - elapsed)
                elapsed_place.text(f"Elapsed: {int(elapsed)}s — ETA: {int(eta)}s")
            else:
                elapsed_place.text(f"Elapsed: {int(elapsed)}s")

            next_token = payload.get("nextPageToken")
            if not next_token:
                break

            page_token = next_token
            if page >= max_pages:
                status.info("Reached max_pages limit")
                break

            time.sleep(0.1)

        # Save to file
        json_file = "studies_api.json"
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({"studies": all_studies}, f, ensure_ascii=False, indent=2)
            status.success(f"Saved {len(all_studies)} studies to {json_file}")
        except Exception as e:
            status.error(f"Error writing {json_file}: {e}")
            return 1, f"write_error: {e}"

        # Load and call populate_from_json from the existing script
        try:
            spec = importlib.util.spec_from_file_location("ddbb_populate", "ddbb/create_and_poblate_ddbb.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            engine, Session = module.create_engine_and_session()
            if not engine or not Session:
                status.error("Failed to create database engine")
                return 1, "engine_error"

            session = Session()
            with st.spinner("Populating database from downloaded JSON..."):
                module.create_schema(engine)
                module.populate_from_json(session, json_file)
            session.close()
            status.success("Database population completed")
            
            # Clean up JSON file after successful population
            try:
                import os as os_module
                if os_module.path.exists(json_file):
                    os_module.remove(json_file)
                    status.info(f"Cleaned up {json_file}")
            except Exception as cleanup_error:
                status.warning(f"Could not delete {json_file}: {cleanup_error}")
            
            return 0, "ok"
        except Exception as e:
            status.error(f"Error populating DB: {e}")
            return 1, str(e)

    if db_empty:
        if st.button("Fill DB"):
            with st.spinner("Starting download and populate..."):
                code, output = run_seed_in_app(max_pages=seed_count, page_size=100)
            if code == 0:
                st.success("DB population finished.")
                st.rerun()
            else:
                st.error("DB population failed. See logs above.")

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

# Add a "Clear Database" button in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Admin")

def clear_database():
    """Clear all data from the database"""
    try:
        conn = get_db_connection_with_retry(retries=3, delay=0.5)
        if not conn:
            st.error("Could not connect to database to clear data")
            return False
        
        cursor = conn.cursor()
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        for table in ['study_design', 'locations', 'sponsors', 'outcomes', 'interventions', 'conditions', 'studies']:
            try:
                cursor.execute(f"TRUNCATE TABLE {table}")
            except:
                pass
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False

if st.sidebar.button("🗑️ Clear All Data"):
    with st.spinner("Clearing database..."):
        if clear_database():
            st.sidebar.success("Database cleared successfully")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Failed to clear database")
