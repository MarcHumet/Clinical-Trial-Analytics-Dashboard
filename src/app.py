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
import sys
from pathlib import Path
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from loguru import logger
import numpy as np
from datetime import datetime, date
from sqlalchemy import create_engine, text

# Ensure project root is on sys.path for ddbb imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv()
seed_count = int(os.getenv('page_size_limit') or 10)
st.set_page_config(page_title="Clinical Trial Analytics Dashboard", layout="wide")

# Function to apply conditional coloring to availability data
def highlight_availability(val):
    """Apply text color based on data availability percentage"""
    if isinstance(val, str) and '%' in val:
        try:
            percentage = float(val.rstrip('%'))
            if percentage >= 95:
                return 'color: #00FF00'  # Green text
            elif percentage > 10:
                return 'color: #FFD700'  # Yellow text
            else:
                return 'color: #FF6B6B'  # Red text
        except:
            return ''
    return ''

# Function to setup dark theme for all plots
def setup_dark_theme():
    """Configure dark theme for matplotlib/seaborn plots"""
    sns.set_style("darkgrid")
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#000000'
    plt.rcParams['axes.facecolor'] = '#000000'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['lines.color'] = 'white'

# Enrollment Success Analytics Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_enrollment_success_data():
    """Load and process enrollment success data from MySQL database"""
    
    # Database connection configuration
    mysql_url = os.getenv('MYSQL_DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URL')
    
    if not mysql_url:
        mysql_user = os.getenv('MYSQL_USER') or os.getenv('DB_USER') or 'user'
        mysql_password = os.getenv('MYSQL_PASSWORD') or os.getenv('DB_PASSWORD') or 'pass'
        # Prefer Docker environment variables, then fall back to service name 'mysql'
        mysql_host = os.getenv('DB_HOST') or os.getenv('MYSQL_HOST') or 'mysql'
        mysql_port = os.getenv('DB_PORT') or os.getenv('MYSQL_PORT') or '3306'
        mysql_db = os.getenv('DB_NAME') or os.getenv('MYSQL_DATABASE') or 'clinicaltrials'
        
        mysql_url = (
            f"mysql+pymysql://{mysql_user}:{mysql_password}@"
            f"{mysql_host}:{mysql_port}/{mysql_db}"
        )
    
    try:
        engine = create_engine(mysql_url)
        with engine.connect() as conn:
            # Load studies data with enrollment metrics
            query = """
            SELECT 
                s.nct_id,
                s.title,
                s.status,
                s.phase,
                s.study_type,
                s.start_date,
                s.completion_date,
                s.primary_completion_date,
                s.enrollment,
                s.enrollment_type,
                s.brief_summary,
                s.gender,
                s.minimum_age,
                s.maximum_age,
                COUNT(DISTINCT c.condition_id) as condition_count,
                COUNT(DISTINCT i.intervention_id) as intervention_count,
                COUNT(DISTINCT l.facility) as location_count,
                GROUP_CONCAT(DISTINCT c.condition_name SEPARATOR '; ') as conditions,
                GROUP_CONCAT(DISTINCT l.country SEPARATOR '; ') as countries
            FROM studies s
            LEFT JOIN conditions c ON s.study_id = c.study_id
            LEFT JOIN interventions i ON s.study_id = i.study_id
            LEFT JOIN locations l ON s.study_id = l.study_id
            GROUP BY s.study_id
            HAVING s.enrollment IS NOT NULL AND s.enrollment > 0
            """
            
            df = pd.read_sql_query(text(query), conn)
            
        # Calculate enrollment metrics
        df = calculate_enrollment_metrics(df)
        return df
        
    except Exception as e:
        st.warning(f"Database connection failed: {str(e)}")
        st.info("Attempting to load from CSV file...")
        
        # Try to load from CSV file as fallback
        try:
            csv_path = Path(__file__).parent / 'results' / 'enrollment_success_metrics.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                st.success("Successfully loaded enrollment success data from CSV file!")
                return df
            else:
                st.error("CSV file not found. Please run the database population script first.")
                return pd.DataFrame()
        except Exception as csv_error:
            st.error(f"Error loading CSV file: {str(csv_error)}")
            return pd.DataFrame()

def calculate_enrollment_metrics(df):
    """Calculate various enrollment success metrics"""
    
    # Convert dates
    date_columns = ['start_date', 'completion_date', 'primary_completion_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate enrollment rate (participants per month)
    df['duration_days'] = (df['completion_date'] - df['start_date']).dt.days
    df['duration_months'] = df['duration_days'] / 30.44
    df['duration_months'] = df['duration_months'].clip(lower=1)  # Minimum 1 month
    
    # Enrollment rate
    df['enrollment_rate'] = df['enrollment'] / df['duration_months']
    df['enrollment_rate'] = df['enrollment_rate'].fillna(0)
    
    # Enrollment rate tiers
    def get_rate_tier(rate):
        if pd.isna(rate) or rate == 0:
            return 'Unknown'
        elif rate > 50:
            return 'Excellent (>50/month)'
        elif rate >= 10:
            return 'Good (10-50/month)'
        elif rate >= 1:
            return 'Adequate (1-10/month)'
        else:
            return 'Slow (<1/month)'
    
    df['enrollment_rate_tier'] = df['enrollment_rate'].apply(get_rate_tier)
    
    # Success percentage vs benchmark (10 participants/month)
    expected_enrollment = df['duration_months'] * 10
    df['success_percentage'] = (df['enrollment'] / expected_enrollment) * 100
    df['success_percentage'] = df['success_percentage'].clip(upper=200)  # Cap at 200%
    
    # Success tiers
    def get_success_tier(pct):
        if pd.isna(pct):
            return 'Unknown'
        elif pct >= 100:
            return 'Exceeded (≥100%)'
        elif pct >= 75:
            return 'Met (75-100%)'
        elif pct >= 50:
            return 'Below (50-75%)'
        else:
            return 'Significantly Below (<50%)'
    
    df['success_tier'] = df['success_percentage'].apply(get_success_tier)
    
    # Status success score
    status_scores = {
        'COMPLETED': 100,
        'ACTIVE_NOT_RECRUITING': 85,
        'RECRUITING': 60,
        'ENROLLING_BY_INVITATION': 55,
        'NOT_YET_RECRUITING': 30,
        'SUSPENDED': 20,
        'TERMINATED': 10,
        'WITHDRAWN': 5,
        'UNKNOWN': 50,
    }
    df['status_score'] = df['status'].map(status_scores).fillna(50)
    
    # Composite success score (0-100)
    # Factor 1: Enrollment completeness (40%)
    enrollment_score = np.minimum(df['enrollment'] / 500 * 40, 40)  # Max 40 points for 500+ participants
    
    # Factor 2: Status success (30%)
    status_component = df['status_score'] * 0.3
    
    # Factor 3: Temporal efficiency (20%)
    rate_component = np.minimum(df['enrollment_rate'] / 50 * 20, 20)  # Max 20 points for 50+/month
    
    # Factor 4: Data completeness (10%)
    data_completeness = (
        (df['brief_summary'].notna()).astype(int) * 2 +
        (df['condition_count'] > 0).astype(int) * 3 +
        (df['intervention_count'] > 0).astype(int) * 2 +
        (df['location_count'] > 0).astype(int) * 3
    )
    data_component = data_completeness  # Max 10 points
    
    df['composite_score'] = enrollment_score + status_component + rate_component + data_component
    
    # Percentile rankings
    df['success_percentile'] = df['composite_score'].rank(pct=True) * 100
    
    return df

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
st.sidebar.title("Section Selector")
page = st.sidebar.radio("Select a page:", ["Home", "Data Overview & Completeness", "Distribution Analysis", "Time Trends", "Enrollment Success Analytics"])

if page == "Home":
    st.subheader("Welcome to the Clinical Trial Analytics Dashboard")
    st.write("""
    This dashboard provides:
    - **Home**: to populate the database with clinical trial data from the ClinicalTrials.gov API 
    - **Data Overview & Completeness**: View comprehensive statistics about clinical trials
    - **Distribution Analysis**: Perform detailed analysis on trial data
    - **Time Trends**: Explore trends over time in clinical trial data
    - **Enrollment Success Analytics**: Analyze enrollment success metrics for clinical trials
             
            First time access you should poblate by pressing button  "Fill MySQL DB"
    """)

    if st.session_state.get("seed_summary"):
        st.info(st.session_state["seed_summary"])
        # Clear after showing once
        del st.session_state["seed_summary"]

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
        # seed_count = seed_count

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
            
            # Run enrollment success analysis after successful population
            with st.spinner("Calculating enrollment success metrics..."):
                try:
                    import subprocess
                    import sys
                    result = subprocess.run(
                        [sys.executable, "src/enrollment_success.py"],
                        cwd=".",
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout
                    )
                    if result.returncode == 0:
                        status.success("✓ Enrollment success metrics calculated and saved")
                        # Show a summary of the results
                        if "Analysis Complete" in result.stdout:
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if "studies analyzed" in line:
                                    status.info(line)
                                    break
                    else:
                        status.warning(f"Enrollment success calculation completed with warnings")
                        if result.stderr:
                            st.text(f"Details: {result.stderr[:200]}...")
                except subprocess.TimeoutExpired:
                    status.warning("Enrollment success calculation timed out but database population was successful")
                except Exception as enrollment_error:
                    status.warning(f"Database populated successfully, but enrollment analysis failed: {str(enrollment_error)[:100]}")

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
        if st.button("Fill MySQL DB"):
            with st.spinner("Starting download and populate..."):
                code, output = run_seed_in_app(max_pages=seed_count, page_size=100)
            if code == 0:
                st.success("DB population finished.")
                st.rerun()
            else:
                st.error("DB population failed. See logs above.")

elif page == "Data Overview & Completeness":
    st.subheader("📋 Data Overview & Completeness")
    conn = get_db_connection_with_retry(retries=3, delay=0.5)
    if not conn:
        st.error("Could not connect to the database. Please ensure MySQL is running and accessible.")
    else:
        # Table selector
        st.write("Select a table to view:")
        table_options = {
            "Studies": "studies",
            "Conditions": "conditions",
            "Interventions": "interventions",
            "Outcomes": "outcomes",
            "Sponsors": "sponsors",
            "Locations": "locations",
            "Study Design": "study_design"
        }
        
        selected_table_name = st.selectbox("Choose table:", list(table_options.keys()), key="overview_table")
        selected_table = table_options[selected_table_name]
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Get table stats
            cursor.execute(f"SELECT COUNT(*) as count FROM {selected_table}")
            row_count = cursor.fetchone()['count']
            
            st.metric(f"Total rows in {selected_table_name}", row_count)
            
            # Display raw data
            st.subheader(f"Raw Data - {selected_table_name}")
            cursor.execute(f"SELECT * FROM {selected_table} LIMIT 100")
            df = pd.DataFrame(cursor.fetchall())
            st.dataframe(df, use_container_width=True)
            
            # Data Availability Report
            st.subheader(f"📊 Data Availability Report - {selected_table_name}")
            
            # Get column information
            cursor.execute(f"DESCRIBE {selected_table}")
            columns_info = cursor.fetchall()
            column_names = [col['Field'] if isinstance(col, dict) else col[0] for col in columns_info]
            
            # Calculate data availability for each column
            availability_data = []
            for col in column_names:
                try:
                    cursor.execute(f"SELECT COUNT(*) as total, SUM(CASE WHEN `{col}` IS NULL THEN 1 ELSE 0 END) as null_count FROM {selected_table}")
                    result = cursor.fetchone()
                    total = result['total'] if isinstance(result, dict) else result[0]
                    null_count = result['null_count'] if isinstance(result, dict) else result[1]
                    null_count = null_count if null_count is not None else 0
                    non_null = total - null_count
                    percentage = (non_null / total * 100) if total > 0 else 0
                    
                    availability_data.append({
                        "Column": col,
                        "Total Records": total,
                        "Non-NULL": non_null,
                        "NULL": null_count,
                        "Data Availability %": f"{percentage:.1f}%"
                    })
                except Exception as e:
                    st.warning(f"Could not analyze column {col}: {e}")
            
            if availability_data:
                availability_df = pd.DataFrame(availability_data)
                # Apply conditional coloring to the Data Availability % column
                styled_df = availability_df.style.applymap(lambda val: highlight_availability(val) if isinstance(val, str) and '%' in val else '', subset=['Data Availability %'])
                st.write(styled_df)
                
                # Summary stats
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                avg_availability = sum([float(x["Data Availability %"].rstrip("%")) for x in availability_data]) / len(availability_data)
                
                with col_summary1:
                    st.metric("Total Columns", len(column_names))
                with col_summary2:
                    st.metric("Average Data Availability", f"{avg_availability:.1f}%")
                with col_summary3:
                    try:
                        cursor.execute(f"SELECT COUNT(*) as complete_count FROM {selected_table} WHERE " + " AND ".join([f"`{c}` IS NOT NULL" for c in column_names]))
                        complete_result = cursor.fetchone()
                        complete_records = complete_result['complete_count'] if isinstance(complete_result, dict) else complete_result[0]
                        st.metric("Complete Records", complete_records)
                    except:
                        st.metric("Complete Records", "N/A")
            
            cursor.close()
        except Error as e:
            st.error(f"Error fetching data: {e}")
        finally:
            conn.close()

elif page == "Distribution Analysis":
    st.subheader("📊 Data Distribution Analysis")
    
    conn = get_db_connection_with_retry(retries=3, delay=0.5)
    if not conn:
        st.error("Could not connect to the database. Please ensure MySQL is running and accessible.")
    else:
        # Table selector
        st.write("Select a table to analyze:")
        table_options = {
            "Studies": "studies",
            "Conditions": "conditions",
            "Interventions": "interventions",
            "Outcomes": "outcomes",
            "Sponsors": "sponsors",
            "Locations": "locations",
            "Study Design": "study_design"
        }
        
        selected_table_name = st.selectbox("Choose table:", list(table_options.keys()))
        selected_table = table_options[selected_table_name]
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Get table summary statistics
            cursor.execute(f"SELECT COUNT(*) as count FROM {selected_table}")
            row_count = cursor.fetchone()['count']
            
            st.metric(f"Total rows in {selected_table_name}", row_count)
            
            # Table Summary Statistics
            st.subheader(f"📊 Table Summary - {selected_table_name}")
            
            # Get column information
            cursor.execute(f"DESCRIBE {selected_table}")
            columns_info = cursor.fetchall()
            column_names = [col['Field'] if isinstance(col, dict) else col[0] for col in columns_info]
            
            # Define key columns for each table to show top 5 values
            key_columns_map = {
                "studies": ["status", "phase", "study_type", "gender"],
                "conditions": ["condition_name"],
                "interventions": ["intervention_type", "name"],
                "outcomes": ["outcome_type"],
                "sponsors": ["agency", "agency_class"],
                "locations": ["country", "city"],
                "study_design": ["allocation", "intervention_model", "masking", "primary_purpose"]
            }
            
            key_columns = key_columns_map.get(selected_table, [])
            
            # Create summary table
            summary_data = []
            for col in key_columns:
                if col in column_names:
                    try:
                        # Get total and distinct counts
                        cursor.execute(f"SELECT COUNT(*) as total, COUNT(DISTINCT `{col}`) as distinct_count FROM {selected_table} WHERE `{col}` IS NOT NULL")
                        counts = cursor.fetchone()
                        total_records = counts['total'] if isinstance(counts, dict) else counts[0]
                        distinct_records = counts['distinct_count'] if isinstance(counts, dict) else counts[1]
                        
                        # Get top 5 most frequent values
                        cursor.execute(f"SELECT `{col}`, COUNT(*) as count FROM {selected_table} WHERE `{col}` IS NOT NULL GROUP BY `{col}` ORDER BY count DESC LIMIT 5")
                        top_values = cursor.fetchall()
                        top_5_str = ", ".join([f"{row[col] if isinstance(row, dict) else row[0]} ({row['count'] if isinstance(row, dict) else row[1]})" for row in top_values])
                        
                        summary_data.append({
                            "Column": col,
                            "Total Records": total_records,
                            "Distinct Values": distinct_records,
                            "Top 5 Most Frequent": top_5_str
                        })
                    except Exception as e:
                        st.warning(f"Could not analyze column {col}: {e}")
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No key columns found for summary analysis.")
            
            # Visualizations based on selected table
            st.subheader(f"Visualizations - {selected_table_name}")
            
            if selected_table == "studies":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Status distribution - Plotly Bar
                    cursor.execute("SELECT status, COUNT(*) as count FROM studies WHERE status IS NOT NULL GROUP BY status ORDER BY count DESC")
                    status_data = pd.DataFrame(cursor.fetchall())
                    if not status_data.empty:
                        fig = px.bar(status_data, x='status', y='count', title="Study Status Distribution (Bar Chart)")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:                    
                    # Phase distribution - Plotly
                    cursor.execute("SELECT phase, COUNT(*) as count FROM studies WHERE phase IS NOT NULL GROUP BY phase ORDER BY count DESC")
                    phase_data = pd.DataFrame(cursor.fetchall())
                    if not phase_data.empty:
                        fig = px.bar(phase_data, x='phase', y='count', title="Study Phase Distribution")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                     
                    # Gender distribution chart
                    # Gender distribution - Plotly pie chart
                    cursor.execute("SELECT gender, COUNT(*) as count FROM studies WHERE gender IS NOT NULL GROUP BY gender ORDER BY count DESC")
                    gender_data = pd.DataFrame(cursor.fetchall())
                    if not gender_data.empty:
                        fig = px.pie(gender_data, values='count', names='gender', title="Gender Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                
                with col4:
                    pass   
                
                # Enrollment statistics with distribution plot
                cursor.execute("SELECT enrollment FROM studies WHERE enrollment IS NOT NULL")
                enrollment_data = pd.DataFrame(cursor.fetchall())
                if not enrollment_data.empty:
                    col5, col6 = st.columns(2)
                    with col5:
                        cursor.execute("SELECT AVG(enrollment) as avg_enrollment, MIN(enrollment) as min, MAX(enrollment) as max, COUNT(*) as total_studies FROM studies WHERE enrollment IS NOT NULL")
                        enrollment_stats = cursor.fetchone()
                        # Get count of studies with 0 enrollment
                        cursor.execute("SELECT COUNT(*) as zero_enrollment FROM studies WHERE enrollment = 0")
                        zero_enrollment_result = cursor.fetchone()
                        zero_enrollment = zero_enrollment_result['zero_enrollment'] if isinstance(zero_enrollment_result, dict) else zero_enrollment_result[0]
                        
                        if enrollment_stats:
                            st.metric("Average Enrollment", f"{enrollment_stats['avg_enrollment']:.0f}")
                            st.metric("Minimum Enrollment", f"{enrollment_stats['min']}")
                            st.metric("Maximum Enrollment", f"{enrollment_stats['max']}")
                            st.metric("Studies with 0 Enrollment", f"{zero_enrollment}")
                            st.caption(f"Total studies with enrollment data: {enrollment_stats['total_studies']}")
                    
                    with col6:
                        # Seaborn histogram with KDE
                        setup_dark_theme()
                        fig, ax = plt.subplots(figsize=(8, 5))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        sns.histplot(data=enrollment_data, x='enrollment', kde=True, bins=30, ax=ax, color="skyblue")
                        ax.set_xlim(0, 10000)  # Set scale from 0 to 10,000
                        ax.set_title("Enrollment Distribution (0-10,000 scale)", color='white', fontsize=12, fontweight='bold')
                        ax.set_xlabel("Enrollment Count", color='white')
                        ax.set_ylabel("Frequency", color='white')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        st.pyplot(fig)
               
            
            elif selected_table == "conditions":
                # Top conditions - Horizontal bar with dark theme
                cursor.execute("SELECT condition_name, COUNT(*) as count FROM conditions GROUP BY condition_name ORDER BY count DESC LIMIT 15")
                top_conditions = pd.DataFrame(cursor.fetchall())
                if not top_conditions.empty:
                    setup_dark_theme()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.patch.set_facecolor('#000000')
                    ax.set_facecolor('#000000')
                    top_cond_sorted = top_conditions.sort_values('count')
                    sns.barplot(data=top_cond_sorted, y='condition_name', x='count', palette="coolwarm", ax=ax)
                    ax.set_title("Top 15 Conditions", color='white', fontsize=14, fontweight='bold')
                    ax.set_xlabel("Count", color='white', fontsize=12)
                    ax.set_ylabel("Condition Name", color='white', fontsize=12)
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_color('white')
                    ax.grid(axis='x', color='white', alpha=0.2, linestyle='--')
                    st.pyplot(fig)
            
            elif selected_table == "interventions":
                # Intervention type distribution
                cursor.execute("SELECT intervention_type, COUNT(*) as count FROM interventions WHERE intervention_type IS NOT NULL GROUP BY intervention_type ORDER BY count DESC")
                intervention_data = pd.DataFrame(cursor.fetchall())
                if not intervention_data.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(intervention_data, x='intervention_type', y='count', title="Intervention Type Distribution (Bar)")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Seaborn lollipop chart style
                        setup_dark_theme()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        intervention_data_sorted = intervention_data.sort_values('count')
                        y_pos = range(len(intervention_data_sorted))
                        ax.hlines(y=y_pos, xmin=0, xmax=intervention_data_sorted['count'], color='white', alpha=0.4, linewidth=2)
                        ax.scatter(intervention_data_sorted['count'], y_pos, color='steelblue', s=100, alpha=1, zorder=3)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(intervention_data_sorted['intervention_type'], color='white')
                        ax.set_xlabel("Count", color='white', fontsize=12)
                        ax.set_title("Intervention Types (Lollipop Chart)", color='white', fontsize=12, fontweight='bold')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        st.pyplot(fig)
            
            elif selected_table == "outcomes":
                # Outcome type distribution
                cursor.execute("SELECT outcome_type, COUNT(*) as count FROM outcomes WHERE outcome_type IS NOT NULL GROUP BY outcome_type ORDER BY count DESC")
                outcome_data = pd.DataFrame(cursor.fetchall())
                if not outcome_data.empty:
                    fig = px.bar(outcome_data, x='outcome_type', y='count', title="Outcome Type Distribution")
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Outcome Type Distribution")
            
            elif selected_table == "sponsors":
                # Top sponsors
                cursor.execute("SELECT agency, COUNT(*) as count FROM sponsors GROUP BY agency ORDER BY count DESC LIMIT 15")
                top_sponsors = pd.DataFrame(cursor.fetchall())
                if not top_sponsors.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(top_sponsors, x='agency', y='count', title="Top 15 Sponsors (Bar)")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Seaborn horizontal bar
                        setup_dark_theme()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        top_sponsors_sorted = top_sponsors.sort_values('count')
                        sns.barplot(data=top_sponsors_sorted, y='agency', x='count', palette="muted", ax=ax)
                        ax.set_title("Top 15 Sponsors (Horizontal)", color='white', fontsize=12, fontweight='bold')
                        ax.set_xlabel("Number of Trials", color='white')
                        ax.set_ylabel("Agency", color='white')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        st.pyplot(fig)
            
            elif selected_table == "locations":
                # Top countries
                cursor.execute("SELECT country, COUNT(*) as count FROM locations WHERE country IS NOT NULL GROUP BY country ORDER BY count DESC LIMIT 15")
                top_countries = pd.DataFrame(cursor.fetchall())
                if not top_countries.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(top_countries, x='country', y='count', title="Top 15 Countries (Bar)")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Seaborn bar with hue effect
                        setup_dark_theme()
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        top_countries_sorted = top_countries.sort_values('count')
                        colors = sns.color_palette("rocket", len(top_countries_sorted))
                        sns.barplot(data=top_countries_sorted, y='country', x='count', palette=colors, ax=ax)
                        ax.set_title("Top 15 Countries (Color Gradient)", color='white', fontsize=12, fontweight='bold')
                        ax.set_xlabel("Number of Locations", color='white')
                        ax.set_ylabel("Country", color='white')
                        ax.tick_params(colors='white')
                        for spine in ax.spines.values():
                            spine.set_color('white')
                        st.pyplot(fig)
            
            elif selected_table == "study_design":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Allocation distribution
                    cursor.execute("SELECT allocation, COUNT(*) as count FROM study_design WHERE allocation IS NOT NULL GROUP BY allocation ORDER BY count DESC")
                    allocation_data = pd.DataFrame(cursor.fetchall())
                    if not allocation_data.empty:
                        fig = px.bar(allocation_data, x='allocation', y='count', title="Allocation Type Distribution")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Primary purpose
                    cursor.execute("SELECT primary_purpose, COUNT(*) as count FROM study_design WHERE primary_purpose IS NOT NULL GROUP BY primary_purpose ORDER BY count DESC")
                    purpose_data = pd.DataFrame(cursor.fetchall())
                    if not purpose_data.empty:
                        fig = px.bar(purpose_data, x='primary_purpose', y='count', title="Primary Purpose Distribution")
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
            
            cursor.close()
            
        except Error as e:
            st.error(f"Error analyzing data: {e}")
        finally:
            conn.close()

elif page == "Time Trends":
    st.subheader("📈 Time Trends Analysis")
    
    conn = get_db_connection_with_retry(retries=3, delay=0.5)
    if not conn:
        st.error("Could not connect to the database. Please ensure MySQL is running and accessible.")
    else:
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Study Initiation Trends Over Time
            st.subheader("🚀 Study Initiation Trends")
            cursor.execute("""
                SELECT YEAR(start_date) as year, COUNT(*) as study_count 
                FROM studies 
                WHERE start_date IS NOT NULL 
                GROUP BY YEAR(start_date) 
                ORDER BY year
            """)
            yearly_data = pd.DataFrame(cursor.fetchall())
            
            if not yearly_data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(yearly_data, x='year', y='study_count', 
                                  title="Studies Started by Year", 
                                  markers=True)
                    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Studies")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(yearly_data, x='year', y='study_count', 
                                 title="Studies Started by Year (Bar)")
                    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Studies")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Phase Evolution Over Time
            st.subheader("🔬 Phase Distribution Evolution")
            cursor.execute("""
                SELECT YEAR(start_date) as year, phase, COUNT(*) as count 
                FROM studies 
                WHERE start_date IS NOT NULL AND phase IS NOT NULL 
                GROUP BY YEAR(start_date), phase 
                ORDER BY year, phase
            """)
            phase_yearly_data = pd.DataFrame(cursor.fetchall())
            
            if not phase_yearly_data.empty:
                fig = px.bar(phase_yearly_data, x='year', y='count', color='phase',
                            title="Phase Distribution Over Time",
                            barmode='stack')
                fig.update_layout(xaxis_title="Year", yaxis_title="Number of Studies")
                st.plotly_chart(fig, use_container_width=True)
            
            # Enrollment Trends
            st.subheader("👥 Enrollment Trends Over Time")
            cursor.execute("""
                SELECT YEAR(start_date) as year, 
                       AVG(enrollment) as avg_enrollment,
                       COUNT(*) as study_count
                FROM studies 
                WHERE start_date IS NOT NULL AND enrollment IS NOT NULL 
                GROUP BY YEAR(start_date) 
                ORDER BY year
            """)
            enrollment_trends = pd.DataFrame(cursor.fetchall())
            
            if not enrollment_trends.empty:
                col3, col4 = st.columns(2)
                with col3:
                    fig = px.line(enrollment_trends, x='year', y='avg_enrollment',
                                  title="Average Enrollment by Year",
                                  markers=True)
                    fig.update_layout(xaxis_title="Year", yaxis_title="Average Enrollment")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col4:
                    fig = px.scatter(enrollment_trends, x='year', y='avg_enrollment', 
                                    size='study_count',
                                    title="Enrollment vs Studies Count",
                                    hover_data=['study_count'])
                    fig.update_layout(xaxis_title="Year", yaxis_title="Average Enrollment")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Study Completion Trends
            st.subheader("🏁 Study Completion Trends")
            cursor.execute("""
                SELECT YEAR(completion_date) as year, COUNT(*) as completed_count 
                FROM studies 
                WHERE completion_date IS NOT NULL 
                GROUP BY YEAR(completion_date) 
                ORDER BY year
            """)
            completion_data = pd.DataFrame(cursor.fetchall())
            
            if not completion_data.empty:
                fig = px.area(completion_data, x='year', y='completed_count',
                             title="Studies Completed by Year")
                fig.update_layout(xaxis_title="Year", yaxis_title="Completed Studies")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top Conditions Trends
            st.subheader("🦠 Top Conditions Over Time")
            cursor.execute("""
                SELECT YEAR(s.start_date) as year, c.condition_name, COUNT(*) as count
                FROM studies s
                JOIN conditions c ON s.study_id = c.study_id
                WHERE s.start_date IS NOT NULL 
                GROUP BY YEAR(s.start_date), c.condition_name
                HAVING COUNT(*) >= 3
                ORDER BY year, count DESC
            """)
            conditions_trends = pd.DataFrame(cursor.fetchall())
            
            if not conditions_trends.empty:
                # Get top 10 conditions overall
                top_conditions = conditions_trends.groupby('condition_name')['count'].sum().nlargest(10).index
                filtered_conditions = conditions_trends[conditions_trends['condition_name'].isin(top_conditions)]
                
                if not filtered_conditions.empty:
                    fig = px.line(filtered_conditions, x='year', y='count', 
                                 color='condition_name',
                                 title="Top 10 Conditions Trends Over Time")
                    fig.update_layout(xaxis_title="Year", yaxis_title="Number of Studies")
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in time trends analysis: {e}")
        finally:
            conn.close()

elif page == "Enrollment Success Analytics":
    st.subheader("📊 Enrollment Success Analytics")
    st.markdown("---")
    
    # Load enrollment success data
    with st.spinner("Loading enrollment success data..."):
        df = load_enrollment_success_data()
    
    if df.empty:
        st.error("No enrollment data available. Please ensure the database is populated with studies that have enrollment information.")
        st.info("💡 Tip: Studies with zero enrollment or missing enrollment data are excluded from this analysis.")
    else:
        # Sidebar filters for enrollment success
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Enrollment Filters")
        
        # Status filter
        status_options = ['All'] + sorted(df['status'].unique().tolist())
        selected_status = st.sidebar.selectbox("Study Status", status_options, key="enroll_status")
        
        # Phase filter
        phase_options = ['All'] + sorted([p for p in df['phase'].unique() if pd.notna(p)])
        if phase_options:
            selected_phase = st.sidebar.selectbox("Study Phase", phase_options, key="enroll_phase")
        else:
            selected_phase = 'All'
        
        # Enrollment range
        if not df['enrollment'].empty:
            min_enrollment = int(df['enrollment'].min())
            max_enrollment = int(df['enrollment'].max())
            enrollment_range = st.sidebar.slider(
                "Enrollment Range", 
                min_value=min_enrollment,
                max_value=min(max_enrollment, 10000),  # Cap at 10k for slider performance
                value=(min_enrollment, min(max_enrollment, 10000)),
                key="enroll_range"
            )
        else:
            enrollment_range = (0, 1000)
        
        # Study type filter
        study_types = ['All'] + sorted([t for t in df['study_type'].unique() if pd.notna(t)])
        selected_study_type = st.sidebar.selectbox("Study Type", study_types, key="enroll_type")
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_status != 'All':
            filtered_df = filtered_df[filtered_df['status'] == selected_status]
        
        if selected_phase != 'All':
            filtered_df = filtered_df[filtered_df['phase'] == selected_phase]
        
        if selected_study_type != 'All':
            filtered_df = filtered_df[filtered_df['study_type'] == selected_study_type]
        
        filtered_df = filtered_df[
            (filtered_df['enrollment'] >= enrollment_range[0]) &
            (filtered_df['enrollment'] <= enrollment_range[1])
        ]
        
        if filtered_df.empty:
            st.warning("No studies match the selected filters. Try adjusting your filter criteria.")
        else:
            # Key metrics section
            st.subheader("📈 Key Enrollment Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Studies", f"{len(filtered_df):,}")
                
            with col2:
                avg_enrollment = filtered_df['enrollment'].mean()
                st.metric("Avg. Enrollment", f"{avg_enrollment:,.0f}")
                
            with col3:
                avg_success_score = filtered_df['composite_score'].mean()
                st.metric("Avg. Success Score", f"{avg_success_score:.1f}/100")
                
            with col4:
                completed_pct = (filtered_df['status'] == 'COMPLETED').mean() * 100
                st.metric("Completed Studies", f"{completed_pct:.1f}%")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Success Overview", 
                "🎯 Performance Analysis", 
                "🏆 Top Performers", 
                "📋 Detailed Data",
                "📖 Methodology"
            ])
            
            with tab1:
                st.subheader("Success Tier Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Success tier pie chart
                    success_counts = filtered_df['success_tier'].value_counts()
                    
                    if not success_counts.empty:
                        colors = {
                            'Exceeded (≥100%)': '#00ff00',
                            'Met (75-100%)': '#90EE90',
                            'Below (50-75%)': '#FFD700',
                            'Significantly Below (<50%)': '#FF6B6B',
                            'Unknown': '#808080'
                        }
                        
                        fig = px.pie(
                            values=success_counts.values,
                            names=success_counts.index,
                            title="Success Tier Distribution",
                            color=success_counts.index,
                            color_discrete_map=colors,
                            height=400
                        )
                        
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Enrollment rate distribution
                    rate_counts = filtered_df['enrollment_rate_tier'].value_counts()
                    
                    if not rate_counts.empty:
                        rate_colors = {
                            'Excellent (>50/month)': '#00ff00',
                            'Good (10-50/month)': '#90EE90',
                            'Adequate (1-10/month)': '#FFD700',
                            'Slow (<1/month)': '#FF6B6B',
                            'Unknown': '#808080'
                        }
                        
                        fig = px.pie(
                            values=rate_counts.values,
                            names=rate_counts.index,
                            title="Enrollment Rate Distribution",
                            color=rate_counts.index,
                            color_discrete_map=rate_colors,
                            height=400
                        )
                        
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Composite score distribution
                st.subheader("Composite Success Score Distribution")
                
                if not filtered_df['composite_score'].empty:
                    fig = px.histogram(
                        filtered_df, 
                        x='composite_score',
                        nbins=30,
                        title="Distribution of Composite Success Scores",
                        labels={'composite_score': 'Composite Success Score (0-100)', 'count': 'Number of Studies'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    
                    # Add vertical lines for score ranges
                    fig.add_vline(x=80, line_dash="dash", line_color="green", 
                                  annotation_text="Highly Successful (80+)")
                    fig.add_vline(x=60, line_dash="dash", line_color="orange", 
                                  annotation_text="Successful (60+)")
                    fig.add_vline(x=40, line_dash="dash", line_color="red", 
                                  annotation_text="Moderate (40+)")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Performance Analysis")
                
                # Enrollment vs Duration scatter plot
                if len(filtered_df) > 0:
                    fig = px.scatter(
                        filtered_df,
                        x='duration_months',
                        y='enrollment',
                        color='success_tier',
                        size='composite_score',
                        hover_data=['nct_id', 'title', 'status', 'enrollment_rate'],
                        title="Enrollment vs Study Duration",
                        labels={
                            'duration_months': 'Study Duration (months)',
                            'enrollment': 'Total Enrollment',
                            'success_tier': 'Success Tier'
                        },
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Success metrics by status
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box plot of composite scores by status
                        fig = px.box(
                            filtered_df,
                            x='status',
                            y='composite_score',
                            title="Success Scores by Study Status",
                            labels={'composite_score': 'Composite Success Score'}
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Average enrollment by phase
                        if 'phase' in filtered_df.columns and filtered_df['phase'].notna().any():
                            phase_avg = filtered_df.groupby('phase')['enrollment'].mean().sort_values(ascending=False)
                            
                            if not phase_avg.empty:
                                fig = px.bar(
                                    x=phase_avg.index,
                                    y=phase_avg.values,
                                    title="Average Enrollment by Phase",
                                    labels={'x': 'Study Phase', 'y': 'Average Enrollment'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("🏆 Top Performing Studies")
                
                # Sort by composite score
                if len(filtered_df) > 0:
                    top_studies = filtered_df.nlargest(min(15, len(filtered_df)), 'composite_score')
                    
                    # Display top studies in detail
                    st.write("**Top Studies by Composite Success Score:**")
                    
                    for i, (_, study) in enumerate(top_studies.head(10).iterrows(), 1):
                        with st.expander(f"#{i} - {study['nct_id']} (Score: {study['composite_score']:.1f}/100)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                title_text = study['title'] if pd.notna(study['title']) else "No title available"
                                st.write(f"**Title:** {title_text[:100]}...")
                                st.write(f"**Status:** {study['status']}")
                                st.write(f"**Phase:** {study['phase'] if pd.notna(study['phase']) else 'N/A'}")
                                st.write(f"**Enrollment:** {study['enrollment']:,}")
                                
                            with col2:
                                st.write(f"**Enrollment Rate:** {study['enrollment_rate']:.1f}/month")
                                st.write(f"**Success vs Benchmark:** {study['success_percentage']:.1f}%")
                                st.write(f"**Success Tier:** {study['success_tier']}")
                                st.write(f"**Percentile Rank:** {study['success_percentile']:.1f}%")
                    
                    # Top performers visualization
                    if len(top_studies) > 0:
                        fig = px.bar(
                            top_studies.head(15),
                            x='composite_score',
                            y='nct_id',
                            orientation='h',
                            title=f"Top {min(15, len(top_studies))} Studies by Composite Success Score",
                            labels={'composite_score': 'Composite Success Score', 'nct_id': 'NCT ID'},
                            color='composite_score',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("📋 Detailed Study Data")
                
                # Select columns to display
                display_columns = [
                    'nct_id', 'title', 'status', 'phase', 'enrollment', 
                    'enrollment_rate', 'success_percentage', 'success_tier',
                    'composite_score', 'success_percentile'
                ]
                
                # Check which columns exist in the dataframe
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                display_df = filtered_df[available_columns].copy()
                
                # Format numeric columns
                if 'enrollment_rate' in display_df.columns:
                    display_df['enrollment_rate'] = display_df['enrollment_rate'].round(2)
                if 'success_percentage' in display_df.columns:
                    display_df['success_percentage'] = display_df['success_percentage'].round(1)
                if 'composite_score' in display_df.columns:
                    display_df['composite_score'] = display_df['composite_score'].round(1)
                if 'success_percentile' in display_df.columns:
                    display_df['success_percentile'] = display_df['success_percentile'].round(1)
                
                # Sort by composite score by default
                if 'composite_score' in display_df.columns:
                    display_df = display_df.sort_values('composite_score', ascending=False)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"enrollment_success_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'enrollment' in filtered_df.columns:
                        st.metric("Median Enrollment", f"{filtered_df['enrollment'].median():,.0f}")
                
                with col2:
                    if 'enrollment_rate' in filtered_df.columns:
                        st.metric("Median Rate (/month)", f"{filtered_df['enrollment_rate'].median():.1f}")
                
                with col3:
                    if 'composite_score' in filtered_df.columns:
                        st.metric("Median Success Score", f"{filtered_df['composite_score'].median():.1f}")
                
                with col4:
                    highly_successful = (filtered_df['composite_score'] >= 80).sum() if 'composite_score' in filtered_df.columns else 0
                    st.metric("Highly Successful Studies", highly_successful)
            
            with tab5:
                st.subheader("📖 Enrollment Success Methodology & KPI Definitions")
                
                st.markdown("---")
                st.markdown("### Overview")
                st.markdown("""
                This dashboard calculates enrollment success using multiple complementary approaches to provide
                a comprehensive view of clinical trial performance. The methodology is based on industry standards
                and best practices for clinical trial analytics.
                """)
                
                st.markdown("---")
                st.markdown("### **METRIC 1: ENROLLMENT RATE** (Participants/Month)")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Formula:** `Total Enrollment ÷ Study Duration (months)`
                    
                    **Purpose:** Measures enrollment velocity - how quickly participants are recruited
                    
                    **Calculation Details:**
                    - Duration calculated from start_date to completion_date
                    - Minimum duration set to 1 month to avoid division errors
                    - Accounts for variable month lengths (30.44 days average)
                    """)
                
                with col2:
                    st.markdown("""
                    **Tier Classification:**
                    - 🟢 **Excellent:** >50 participants/month (very rapid)
                    - 🟡 **Good:** 10-50 participants/month (typical academic)
                    - 🟠 **Adequate:** 1-10 participants/month (slow but viable)  
                    - 🔴 **Slow:** <1 participant/month (concerning)
                    
                    **Use Case:** Identify recruitment bottlenecks and compare enrollment pace
                    """)
                
                st.markdown("---")
                st.markdown("### **METRIC 2: SUCCESS PERCENTAGE** (vs. Industry Benchmark)")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Formula:** `(Actual Enrollment ÷ Expected Enrollment) × 100%`
                    
                    **Benchmark:** 10 participants/month (industry standard)
                    
                    **Calculation:**
                    - Expected = Duration (months) × 10 participants/month
                    - Capped at 200% to prevent extreme outliers
                    - Based on typical clinical trial performance data
                    """)
                
                with col2:
                    st.markdown("""
                    **Success Tiers:**
                    - 🟢 **Exceeded:** ≥100% (outperformed expectations)
                    - 🟡 **Met:** 75-100% (met expectations)
                    - 🟠 **Below:** 50-75% (underperformed)
                    - 🔴 **Significantly Below:** <50% (major shortfall)
                    
                    **Use Case:** Compare performance against industry standards
                    """)
                
                st.markdown("---")
                st.markdown("### **METRIC 3: STATUS SUCCESS SCORE** (0-100 Points)")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Purpose:** Assigns success score based on study completion status
                    
                    **Status Weights:**
                    - **COMPLETED:** 100 points (study finished successfully)
                    - **ACTIVE_NOT_RECRUITING:** 85 points (enrollment closed, ongoing)
                    - **RECRUITING:** 60 points (still enrolling)
                    - **ENROLLING_BY_INVITATION:** 55 points (selective recruitment)
                    - **NOT_YET_RECRUITING:** 30 points (not started)
                    """)
                
                with col2:
                    st.markdown("""
                    **Continued Status Weights:**
                    - **SUSPENDED:** 20 points (temporarily halted)
                    - **TERMINATED:** 10 points (stopped early)
                    - **WITHDRAWN:** 5 points (never started/cancelled)
                    - **UNKNOWN:** 50 points (default for missing status)
                    
                    **Use Case:** Assess overall study viability and completion likelihood
                    """)
                
                st.markdown("---")
                st.markdown("### **METRIC 4: COMPOSITE SUCCESS SCORE** ⭐ (0-100 Points) - **RECOMMENDED**")
                
                st.markdown("""
                This is our primary recommendation metric that combines multiple factors for a holistic assessment:
                """)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Component Weights:**
                    
                    **1. Enrollment Completeness (40 points max)**
                    - Formula: `min(40, (enrollment ÷ 500) × 40)`
                    - Benchmark: 500+ participants = maximum points
                    - Measures: Absolute recruitment success
                    
                    **2. Status Success (30 points max)**
                    - Formula: `(Status Score ÷ 100) × 30`
                    - Uses: Status success score from Metric 3
                    - Measures: Study completion success
                    """)
                
                with col2:
                    st.markdown("""
                    **3. Temporal Efficiency (20 points max)**
                    - Formula: `min(20, (enrollment_rate ÷ 50) × 20)`
                    - Benchmark: 50+ participants/month = maximum points
                    - Measures: Speed of recruitment
                    
                    **4. Data Completeness (10 points max)**
                    - Components: Summary, conditions, interventions, locations
                    - Scoring: 2-3 points per complete data category
                    - Measures: Study documentation quality
                    """)
                
                st.markdown("**Composite Score Interpretation:**")
                st.markdown("""
                - 🟢 **80-100:** Highly successful (top tier performance)
                - 🟡 **60-79:** Successful (above average performance)
                - 🟠 **40-59:** Moderate success (average performance)
                - 🟠 **20-39:** Below average (needs improvement)
                - 🔴 **0-19:** Struggled significantly (intervention needed)
                """)
                
                st.markdown("---")
                st.markdown("### **PERCENTILE RANKING**")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    **Definition:** Comparative ranking against all studies in database
                    
                    **Calculation:** `composite_score.rank(pct=True) × 100`
                    
                    **Interpretation:**
                    - 95th percentile = Better than 95% of studies
                    - 50th percentile = Median performance
                    - 5th percentile = Worse than 95% of studies
                    """)
                
                with col2:
                    st.markdown("""
                    **Use Cases:**
                    - Benchmark against peer studies
                    - Identify top and bottom performers  
                    - Set realistic performance targets
                    - Portfolio performance assessment
                    
                    **Note:** Rankings are relative to the current database
                    """)
                
                st.markdown("---")
                st.markdown("### **Data Quality & Limitations**")
                
                st.markdown("""
                **Included Studies:** Only studies with enrollment > 0 are analyzed
                
                **Excluded Data:** 
                - Studies with missing enrollment information
                - Studies with enrollment = 0
                - Studies with invalid date ranges
                
                **Assumptions:**
                - Industry benchmark: 10 participants/month average
                - Optimal study size: 500 participants for maximum score
                - Month length: 30.44 days (accounting for leap years)
                
                **Limitations:**
                - Phase-specific benchmarks not applied
                - Therapeutic area variations not considered  
                - Patient population complexity not factored
                - Geographic enrollment differences not weighted
                """)
                
                st.markdown("---")
                st.info("💡 **Tip:** Use the Composite Success Score as your primary metric, supplemented by individual components for detailed analysis.")

    st.subheader("Search Clinical Studies")
    search_query = st.text_input("Enter a search term:")
    
    if search_query:
        conn = get_db_connection_with_retry(retries=3, delay=0.5)
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
        
        # # Also delete the enrollment success CSV file
        # try:
        #     csv_path = Path(__file__).parent / 'results' / 'enrollment_success_metrics.csv'
        #     if csv_path.exists():
        #         csv_path.unlink()
        #         st.info("Enrollment success CSV file deleted")
        # except Exception as csv_error:
        #     st.warning(f"Could not delete CSV file: {csv_error}")
        
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
