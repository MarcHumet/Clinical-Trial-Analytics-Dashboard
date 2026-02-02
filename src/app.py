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
page = st.sidebar.radio("Select a page:", ["Home", "Data Overview & Completeness", "Distribution Analysis", "Search Studies"])

if page == "Home":
    st.subheader("Welcome to the Clinical Trial Analytics Dashboard")
    st.write("""
    This dashboard provides:
    - **Data Overview & Completeness*: View comprehensive statistics about clinical trials
    - **Analysis**: Perform detailed analysis on trial data
    - **Search Studies**: Find and explore specific clinical trials
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

elif page == "Search Studies":
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
