# Streamlit Dashboard - Weak Points Analysis & Improvements

**Date:** February 17, 2026  
**Dashboard:** Clinical Trial Analytics Dashboard  
**File:** `/src/app.py` (1597 lines)

---

## 📋 Executive Summary

The Clinical Trial Analytics Dashboard is functionally robust but has several areas that need improvement in **performance**, **error handling**, **security**, **code organization**, and **user experience**. This document outlines critical weak points with actionable recommendations.

---

## 🔴 CRITICAL ISSUES

### 1. **Database Connection Pattern - Anti-Pattern**

**Problem:**
- Two different database connection functions: `get_db_connection()` and `get_db_connection_with_retry()`
- `mysql.connector` and `sqlalchemy` used inconsistently throughout the code
- No connection pooling or session management
- Connections are created and closed repeatedly without reuse

**Code Location:** Lines 250-280

**Impact:**
- Performance degradation with high traffic
- Risk of connection pool exhaustion
- Inconsistent error handling

**Recommendation:**
```python
# Create a single connection manager with proper pooling
from sqlalchemy.pool import QueuePool

@st.cache_resource
def get_engine():
    """Single source of truth for database connections"""
    mysql_url = get_mysql_url()  # Centralize URL building
    return create_engine(
        mysql_url,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600     # Recycle connections after 1 hour
    )

def get_session():
    """Get a session from the engine pool"""
    engine = get_engine()
    return Session(engine)
```

---

### 2. **SQL Injection Vulnerability**

**Problem:**
- Line 1528: Direct string interpolation in search query
```python
query = f"""
SELECT * FROM studies 
WHERE title LIKE %s OR description LIKE %s 
LIMIT 50
"""
```
- Line 501: `DESCRIBE {selected_table}` uses f-string without validation
- Multiple locations using f-strings for table names without validation

**Impact:**
- Security vulnerability allowing SQL injection attacks
- Potential data breach or database corruption

**Recommendation:**
```python
# Whitelist allowed tables
ALLOWED_TABLES = {
    'studies', 'conditions', 'interventions', 
    'outcomes', 'sponsors', 'locations', 'study_design'
}

def validate_table_name(table_name):
    """Validate table name against whitelist"""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")
    return table_name

# Use parameterized queries exclusively
def safe_search(search_query, conn):
    query = """
        SELECT * FROM studies 
        WHERE title LIKE %s OR brief_summary LIKE %s 
        LIMIT 50
    """
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute(query, (f"%{search_query}%", f"%{search_query}%"))
        return cursor.fetchall()
```

---

### 3. **Blocking Operations in Main Thread**

**Problem:**
- Line 411-443: `run_seed_in_app()` runs API fetches synchronously in main thread
- Can freeze UI for 3+ minutes during data download
- No cancellation mechanism for long-running operations

**Impact:**
- Poor user experience during data loading
- Browser timeouts possible
- No way to cancel once started

**Recommendation:**
```python
import asyncio
import aiohttp

async def fetch_page_async(session, api_url, params):
    """Async fetch of single page"""
    async with session.get(api_url, params=params, timeout=30) as response:
        return await response.json()

async def run_seed_async(max_pages: int, page_size: int):
    """Async data fetching using aiohttp"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        # ... create tasks for parallel fetching
        results = await asyncio.gather(*tasks)
        return results

# Or use Streamlit's background execution
if st.button("Fill MySQL DB"):
    with st.spinner("Starting download..."):
        # Run in background and poll for updates
        job_id = start_background_job()
        while not is_job_complete(job_id):
            progress = get_job_progress(job_id)
            st.progress(progress)
            time.sleep(1)
```

---

### 4. **Missing Error Recovery & Graceful Degradation**

**Problem:**
- Database failure fallback to CSV (line 137) but no mechanism to sync back
- No retry logic for transient failures
- Full page errors instead of component-level error handling

**Current Code (Line 137):**
```python
except Exception as e:
    st.warning(f"Database connection failed: {str(e)}")
    st.info("Attempting to load from CSV file...")
```

**Impact:**
- Poor user experience when DB is temporarily unavailable
- No automatic recovery when DB comes back online
- Error cascades affect entire dashboard

**Recommendation:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def load_enrollment_with_retry():
    """Load with exponential backoff retry"""
    try:
        return load_from_database()
    except DatabaseError as e:
        logger.warning(f"DB attempt failed: {e}")
        raise

def load_enrollment_success_data():
    """Load with graceful fallback"""
    try:
        return load_enrollment_with_retry()
    except Exception as e:
        logger.error(f"All DB attempts failed: {e}")
        df = load_from_csv_fallback()
        st.warning("⚠️ Using cached data. Database unavailable.")
        return df

# Add component-level error boundaries
try:
    render_enrollment_charts(df)
except Exception as e:
    st.error("Unable to render enrollment charts")
    logger.exception(e)
    # Rest of dashboard continues to work
```

---

## 🟠 MAJOR ISSUES

### 5. **Memory Management - Large Data Loading**

**Problem:**
- Line 69-140: `load_enrollment_success_data()` loads entire dataset into memory
- No pagination for large result sets
- Complex GROUP BY queries without LIMIT clauses
- Multiple large DataFrames created simultaneously

**Impact:**
- Memory exhaustion with large datasets (>100k studies)
- Slow page loads
- Potential crashes

**Recommendation:**
```python
# Add pagination
@st.cache_data(ttl=300)
def load_enrollment_success_data(page=1, page_size=1000):
    """Load data with pagination"""
    offset = (page - 1) * page_size
    query = """
        SELECT ... 
        FROM studies s
        ...
        GROUP BY s.study_id
        HAVING s.enrollment IS NOT NULL AND s.enrollment > 0
        LIMIT %s OFFSET %s
    """
    df = pd.read_sql_query(text(query), conn, params=(page_size, offset))
    return df

# Add data streaming for large exports
def stream_large_export(query, chunk_size=10000):
    """Stream data in chunks for export"""
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        yield chunk.to_csv(index=False, header=(chunk_size==0))

# Lazy loading for tabs
if tab == "Detailed Data":
    if st.button("Load Detailed Data"):
        df_detailed = load_detailed_data()
        st.dataframe(df_detailed)
```

---

### 6. **Inefficient Cache Strategy**

**Problem:**
- Line 69: `@st.cache_data(ttl=300)` - 5 minute TTL might be too short or too long
- No cache key differentiation by filters
- Entire dataset cached even when only subset needed
- `@st.cache_resource` used for database connections but no cleanup on error

**Impact:**
- Repeated expensive database queries
- Stale data shown to users
- Memory bloat from cached data

**Recommendation:**
```python
# Add cache keys based on filters
@st.cache_data(ttl=600, show_spinner="Loading enrollment data...")
def load_enrollment_success_data(
    status_filter=None,
    phase_filter=None,
    enrollment_range=None
):
    """Cache different filter combinations separately"""
    query = build_filtered_query(status_filter, phase_filter, enrollment_range)
    return execute_query(query)

# Add cache invalidation
def invalidate_enrollment_cache():
    """Clear cache when data is refreshed"""
    load_enrollment_success_data.clear()
    st.cache_data.clear()

# Add cache size limits
@st.cache_data(ttl=300, max_entries=10)
def load_data_with_limit(...):
    """Limit cache to 10 most recent queries"""
    pass
```

---

### 7. **No Input Validation**

**Problem:**
- Line 1520: Search query has no sanitization or length limits
- Date inputs not validated for reasonable ranges
- Enum values (status, phase) not validated against known values
- Numeric inputs can crash with invalid values

**Impact:**
- Application crashes from malformed input
- Poor user experience
- Security vulnerabilities

**Recommendation:**
```python
from pydantic import BaseModel, validator, Field

class StudyFilters(BaseModel):
    """Validated filter model"""
    status: str = Field(..., regex=r'^[A-Z_]+$')
    phase: Optional[str] = Field(None, regex=r'^PHASE [1-4]$|^N/A$')
    enrollment_min: int = Field(0, ge=0, le=1000000)
    enrollment_max: int = Field(10000, ge=0, le=1000000)
    search_query: str = Field("", max_length=500)
    
    @validator('enrollment_max')
    def validate_enrollment_range(cls, v, values):
        if 'enrollment_min' in values and v < values['enrollment_min']:
            raise ValueError('Max enrollment must be >= min enrollment')
        return v

# Use in app
try:
    filters = StudyFilters(
        status=selected_status,
        phase=selected_phase,
        enrollment_min=enrollment_range[0],
        enrollment_max=enrollment_range[1]
    )
    filtered_df = apply_filters(df, filters)
except ValidationError as e:
    st.error(f"Invalid filter values: {e}")
```

---

### 8. **Code Duplication & Organization**

**Problem:**
- Database connection code duplicated (mysql.connector + sqlalchemy)
- Similar chart rendering code repeated across pages
- Large monolithic file (1597 lines)
- Mixed concerns (UI, business logic, data access)

**Impact:**
- Hard to maintain and test
- Bugs propagate across similar code sections
- Difficult to add new features

**Recommendation:**

**File Structure:**
```
src/
├── app.py                      # Main entry point (200 lines max)
├── components/
│   ├── __init__.py
│   ├── charts.py              # Reusable chart components
│   ├── filters.py             # Filter UI components
│   └── metrics.py             # Metric display components
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── data_overview.py
│   ├── distribution_analysis.py
│   ├── time_trends.py
│   └── enrollment_success.py
├── services/
│   ├── __init__.py
│   ├── database.py            # Database access layer
│   ├── api_service.py         # ClinicalTrials.gov API
│   └── cache_service.py       # Cache management
├── models/
│   ├── __init__.py
│   ├── study.py               # Data models
│   └── filters.py             # Filter models
└── utils/
    ├── __init__.py
    ├── validators.py          # Input validation
    ├── formatters.py          # Data formatting
    └── theme.py               # UI theme configuration
```

**Example Refactoring:**

```python
# src/services/database.py
class DatabaseService:
    """Centralized database access"""
    
    def __init__(self):
        self.engine = self._create_engine()
    
    def get_studies(self, filters=None):
        """Get studies with optional filters"""
        query = self._build_study_query(filters)
        return pd.read_sql_query(query, self.engine)
    
    def get_enrollment_metrics(self):
        """Get enrollment success data"""
        # Implementation here
        pass

# src/components/charts.py
class ChartComponents:
    """Reusable chart components"""
    
    @staticmethod
    def render_pie_chart(data, title, color_map=None):
        """Render a standardized pie chart"""
        fig = px.pie(
            values=data.values,
            names=data.index,
            title=title,
            color=data.index if color_map else None,
            color_discrete_map=color_map
        )
        fig.update_traces(textinfo='percent+label')
        return fig
    
    @staticmethod
    def render_bar_chart(df, x, y, title):
        """Render a standardized bar chart"""
        # Implementation here
        pass

# src/pages/enrollment_success.py
def render_enrollment_success_page():
    """Enrollment success analytics page"""
    st.subheader("📊 Enrollment Success Analytics")
    
    # Load data
    db_service = DatabaseService()
    df = db_service.get_enrollment_metrics()
    
    # Render filters
    filters = FilterComponents.render_enrollment_filters()
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Render metrics and charts
    MetricComponents.render_key_metrics(filtered_df)
    ChartComponents.render_enrollment_charts(filtered_df)
```

---

## 🟡 MODERATE ISSUES

### 9. **Theme Configuration Scattered**

**Problem:**
- Line 49-64: `setup_dark_theme()` function modifies global matplotlib settings
- Called multiple times throughout code
- No centralized theme configuration

**Recommendation:**
```python
# src/utils/theme.py
class DashboardTheme:
    """Centralized theme configuration"""
    
    COLORS = {
        'background': '#000000',
        'text': '#FFFFFF',
        'primary': '#1f77b4',
        'success': '#00ff00',
        'warning': '#FFD700',
        'danger': '#FF6B6B',
        'info': '#808080'
    }
    
    @classmethod
    def setup_matplotlib(cls):
        """Configure matplotlib once"""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': cls.COLORS['background'],
            'axes.facecolor': cls.COLORS['background'],
            'text.color': cls.COLORS['text'],
            # ... rest of config
        })
    
    @classmethod
    def get_success_colors(cls):
        """Get standard success tier colors"""
        return {
            'Exceeded': cls.COLORS['success'],
            'Met': '#90EE90',
            'Below': cls.COLORS['warning'],
            'Significantly Below': cls.COLORS['danger']
        }

# Call once at app startup
DashboardTheme.setup_matplotlib()
```

---

### 10. **Hardcoded Values & Magic Numbers**

**Problem:**
- Line 28: `seed_count = int(os.getenv('page_size_limit') or 10)`
- Line 213: Hardcoded status scores
- Line 1225: Enrollment benchmark of 10/month hardcoded
- Line 1106: Color codes scattered throughout

**Recommendation:**
```python
# src/config.py
from dataclasses import dataclass

@dataclass
class DashboardConfig:
    """Centralized configuration"""
    
    # API Settings
    API_URL: str = "https://clinicaltrials.gov/api/v2/studies"
    API_PAGE_SIZE: int = 100
    API_MAX_PAGES: int = 100
    API_TIMEOUT: int = 30
    
    # Database Settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 3600
    
    # Cache Settings
    CACHE_TTL: int = 300
    CACHE_MAX_ENTRIES: int = 10
    
    # Enrollment Metrics
    ENROLLMENT_BENCHMARK_MONTHLY: int = 10
    ENROLLMENT_EXCELLENT_THRESHOLD: int = 50
    ENROLLMENT_GOOD_THRESHOLD: int = 10
    ENROLLMENT_ADEQUATE_THRESHOLD: int = 1
    
    # Status Scores
    STATUS_SCORES: dict = {
        'COMPLETED': 100,
        'ACTIVE_NOT_RECRUITING': 85,
        'RECRUITING': 60,
        # ... rest
    }
    
    # Composite Score Weights
    COMPOSITE_WEIGHT_ENROLLMENT: float = 0.40
    COMPOSITE_WEIGHT_STATUS: float = 0.30
    COMPOSITE_WEIGHT_TEMPORAL: float = 0.20
    COMPOSITE_WEIGHT_DATA: float = 0.10

# Usage
config = DashboardConfig()
if enrollment_rate > config.ENROLLMENT_EXCELLENT_THRESHOLD:
    tier = "Excellent"
```

---

### 11. **Incomplete Error Messages**

**Problem:**
- Line 137: `st.warning(f"Database connection failed: {str(e)}")` - full exception exposed
- No user-friendly error messages
- No logging infrastructure
- Technical details shown to end users

**Recommendation:**
```python
import logging
from loguru import logger

# Setup structured logging
logger.add(
    "logs/dashboard_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time} {level} {message}"
)

class UserFriendlyError:
    """Convert technical errors to user-friendly messages"""
    
    ERROR_MESSAGES = {
        'ConnectionError': "Unable to connect to database. Please try again later.",
        'TimeoutError': "Request timed out. The system is busy, please try again.",
        'ValidationError': "Invalid input provided. Please check your entries.",
        'DataError': "Unable to load data. Please refresh the page."
    }
    
    @classmethod
    def handle_error(cls, error, context=""):
        """Log technical details, show friendly message"""
        error_type = type(error).__name__
        
        # Log for developers
        logger.error(f"Error in {context}: {error_type} - {str(error)}", 
                    exc_info=True)
        
        # Show to users
        user_msg = cls.ERROR_MESSAGES.get(
            error_type, 
            "An unexpected error occurred. Please try again."
        )
        st.error(user_msg)
        
        # Show details in expander for power users
        with st.expander("Technical Details"):
            st.code(f"{error_type}: {str(error)}")

# Usage
try:
    df = load_enrollment_success_data()
except Exception as e:
    UserFriendlyError.handle_error(e, context="load_enrollment_data")
    df = pd.DataFrame()
```

---

### 12. **No Rate Limiting for API Calls**

**Problem:**
- Line 411-443: No rate limiting for ClinicalTrials.gov API
- Could get IP banned
- No exponential backoff on failures
- `time.sleep(0.1)` is too aggressive

**Recommendation:**
```python
from ratelimit import limits, sleep_and_retry
import requests

class ClinicalTrialsAPIClient:
    """Rate-limited API client"""
    
    # ClinicalTrials.gov recommends max 1 request per second
    CALLS_PER_SECOND = 1
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Clinical-Trial-Dashboard/1.0'
        })
    
    @sleep_and_retry
    @limits(calls=CALLS_PER_SECOND, period=1)
    def fetch_page(self, page_token=None, page_size=100):
        """Fetch single page with rate limiting"""
        params = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token
        
        try:
            response = self.session.get(
                self.API_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def fetch_with_retry(self, page_token=None):
        """Fetch with exponential backoff"""
        return self.fetch_page(page_token)
```

---

## 🟢 MINOR ISSUES

### 13. **Performance - Redundant Database Queries**

**Problem:**
- Multiple calls to check DB state: `is_db_empty()` called multiple times
- Separate queries for counts that could be combined
- No query result caching at database level

**Recommendation:**
```python
@st.cache_data(ttl=60)
def get_database_stats():
    """Get all database stats in single query"""
    stats = {}
    with get_session() as session:
        # Single query for all table counts
        for table in ALLOWED_TABLES:
            result = session.execute(
                text(f"SELECT COUNT(*) as count FROM {table}")
            ).fetchone()
            stats[table] = result.count
    return stats

# Usage
stats = get_database_stats()
if stats['studies'] == 0:
    st.warning("Database is empty")
```

---

### 14. **Accessibility Issues**

**Problem:**
- No alt text for charts
- Color-only indicators (no icons or text alternatives)
- No keyboard navigation hints
- Color scheme not colorblind-friendly

**Recommendation:**
```python
# Add accessibility features
def render_accessible_chart(fig, description):
    """Render chart with accessibility features"""
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📊 Chart Description (Screen Reader)"):
        st.write(description)

# Use colorblind-friendly palettes
COLORBLIND_SAFE_PALETTE = [
    '#0173B2',  # Blue
    '#DE8F05',  # Orange
    '#029E73',  # Green
    '#CC78BC',  # Purple
    '#CA9161',  # Brown
    '#FBAFE4',  # Pink
]

# Add status indicators with icons
def status_indicator(value, threshold):
    """Visual + text + icon indicator"""
    if value >= threshold:
        return "✅ Excellent", "green"
    elif value >= threshold * 0.75:
        return "⚠️ Good", "orange"
    else:
        return "❌ Needs Improvement", "red"

icon, color = status_indicator(score, 80)
st.markdown(f"**Status:** :{color}[{icon}]")
```

---

### 15. **Missing Monitoring & Analytics**

**Problem:**
- No application performance monitoring
- No user interaction tracking
- No error alerting
- No dashboard usage metrics

**Recommendation:**
```python
# Add application monitoring
from opentelemetry import trace
from streamlit_analytics import track_pageview, track_event

tracer = trace.get_tracer(__name__)

def track_dashboard_usage():
    """Track dashboard usage metrics"""
    track_pageview()
    
    # Track filter usage
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = 0
    
    st.session_state.filters_applied += 1
    track_event('filter_applied', {
        'count': st.session_state.filters_applied
    })

# Add performance monitoring
with tracer.start_as_current_span("load_enrollment_data"):
    start_time = time.time()
    df = load_enrollment_success_data()
    duration = time.time() - start_time
    
    logger.info(f"Data loaded in {duration:.2f}s, {len(df)} records")
    
    if duration > 5.0:
        logger.warning(f"Slow query detected: {duration:.2f}s")

# Add error alerting
from sentry_sdk import capture_exception

try:
    process_data()
except Exception as e:
    capture_exception(e)  # Send to Sentry
    logger.exception("Critical error in data processing")
    st.error("An error occurred. Our team has been notified.")
```

---

## 📊 PRIORITY RECOMMENDATIONS SUMMARY

| Priority | Issue | Effort | Impact | 
|----------|-------|--------|--------|
| 🔴 P0 | SQL Injection Vulnerability | Medium | Critical |
| 🔴 P0 | Database Connection Pattern | Medium | High |
| 🔴 P0 | Input Validation | Medium | High |
| 🟠 P1 | Blocking Operations | High | High |
| 🟠 P1 | Error Recovery | Medium | Medium |
| 🟠 P1 | Memory Management | High | High |
| 🟠 P2 | Code Organization | High | Medium |
| 🟠 P2 | Cache Strategy | Low | Medium |
| 🟡 P3 | Rate Limiting | Low | Low |
| 🟡 P3 | Monitoring | Medium | Low |

---

## 🚀 QUICK WINS (Implement First)

1. **Add input validation** (2-4 hours)
   - Use Pydantic models
   - Whitelist table names
   - Validate search queries

2. **Fix SQL injection** (1-2 hours)
   - Use parameterized queries everywhere
   - Remove f-string SQL queries

3. **Centralize database connections** (4-6 hours)
   - Create single DatabaseService class
   - Add connection pooling
   - Implement proper error handling

4. **Add user-friendly error messages** (2-3 hours)
   - Implement error handling wrapper
   - Add logging
   - Hide technical details from users

5. **Extract configuration** (2-3 hours)
   - Create config.py
   - Move all hardcoded values
   - Environment-based configuration

---

## 📈 LONG-TERM IMPROVEMENTS

1. **Microservices Architecture**
   - Separate API service for data fetching
   - Background job processor for long-running tasks
   - Redis for caching and session management

2. **Testing Infrastructure**
   - Unit tests for business logic
   - Integration tests for database queries
   - End-to-end tests for critical workflows

3. **Performance Optimization**
   - Implement pagination throughout
   - Add database indexes
   - Use materialized views for complex queries
   - Implement CDN for static assets

4. **Advanced Features**
   - Real-time data updates via WebSocket
   - Export to multiple formats (Excel, PDF)
   - Scheduled report generation
   - User dashboards and preferences

---

## 🔍 CODE QUALITY METRICS

**Current State:**
- **Lines of Code:** 1,597 (main file)
- **Cyclomatic Complexity:** High (>10 in several functions)
- **Code Duplication:** ~15-20% estimated
- **Test Coverage:** 0%
- **Technical Debt Ratio:** ~35-40%

**Recommended Targets:**
- **Max File Size:** 300 lines
- **Cyclomatic Complexity:** <8 per function
- **Code Duplication:** <5%
- **Test Coverage:** >80%
- **Technical Debt Ratio:** <15%

---

## ✅ IMPLEMENTATION CHECKLIST

### Phase 1: Security & Stability (Week 1-2)
- [ ] Fix SQL injection vulnerabilities
- [ ] Add input validation with Pydantic
- [ ] Implement proper error handling
- [ ] Add logging infrastructure
- [ ] Create database connection pooling

### Phase 2: Organization & Maintainability (Week 3-4)
- [ ] Refactor into modular structure
- [ ] Extract configuration
- [ ] Remove code duplication
- [ ] Add type hints throughout
- [ ] Write documentation

### Phase 3: Performance & UX (Week 5-6)
- [ ] Implement pagination
- [ ] Optimize cache strategy
- [ ] Add background job processing
- [ ] Improve error messages
- [ ] Add loading indicators

### Phase 4: Testing & Monitoring (Week 7-8)
- [ ] Write unit tests
- [ ] Add integration tests
- [ ] Implement monitoring
- [ ] Add performance tracking
- [ ] Set up error alerting

---

## 📚 RESOURCES

### Libraries to Add
```bash
pip install pydantic tenacity ratelimit sentry-sdk opentelemetry-api streamlit-analytics
```

### Recommended Reading
- [Streamlit Best Practices](https://docs.streamlit.io/library/advanced-features/configuration)
- [SQLAlchemy Performance Guide](https://docs.sqlalchemy.org/en/20/faq/performance.html)
- [Secure Coding in Python](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

---

**Document Version:** 1.0  
**Last Updated:** February 17, 2026  
**Next Review:** March 17, 2026
