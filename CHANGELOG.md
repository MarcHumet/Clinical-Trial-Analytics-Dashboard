# Changelog

All notable changes to the Clinical Trial Analytics Dashboard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-24

### Added

#### Security
- Input validation system (`src/validators.py`) for all user inputs
- Table name whitelist to prevent SQL injection attacks
- Search query sanitization with length limits
- Column name validation for dynamic queries

#### Infrastructure
- Centralized database service (`src/database_service.py`) with connection pooling
- SQLAlchemy connection pool (pool_size=10, max_overflow=20)
- Automatic connection recycling (1 hour)
- Pool pre-ping for connection health checks
- Retry logic with exponential backoff for transient failures

#### Configuration
- Centralized configuration file (`src/config.py`)
- `DashboardConfig` dataclass for all settings:
  - Database settings (pool sizes, timeouts)
  - API settings (URLs, page sizes)
  - Cache settings (TTL, max entries)
  - Enrollment thresholds and benchmarks
  - Status scores and composite weights
  - UI theme colors
  - Query limits and security settings

#### Error Handling
- Error handling utility (`src/error_handler.py`)
- User-friendly error message mapping
- Contextual error logging
- `safe_execute()` wrapper for resilient operations
- Technical details optionally hidden from end users

### Changed

#### Database Operations
- Replaced `DESCRIBE {table}` with secure `information_schema` queries
- Updated enrollment data loading to use `DatabaseService`
- Refactored Data Overview page to use validated queries
- Updated search functionality with input validation
- Database empty check now uses centralized service

#### Code Quality
- Replaced hardcoded values with configuration references
- Using `config.ENROLLMENT_EXCELLENT_THRESHOLD` instead of magic number `50`
- Using `config.STATUS_SCORES` instead of inline dictionary
- Theme colors now reference `config.COLOR_*` constants
- Cache TTL now uses `config.CACHE_TTL`

#### Error Messages
- Improved error messages in enrollment data loading
- Better fallback behavior when database is unavailable
- Warnings instead of errors for non-critical failures

### Deprecated
- `get_db_connection()` - Use `DatabaseService.get_connection()` instead
- `get_db_connection_with_retry()` - Use `DatabaseService.execute_query_with_retry()` instead
- Direct SQL string interpolation - Use parameterized queries with validation

### Security
- **CRITICAL**: Fixed SQL injection vulnerability in `DESCRIBE` queries
- **CRITICAL**: Fixed SQL injection vulnerability in table selection
- **CRITICAL**: Added input validation for all user-provided data
- Implemented whitelist validation for table names
- Sanitized search queries before database operations
- Validated column names before dynamic query construction

### Fixed
- Database connection pattern anti-pattern (multiple connection methods)
- Missing retry logic for transient database failures
- Poor error messages exposing technical details
- Hardcoded configuration values scattered throughout code
- Inconsistent use of mysql.connector and sqlalchemy

## [1.0.0] - 2026-02-17

### Initial Release
- Basic Streamlit dashboard for clinical trial analytics
- API integration with ClinicalTrials.gov
- MySQL database for data storage
- Docker compose setup
- Enrollment success analytics
- Data overview and completeness analysis
- Distribution analysis
- Time trends visualization
- Jupyter notebook for exploratory data analysis

---

## Migration Guide

### For Developers

#### Using the New Database Service

**Before:**
```python
conn = get_db_connection_with_retry(retries=3, delay=0.5)
cursor = conn.cursor(dictionary=True)
cursor.execute(f"SELECT * FROM {table_name}")
results = cursor.fetchall()
conn.close()
```

**After:**
```python
from src.database_service import DatabaseService
from src.validators import validator

# Validate input
validated_table = validator.validate_table_name(table_name)

# Use DatabaseService with retry logic
query = f"SELECT * FROM {validated_table}"
df = DatabaseService.execute_query_with_retry(query)
```

#### Using Configuration

**Before:**
```python
if enrollment_rate > 50:
    tier = "Excellent"
```

**After:**
```python
from src.config import config

if enrollment_rate > config.ENROLLMENT_EXCELLENT_THRESHOLD:
    tier = "Excellent"
```

#### Using Error Handling

**Before:**
```python
try:
    result = risky_operation()
except Exception as e:
    st.error(f"Error: {str(e)}")
```

**After:**
```python
from src.error_handler import ErrorHandler, safe_execute

# Option 1: Manual error handling
try:
    result = risky_operation()
except Exception as e:
    ErrorHandler.handle_error(e, context="operation name", show_user=True)

# Option 2: Safe wrapper
result = safe_execute(
    risky_operation,
    context="operation name",
    fallback_value=default_value
)
```

### Breaking Changes

None. All changes maintain backward compatibility. Legacy functions are deprecated but still functional.

### Testing Notes

- ✅ All new modules import successfully
- ✅ Database engine creation with pooling verified
- ✅ Configuration loading tested
- ✅ Validator functions tested
- ⏳ Full end-to-end testing requires running database instance

### Known Issues

- Some sections (Distribution Analysis, Time Trends) still use legacy database connections
  - Functionality preserved, will be updated in Phase 2
- Async operations for long-running tasks not yet implemented (planned for Phase 2)
- Memory pagination for large datasets not yet implemented (planned for Phase 2)
