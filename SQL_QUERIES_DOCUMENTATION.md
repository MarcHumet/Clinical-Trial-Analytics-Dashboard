# SQL Queries Documentation

This document contains all SQL queries used in the Clinical Trial Analytics Dashboard project, organized by file and purpose.

---

## Table of Contents
1. [Database Schema Definition](#database-schema-definition)
2. [Database Verification Queries](#database-verification-queries)
3. [Application Queries](#application-queries)
4. [Enrollment Success Analytics](#enrollment-success-analytics)
5. [Data Update Operations](#data-update-operations)

---

## Database Schema Definition

### File: `ddbb/ddbb.sql`

#### Studies Table
```sql
CREATE TABLE studies (
	study_id INTEGER NOT NULL, 
	nct_id VARCHAR(20) NOT NULL, 
	title TEXT, 
	acronym VARCHAR(50), 
	status VARCHAR(50), 
	phase VARCHAR(50), 
	study_type VARCHAR(50), 
	start_date DATE, 
	completion_date DATE, 
	primary_completion_date DATE, 
	enrollment INTEGER, 
	enrollment_type VARCHAR(20), 
	brief_summary TEXT, 
	eligibility_criteria TEXT, 
	minimum_age VARCHAR(20), 
	maximum_age VARCHAR(20), 
	gender VARCHAR(20), 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (study_id), 
	UNIQUE (nct_id)
);

CREATE INDEX idx_studies_start_date ON studies (start_date);
CREATE INDEX idx_studies_status ON studies (status);
CREATE INDEX idx_studies_phase ON studies (phase);
```

#### Conditions Table
```sql
CREATE TABLE conditions (
	condition_id INTEGER NOT NULL, 
	study_id INTEGER, 
	condition_name VARCHAR(255) NOT NULL, 
	mesh_term VARCHAR(255), 
	PRIMARY KEY (condition_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);

CREATE INDEX idx_conditions_name ON conditions (condition_name);
```

#### Interventions Table
```sql
CREATE TABLE interventions (
	intervention_id INTEGER NOT NULL, 
	study_id INTEGER, 
	intervention_type VARCHAR(50), 
	name VARCHAR(255), 
	description TEXT, 
	PRIMARY KEY (intervention_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
```

#### Locations Table
```sql
CREATE TABLE locations (
	location_id INTEGER NOT NULL, 
	study_id INTEGER, 
	facility VARCHAR(255), 
	city VARCHAR(100), 
	state VARCHAR(100), 
	country VARCHAR(100), 
	continent VARCHAR(50), 
	PRIMARY KEY (location_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);

CREATE INDEX idx_locations_country ON locations (country);
```

#### Outcomes Table
```sql
CREATE TABLE outcomes (
	outcome_id INTEGER NOT NULL, 
	study_id INTEGER, 
	outcome_type VARCHAR(20), 
	measure TEXT, 
	time_frame VARCHAR(255), 
	description TEXT, 
	PRIMARY KEY (outcome_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
```

#### Sponsors Table
```sql
CREATE TABLE sponsors (
	sponsor_id INTEGER NOT NULL, 
	study_id INTEGER, 
	agency VARCHAR(255), 
	agency_class VARCHAR(50), 
	lead_or_collaborator VARCHAR(20), 
	PRIMARY KEY (sponsor_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);

CREATE INDEX idx_sponsors_agency ON sponsors (agency);
```

#### Study Design Table
```sql
CREATE TABLE study_design (
	design_id INTEGER NOT NULL, 
	study_id INTEGER, 
	allocation VARCHAR(50), 
	intervention_model VARCHAR(100), 
	masking VARCHAR(100), 
	primary_purpose VARCHAR(50), 
	observational_model VARCHAR(50), 
	PRIMARY KEY (design_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
```

---

## Database Verification Queries

### File: `ddbb/check_ddbb.py`

#### Show All Tables
```sql
SHOW TABLES
```

#### Count Records in Table
```sql
SELECT COUNT(*) AS cnt FROM `{table_name}`
```

#### Get Column Information
```sql
SELECT COLUMN_NAME, COLUMN_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
```

#### Sample Studies Data
```sql
SELECT study_id, nct_id, title, status, phase 
FROM studies 
LIMIT 3
```

#### Conditions Summary
```sql
SELECT c.condition_name, COUNT(*) as cnt 
FROM conditions c 
GROUP BY c.condition_name 
LIMIT 5
```

#### Interventions by Type
```sql
SELECT intervention_type, COUNT(*) as cnt 
FROM interventions 
GROUP BY intervention_type
```

#### Outcomes by Type
```sql
SELECT outcome_type, COUNT(*) as cnt 
FROM outcomes 
GROUP BY outcome_type
```

#### Sponsors by Agency Class
```sql
SELECT agency_class, COUNT(*) as cnt 
FROM sponsors 
WHERE agency_class IS NOT NULL 
GROUP BY agency_class
```

#### Locations by Country
```sql
SELECT country, COUNT(*) as cnt 
FROM locations 
WHERE country IS NOT NULL 
GROUP BY country 
LIMIT 5
```

#### List All Indexes
```sql
SELECT DISTINCT INDEX_NAME 
FROM INFORMATION_SCHEMA.STATISTICS 
WHERE TABLE_SCHEMA=%s
```

---

## Application Queries

### File: `src/app.py`

#### Main Dashboard Query - Comprehensive Study Data
```sql
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
```

#### Check Total Studies
```sql
SELECT COUNT(*) as total 
FROM studies
```

#### Count Records in Selected Table
```sql
SELECT COUNT(*) as count 
FROM {selected_table}
```

#### Get All Data from Table
```sql
SELECT * 
FROM {selected_table} 
LIMIT 100
```

#### Describe Table Structure
```sql
DESCRIBE {selected_table}
```

#### Data Completeness Check
```sql
SELECT COUNT(*) as total, 
       SUM(CASE WHEN `{col}` IS NULL THEN 1 ELSE 0 END) as null_count 
FROM {selected_table}
```

#### Complete Records Count
```sql
SELECT COUNT(*) as complete_count 
FROM {selected_table} 
WHERE {column conditions with IS NOT NULL}
```

#### Distinct Value Count
```sql
SELECT COUNT(*) as total, 
       COUNT(DISTINCT `{col}`) as distinct_count 
FROM {selected_table} 
WHERE `{col}` IS NOT NULL
```

#### Top Values by Column
```sql
SELECT `{col}`, COUNT(*) as count 
FROM {selected_table} 
WHERE `{col}` IS NOT NULL 
GROUP BY `{col}` 
ORDER BY count DESC 
LIMIT 5
```

#### Studies by Status
```sql
SELECT status, COUNT(*) as count 
FROM studies 
WHERE status IS NOT NULL 
GROUP BY status 
ORDER BY count DESC
```

#### Studies by Phase
```sql
SELECT phase, COUNT(*) as count 
FROM studies 
WHERE phase IS NOT NULL 
GROUP BY phase 
ORDER BY count DESC
```

#### Studies by Gender
```sql
SELECT gender, COUNT(*) as count 
FROM studies 
WHERE gender IS NOT NULL 
GROUP BY gender 
ORDER BY count DESC
```

#### Enrollment Statistics
```sql
SELECT enrollment 
FROM studies 
WHERE enrollment IS NOT NULL
```

#### Enrollment Averages and Extremes
```sql
SELECT AVG(enrollment) as avg_enrollment, 
       MIN(enrollment) as min, 
       MAX(enrollment) as max, 
       COUNT(*) as total_studies 
FROM studies 
WHERE enrollment IS NOT NULL
```

#### Zero Enrollment Count
```sql
SELECT COUNT(*) as zero_enrollment 
FROM studies 
WHERE enrollment = 0
```

#### Top Conditions
```sql
SELECT condition_name, COUNT(*) as count 
FROM conditions 
GROUP BY condition_name 
ORDER BY count DESC 
LIMIT 15
```

#### Interventions by Type
```sql
SELECT intervention_type, COUNT(*) as count 
FROM interventions 
WHERE intervention_type IS NOT NULL 
GROUP BY intervention_type 
ORDER BY count DESC
```

#### Outcomes by Type
```sql
SELECT outcome_type, COUNT(*) as count 
FROM outcomes 
WHERE outcome_type IS NOT NULL 
GROUP BY outcome_type 
ORDER BY count DESC
```

#### Top Sponsor Agencies
```sql
SELECT agency, COUNT(*) as count 
FROM sponsors 
GROUP BY agency 
ORDER BY count DESC 
LIMIT 15
```

#### Top Countries with Studies
```sql
SELECT country, COUNT(*) as count 
FROM locations 
WHERE country IS NOT NULL 
GROUP BY country 
ORDER BY count DESC 
LIMIT 15
```

#### Study Design - Allocation
```sql
SELECT allocation, COUNT(*) as count 
FROM study_design 
WHERE allocation IS NOT NULL 
GROUP BY allocation 
ORDER BY count DESC
```

#### Study Design - Primary Purpose
```sql
SELECT primary_purpose, COUNT(*) as count 
FROM study_design 
WHERE primary_purpose IS NOT NULL 
GROUP BY primary_purpose 
ORDER BY count DESC
```

---

## Time Trends Analysis

### File: `src/app.py` (Time Trends Section)

#### Studies by Year
```sql
SELECT YEAR(start_date) as year, COUNT(*) as study_count 
FROM studies 
WHERE start_date IS NOT NULL 
GROUP BY YEAR(start_date) 
ORDER BY year
```

#### Phase Distribution Over Time
```sql
SELECT YEAR(start_date) as year, phase, COUNT(*) as count 
FROM studies 
WHERE start_date IS NOT NULL AND phase IS NOT NULL 
GROUP BY YEAR(start_date), phase 
ORDER BY year, phase
```

#### Enrollment Trends by Year
```sql
SELECT YEAR(start_date) as year, 
       AVG(enrollment) as avg_enrollment,
       COUNT(*) as study_count
FROM studies 
WHERE start_date IS NOT NULL AND enrollment IS NOT NULL 
GROUP BY YEAR(start_date) 
ORDER BY year
```

#### Study Completion Trends
```sql
SELECT YEAR(completion_date) as year, COUNT(*) as completed_count 
FROM studies 
WHERE completion_date IS NOT NULL 
GROUP BY YEAR(completion_date) 
ORDER BY year
```

#### Top Conditions Over Time
```sql
SELECT YEAR(s.start_date) as year, c.condition_name, COUNT(*) as count
FROM studies s
JOIN conditions c ON s.study_id = c.study_id
WHERE s.start_date IS NOT NULL 
GROUP BY YEAR(s.start_date), c.condition_name
HAVING COUNT(*) >= 3
ORDER BY year, count DESC
```

---

## Enrollment Success Analytics

### File: `src/enrollment_success.py`

#### Load All Studies
```sql
SELECT * 
FROM studies
```

---

## Data Update Operations

### File: `ddbb/search_condition_name.py`

#### Get Distinct Condition Names
```sql
SELECT DISTINCT condition_name 
FROM conditions
```

#### Update Conditions with MeSH Terms (Commented Out)
```sql
UPDATE conditions
SET mesh_term = %s
WHERE condition_name = %s
```

---

## Location Data Updates

### File: `ddbb/update_continent_from_country.py`

#### Get All Distinct Countries
```sql
SELECT DISTINCT country 
FROM locations 
WHERE country IS NOT NULL
```

#### Update Continent for Country
```sql
UPDATE locations 
SET continent = %s 
WHERE country = %s
```

---

## Summary Statistics

### Total Number of SQL Queries Identified: **50+**

### Query Categories:
- **Schema Definition**: 7 CREATE TABLE statements + 6 CREATE INDEX statements
- **Data Retrieval**: 35+ SELECT queries
- **Data Modification**: 2 UPDATE queries
- **Metadata Queries**: 4 INFORMATION_SCHEMA queries
- **Aggregation Queries**: 25+ GROUP BY/COUNT queries
- **Join Queries**: 3 complex JOIN queries

### Most Common Operations:
1. `SELECT` with `COUNT(*)` - Data aggregation
2. `GROUP BY` - Categorization and grouping
3. `WHERE` clauses - Data filtering
4. `JOIN` operations - Relational data retrieval
5. `ORDER BY` with `LIMIT` - Top-N queries

---

## Notes

- Most queries use parameterized inputs to prevent SQL injection
- Many queries include `IS NOT NULL` filters to handle data quality
- Time-based analysis extensively uses `YEAR()` function
- String concatenation uses `GROUP_CONCAT()` for aggregating related data
- Indexes are strategically placed on frequently queried columns (dates, status, phase, country)

---

*Last Updated: February 17, 2026*
*Generated from Clinical Trial Analytics Dashboard codebase*
