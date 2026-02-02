# Clinical-Trial-Analytics-Dashboard
Developed a full-stack data workflow that automates ClinicalTrials.gov API extraction, data transformation, and visualization via analytical dashboards.




docker compose up -d


# How to run it
In a bash terminal go to root path of the repository where Docker-compose.yml is allocated and proceed

1. Build the images
```bash
docker compose build
```
2.Start the services
```bash
docker compose up -d
```
3. Download the model so the API can use it
```bash
docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2
```

# How it was developed

1.  Implement ddbb mysql with docker and retrive data from goverment's API (set a maximum of 100 pages tha involve 10000 studies)
2. Implement streamlit frontend with docker to add up funtionalities.
3. Create&test inital page of streamlit to download data from API and fill the mySQL ddbb.
4. Create&test of initial overview of data in a table and its completness
    Prompt example used with Copilot:
```
    change data overview with a table selector and then show:

    -  Initially the table raw data
    -  The Data Availability Report including missing data
``` 

criteria of colors of text in availability's table:

    🟢 Green text (#00FF00): >= 95% availability

    🟡 Yellow text (#FFD700): > 10% and < 95% availability

    🔴 Red text (#FF6B6B): <= 10% availability

5. Fill missing columns using info from ddbb: 

| Column     | Table      | Methodology |
|------------|------------|-------------|
| continent | Locations | Derive from existing `country` column using Python packages `pycountry` (country lookup/matching) and `pycountry-convert` (country code → continent conversion): fuzzy match country name to alpha-2 code, map to continent code, convert to name (e.g., "United States" → "US" → "NA" → "North America"). [file:1] |
| mesh_term | conditions | Populate missing `mesh_term` via Medical Subject Headings (MeSH) API: extract keywords from `condition_name`, query NLM MeSH API (`https://id.nlm.nih.gov/mesh/lookup/descriptor`), match closest MeSH descriptor and retrieve term. Handle fuzzy matching for variants; batch requests for efficiency. [file:1] |


6. Highlight document inconsistences: 

| Entity Relationship Diagram      | Table Definitions (.sql)      | resolution adopted |
|------------|------------|-------------|
| no "time_perspective"  | time_perspective VARCHAR(50)   | As it is not found in the downloaded json I delete this column |
| "type" column in  interventions table  | "intervention_type" column in  interventions table   |set "intervention_type" column in  interventions table   |