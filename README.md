# Clinical-Trial-Analytics-Dashboard
Developed a full-stack data workflow that automates ClinicalTrials.gov API extraction, data transformation, and visualization via analytical dashboards.




docker compose up -d


# How to run it

## Quick Start (Docker)
```bash
# Clone and start
git clone https://github.com/MarcHumet/Clinical-Trial-Analytics-Dashboard.git
cd Clinical-Trial-Analytics-Dashboard
docker-compose up -d
```
When process is finished, 2 dockers should be running. To get the Stremlit frontend, navigate in a Browser to the next url:

    http://localhost:8501/

Fortunatelly, Streamlit page should appear!

FIRST TIME USE you should press  button  "Fill MySQL DB" in the "home" section to fill ddbb with data from the API!! (You should repush the buttoh if ddbb is emptied)
 

[Jupyter Notebook](CLINICAL_TRIAL_ANALYTICAL_EDA.ipynb) is also located in this report, 


# How it was developed

1. Implement ddbb mysql with docker and retrive data from goverment's API (set a maximum of 100 pages tha involve 10000 studies)
2. Implement streamlit frontend with docker to add up funtionalities.
3. Create&test inital page of streamlit to download data from API and fill the mySQL ddbb.

4. Fill missing columns using info from ddbb: 

| Column     | Table      | Methodology |
|------------|------------|-------------|
| continent | Locations | Derive from existing `country` column using Python packages `pycountry` (country lookup/matching) and `pycountry-convert` (country code → continent conversion): fuzzy match country name to alpha-2 code, map to continent code, convert to name (e.g., "United States" → "US" → "NA" → "North America"). |
| mesh_term | conditions | Populate missing `mesh_term` via Medical Subject Headings (MeSH) API: extract keywords from `condition_name`, query NLM MeSH API (`https://id.nlm.nih.gov/mesh/lookup/descriptor`), match closest MeSH descriptor and retrieve term. Handle fuzzy matching for variants; batch requests for efficiency. (Not implemented in ddbb, process too long to accomplish) |

5. Create&test of initial overview of data in a table and its completness

6. Generate full Jupyter Notebook
7. Add enrollment success section 
8. Integral test and Document 

 AI guidance and bi-coding was used estensivaly when working with VSC with copilot. Much less used when working on Jupyter Notebook.


# Answer to Business questions
**Relevant questions** proposed in the initial DA test are answered in  the 5th section of the following Jupyter Notebook (where most data work is done using pandas and sql queries (mysql) besides other python packages):  

[Open the analysis notebook](CLINICAL_TRIAL_ANALYTICAL_EDA.ipynb)

# Bonus Questions
Please provide brief answers to these questions:
1. Stakeholder Communication: How would you adapt your dashboard and presentation for a non-technical executive
versus a clinical operations manager?
First understand needs and knowdelege level to adapt kind of graphs and 
2. Data Quality at Scale: What automated data quality checks would you implement if this pipeline ran daily with new
incoming data?
    Next points showld be checked automatically:
    - Missing data (accepted ratio? critical data with 0% missing data allowed?)
    - Data Formats keeps the same way (create alerts and manage input for data formats variability according to historical format of input values and observed variations)
    - Check duplicity of data. 
    - Check for unexpected long performance time for early degradation detection of the pipeline.
3. Self-Service Analytics: How would you design this solution to enable stakeholders to explore the data themselves
without your direct involvement?
    - Digest data (clean, filter, transform) to provide high data quality. 
    - Actionable dashboard where stakeholders can play with requested variables to check agreed KPIs or relevant variables. Preferibly visual output adapted to stakeholder knowdlege!
4. Compliance Considerations: If this were a GxP-regulated environment, what additional documentation or validation
would be required for your analysis?
    The key point is to prove the whole process is repeatable, traceable, and secure. 
    Next documents should be written before start:
    - User requierements
    - Functional requirements
    - Traceability plan
    - Most relevant document: Statistical Analysis Plan (define how you handle missing data (Imputation) and outliers before you see the results. Archive final code, frozen dataset,... in a Trial Master File)
    After work is done:
    Validate the solution from the points of IQ (installation qualification), OQ (operational qualification) and PQ (performance qualification)
    All way long:
    Risk Analysis of Data Integrity and Security (ALCOA+)

5. Advanced Analytics: What predictive or machine learning models could add value to this clinical trial analytics use case?
    - Use predictive models for trial success: probability of trial completion (XGBoost / LightGBM /multivariable regression model)
    - Survival Analysis (Time-to-Event): predict when trial is going to fail (survival curve) with Random Survival Forests (RSF) or 
    DeepSurv
