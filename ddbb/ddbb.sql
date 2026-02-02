-- studies definition

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


-- conditions definition

CREATE TABLE conditions (
	condition_id INTEGER NOT NULL, 
	study_id INTEGER, 
	condition_name VARCHAR(255) NOT NULL, 
	mesh_term VARCHAR(255), 
	PRIMARY KEY (condition_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);

CREATE INDEX idx_conditions_name ON conditions (condition_name);


-- interventions definition

CREATE TABLE interventions (
	intervention_id INTEGER NOT NULL, 
	study_id INTEGER, 
	intervention_type VARCHAR(50), 
	name VARCHAR(255), 
	description TEXT, 
	PRIMARY KEY (intervention_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);


-- locations definition

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


-- outcomes definition

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


-- sponsors definition

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


-- study_design definition

CREATE TABLE study_design (
	design_id INTEGER NOT NULL, 
	study_id INTEGER, 
	allocation VARCHAR(50), 
	intervention_model VARCHAR(100), 
	masking VARCHAR(100), 
	primary_purpose VARCHAR(50), 
	observational_model VARCHAR(50), 
	time_perspective VARCHAR(50), 
	PRIMARY KEY (design_id), 
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);