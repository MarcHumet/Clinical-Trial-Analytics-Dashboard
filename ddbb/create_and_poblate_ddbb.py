from sqlalchemy import create_engine, Column, Integer, String, Text, Date, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pydantic import BaseModel, Field, field_validator, ConfigDict
import requests
import time
import os
from dotenv import load_dotenv
from faker import Faker
import random
from datetime import datetime, timedelta, date
import re
from typing import Optional, List
import json

# load .env (if present)
load_dotenv()

# Configuration from environment (.env)
API_URL = os.getenv('API_URL', 'https://clinicaltrials.gov/api/v2/studies')

# MySQL Database configuration from .env
MYSQL_USER = os.getenv('MYSQL_USER', 'user')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'pass')
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'clinicaltrials')

# Database URL for MySQL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

Base = declarative_base()
fake = Faker()


# ============================================================================
# Pydantic Validation Models
# ============================================================================

class StudySchema(BaseModel):
    """Validation schema for Study data"""
    model_config = ConfigDict(from_attributes=True)
    
    nct_id: str = Field(..., min_length=1, max_length=20)
    title: Optional[str] = None
    acronym: Optional[str] = Field(None, max_length=50)
    status: Optional[str] = Field(None, max_length=50)
    phase: Optional[str] = Field(None, max_length=50)
    study_type: Optional[str] = Field(None, max_length=50)
    start_date: Optional[date] = None
    completion_date: Optional[date] = None
    primary_completion_date: Optional[date] = None
    enrollment: Optional[int] = Field(None, ge=0)
    enrollment_type: Optional[str] = Field(None, max_length=20)
    brief_summary: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    minimum_age: Optional[str] = Field(None, max_length=20)
    maximum_age: Optional[str] = Field(None, max_length=20)
    gender: Optional[str] = Field(None, max_length=20)
    
    @field_validator('nct_id')
    @classmethod
    def validate_nct_id(cls, v):
        if not v.startswith('NCT'):
            raise ValueError('NCT ID must start with NCT')
        return v


class ConditionSchema(BaseModel):
    """Validation schema for Condition data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    condition_name: str = Field(..., min_length=1, max_length=255)
    mesh_term: Optional[str] = Field(None, max_length=255)


class InterventionSchema(BaseModel):
    """Validation schema for Intervention data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    intervention_type: Optional[str] = Field(None, max_length=50)
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class OutcomeSchema(BaseModel):
    """Validation schema for Outcome data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    outcome_type: Optional[str] = Field(None, max_length=20)
    measure: Optional[str] = None
    time_frame: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class SponsorSchema(BaseModel):
    """Validation schema for Sponsor data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    agency: Optional[str] = Field(None, max_length=255)
    agency_class: Optional[str] = Field(None, max_length=50)
    lead_or_collaborator: Optional[str] = Field(None, max_length=20)


class LocationSchema(BaseModel):
    """Validation schema for Location data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    facility: Optional[str] = Field(None, max_length=255)
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)
    continent: Optional[str] = Field(None, max_length=50)


class StudyDesignSchema(BaseModel):
    """Validation schema for StudyDesign data"""
    model_config = ConfigDict(from_attributes=True)
    
    study_id: int = Field(..., gt=0)
    allocation: Optional[str] = Field(None, max_length=50)
    intervention_model: Optional[str] = Field(None, max_length=100)
    masking: Optional[str] = Field(None, max_length=100)
    primary_purpose: Optional[str] = Field(None, max_length=50)
    observational_model: Optional[str] = Field(None, max_length=50)
    time_perspective: Optional[str] = Field(None, max_length=50)


# ============================================================================
# SQLAlchemy Models
# ============================================================================

class Study(Base):
    __tablename__ = "studies"
    
    study_id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(String(20), unique=True, nullable=False)
    title = Column(Text)
    acronym = Column(String(50))
    status = Column(String(50))
    phase = Column(String(50))
    study_type = Column(String(50))
    start_date = Column(Date)
    completion_date = Column(Date)
    primary_completion_date = Column(Date)
    enrollment = Column(Integer)
    enrollment_type = Column(String(20))
    brief_summary = Column(Text)
    eligibility_criteria = Column(Text)
    minimum_age = Column(String(20))
    maximum_age = Column(String(20))
    gender = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    conditions = relationship("Condition", back_populates="study", cascade="all, delete-orphan")
    interventions = relationship("Intervention", back_populates="study", cascade="all, delete-orphan")
    outcomes = relationship("Outcome", back_populates="study", cascade="all, delete-orphan")
    sponsors = relationship("Sponsor", back_populates="study", cascade="all, delete-orphan")
    locations = relationship("Location", back_populates="study", cascade="all, delete-orphan")
    study_design = relationship("StudyDesign", back_populates="study", cascade="all, delete-orphan", uselist=False)
    
    __table_args__ = (
        Index('idx_studies_status', 'status'),
        Index('idx_studies_phase', 'phase'),
        Index('idx_studies_start_date', 'start_date'),
    )


class Condition(Base):
    __tablename__ = "conditions"
    
    condition_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    condition_name = Column(String(255), nullable=False)
    mesh_term = Column(String(255))
    
    study = relationship("Study", back_populates="conditions")
    
    __table_args__ = (
        Index('idx_conditions_name', 'condition_name'),
    )


class Intervention(Base):
    __tablename__ = "interventions"
    
    intervention_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    intervention_type = Column(String(50))
    name = Column(String(255))
    description = Column(Text)
    
    study = relationship("Study", back_populates="interventions")


class Outcome(Base):
    __tablename__ = "outcomes"
    
    outcome_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    outcome_type = Column(String(20))
    measure = Column(Text)
    time_frame = Column(String(255))
    description = Column(Text)
    
    study = relationship("Study", back_populates="outcomes")


class Sponsor(Base):
    __tablename__ = "sponsors"
    
    sponsor_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    agency = Column(String(255))
    agency_class = Column(String(50))
    lead_or_collaborator = Column(String(20))
    
    study = relationship("Study", back_populates="sponsors")
    
    __table_args__ = (
        Index('idx_sponsors_agency', 'agency'),
    )


class Location(Base):
    __tablename__ = "locations"
    
    location_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    facility = Column(String(255))
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    continent = Column(String(50))
    
    study = relationship("Study", back_populates="locations")
    
    __table_args__ = (
        Index('idx_locations_country', 'country'),
    )


class StudyDesign(Base):
    __tablename__ = "study_design"
    
    design_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    allocation = Column(String(50))
    intervention_model = Column(String(100))
    masking = Column(String(100))
    primary_purpose = Column(String(50))
    observational_model = Column(String(50))
    time_perspective = Column(String(50))
    
    study = relationship("Study", back_populates="study_design")


# ============================================================================
# Database Functions
# ============================================================================


def parse_date_field(val):
    """Try several date formats and return a date object or None.

    Accepts strings like 'YYYY-MM-DD', 'YYYY-MM', 'YYYY' and returns
    a datetime.date using the first day of the month when day is missing.
    """
    if not val:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    s = str(val).strip()
    # Try common formats
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            pass
    # Try ISO
    try:
        return datetime.fromisoformat(s).date()
    except Exception:
        pass
    # Fallback: regex for YYYY-MM
    m = re.search(r"(\d{4})-(\d{1,2})", s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        try:
            return date(y, mo, 1)
        except Exception:
            return None
    return None

def create_engine_and_session():
    """Create database engine and session"""
    try:
        # For MySQL, use specific engine options
        engine = create_engine(
            DATABASE_URL, 
            echo=False,
            pool_pre_ping=True,  # Verify connection before use
            pool_recycle=3600,   # Recycle connections every hour
        )
        Session = sessionmaker(bind=engine)
        return engine, Session
    except Exception as e:
        print(f"Error creating engine: {e}")
        return None, None


def create_schema(engine):
    """Create all database tables"""
    try:
        # Drop all existing tables first to ensure clean schema
        Base.metadata.drop_all(engine)
        print("✓ Dropped existing schema")
        
        # Create all tables fresh
        Base.metadata.create_all(engine)
        print("✓ Schema created successfully")
    except Exception as e:
        print(f"Error creating schema: {e}")


def generate_studies(session, num_studies: int = 50):
    """Generate and insert sample studies"""
    statuses = ["Not yet recruiting", "Recruiting", "Enrolling by invitation", 
                "Active, not recruiting", "Completed", "Terminated", "Suspended"]
    phases = ["Early Phase 1", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "N/A"]
    study_types = ["Interventional", "Observational", "Observational [Patient Registry]"]
    enrollment_types = ["Actual", "Anticipated"]
    genders = ["All", "Female", "Male"]
    
    studies = []
    validation_errors = []
    
    for i in range(num_studies):
        try:
            nct_id = f"NCT{fake.random_int(10000000, 99999999)}"
            start_date = fake.date_between(start_date='-5y')
            completion_date = fake.date_between(start_date=start_date) if random.random() > 0.3 else None
            primary_completion_date = completion_date if random.random() > 0.4 else None
            
            # Validate with Pydantic
            study_data = StudySchema(
                nct_id=nct_id,
                title=fake.sentence(nb_words=6),
                acronym=fake.word().upper(),
                status=random.choice(statuses),
                phase=random.choice(phases),
                study_type=random.choice(study_types),
                start_date=start_date,
                completion_date=completion_date,
                primary_completion_date=primary_completion_date,
                enrollment=random.randint(10, 1000),
                enrollment_type=random.choice(enrollment_types),
                brief_summary=fake.text(max_nb_chars=200),
                eligibility_criteria=fake.text(max_nb_chars=300),
                minimum_age=f"{random.randint(18, 65)} years" if random.random() > 0.2 else "18 years and older",
                maximum_age=f"{random.randint(65, 100)} years" if random.random() > 0.3 else "100 years and older",
                gender=random.choice(genders)
            )
            
            study = Study(**study_data.model_dump())
            studies.append(study)
        except Exception as e:
            validation_errors.append(f"Study {i}: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Validation errors: {len(validation_errors)}")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    session.add_all(studies)
    session.commit()
    print(f"✓ Generated {len(studies)} studies")
    return studies


def generate_conditions(session, studies):
    """Generate and insert sample conditions"""
    conditions_list = [
        "Diabetes", "Hypertension", "Cancer", "Asthma", "Heart Disease",
        "Depression", "Arthritis", "Obesity", "Chronic Obstructive Pulmonary Disease",
        "Alzheimer Disease", "Parkinson Disease", "Multiple Sclerosis"
    ]
    
    conditions = []
    validation_errors = []
    
    for study in studies:
        num_conditions = random.randint(1, 4)
        for _ in range(num_conditions):
            try:
                condition_data = ConditionSchema(
                    study_id=study.study_id,
                    condition_name=random.choice(conditions_list),
                    mesh_term=f"D{fake.random_int(10000, 99999)}"
                )
                
                condition = Condition(**condition_data.model_dump())
                conditions.append(condition)
            except Exception as e:
                validation_errors.append(f"Condition: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Condition validation errors: {len(validation_errors)}")
    
    session.add_all(conditions)
    session.commit()
    print(f"✓ Generated {len(conditions)} conditions")


def generate_interventions(session, studies):
    """Generate and insert sample interventions"""
    intervention_types = ["Drug", "Procedure", "Behavioral", "Device", "Biological", "Dietary Supplement"]
    
    interventions = []
    validation_errors = []
    
    for study in studies:
        num_interventions = random.randint(1, 3)
        for _ in range(num_interventions):
            try:
                intervention_data = InterventionSchema(
                    study_id=study.study_id,
                    intervention_type=random.choice(intervention_types),
                    name=fake.word(),
                    description=fake.text(max_nb_chars=150)
                )
                
                intervention = Intervention(**intervention_data.model_dump())
                interventions.append(intervention)
            except Exception as e:
                validation_errors.append(f"Intervention: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Intervention validation errors: {len(validation_errors)}")
    
    session.add_all(interventions)
    session.commit()
    print(f"✓ Generated {len(interventions)} interventions")


def generate_outcomes(session, studies):
    """Generate and insert sample outcomes"""
    outcome_types = ["Primary", "Secondary", "Other Pre-specified"]
    
    outcomes = []
    validation_errors = []
    
    for study in studies:
        num_outcomes = random.randint(1, 3)
        for _ in range(num_outcomes):
            try:
                outcome_data = OutcomeSchema(
                    study_id=study.study_id,
                    outcome_type=random.choice(outcome_types),
                    measure=fake.sentence(nb_words=8),
                    time_frame=f"{random.randint(1, 36)} months",
                    description=fake.text(max_nb_chars=100)
                )
                
                outcome = Outcome(**outcome_data.model_dump())
                outcomes.append(outcome)
            except Exception as e:
                validation_errors.append(f"Outcome: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Outcome validation errors: {len(validation_errors)}")
    
    session.add_all(outcomes)
    session.commit()
    print(f"✓ Generated {len(outcomes)} outcomes")


def generate_sponsors(session, studies):
    """Generate and insert sample sponsors"""
    agency_classes = ["NIH", "U.S. Fed", "Other", "Industry", "University"]
    lead_types = ["lead", "collaborator"]
    
    sponsors = []
    validation_errors = []
    
    for study in studies:
        num_sponsors = random.randint(1, 3)
        for _ in range(num_sponsors):
            try:
                sponsor_data = SponsorSchema(
                    study_id=study.study_id,
                    agency=fake.company(),
                    agency_class=random.choice(agency_classes),
                    lead_or_collaborator=random.choice(lead_types)
                )
                
                sponsor = Sponsor(**sponsor_data.model_dump())
                sponsors.append(sponsor)
            except Exception as e:
                validation_errors.append(f"Sponsor: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Sponsor validation errors: {len(validation_errors)}")
    
    session.add_all(sponsors)
    session.commit()
    print(f"✓ Generated {len(sponsors)} sponsors")


def generate_locations(session, studies):
    """Generate and insert sample locations"""
    countries = ["United States", "Canada", "United Kingdom", "France", "Germany", 
                 "Australia", "Japan", "India", "Brazil", "Mexico"]
    continents = {
        "United States": "North America",
        "Canada": "North America",
        "United Kingdom": "Europe",
        "France": "Europe",
        "Germany": "Europe",
        "Australia": "Oceania",
        "Japan": "Asia",
        "India": "Asia",
        "Brazil": "South America",
        "Mexico": "North America"
    }
    
    locations = []
    validation_errors = []
    
    for study in studies:
        num_locations = random.randint(1, 5)
        for _ in range(num_locations):
            try:
                country = random.choice(countries)
                location_data = LocationSchema(
                    study_id=study.study_id,
                    facility=fake.company(),
                    city=fake.city(),
                    state=fake.state(),
                    country=country,
                    continent=continents.get(country, "Unknown")
                )
                
                location = Location(**location_data.model_dump())
                locations.append(location)
            except Exception as e:
                validation_errors.append(f"Location: {str(e)}")
    
    if validation_errors:
        print(f"⚠ Location validation errors: {len(validation_errors)}")
    
    session.add_all(locations)
    session.commit()
    print(f"✓ Generated {len(locations)} locations")


def generate_study_design(session, studies):
    """Generate and insert sample study design details"""
    allocations = ["Randomized", "Non-Randomized", "N/A"]
    models = ["Single Group Assignment", "Parallel Assignment", "Crossover Assignment", "Factorial Assignment"]
    maskings = ["None (Open Label)", "Single Blind", "Double Blind", "Triple Blind"]
    purposes = ["Treatment", "Prevention", "Diagnostic", "Supportive Care", "Basic Science", "Device Feasibility"]
    observational_models = ["Cohort", "Case-Control", "Cross-Sectional", "Case-Only", "Ecologic or Community Studies"]
    perspectives = ["Prospective", "Retrospective", "Cross-Sectional"]
    
    designs = []
    validation_errors = []
    
    for study in studies:
        try:
            design_data = StudyDesignSchema(
                study_id=study.study_id,
                allocation=random.choice(allocations),
                intervention_model=random.choice(models),
                masking=random.choice(maskings),
                primary_purpose=random.choice(purposes),
                observational_model=random.choice(observational_models),
                time_perspective=random.choice(perspectives)
            )
            
            design = StudyDesign(**design_data.model_dump())
            designs.append(design)
        except Exception as e:
            validation_errors.append(f"StudyDesign: {str(e)}")
    
    if validation_errors:
        print(f"⚠ StudyDesign validation errors: {len(validation_errors)}")
    
    session.add_all(designs)
    session.commit()
    print(f"✓ Generated {len(designs)} study designs")


def populate_database(num_studies: int = 50):
    """Main function to populate the database"""
    engine, Session = create_engine_and_session()
    
    if not engine or not Session:
        print("Failed to create database engine")
        return
    
    session = Session()
    
    try:
        print("Creating database schema...")
        create_schema(engine)
        
        print(f"Generating and populating {num_studies} studies...")
        studies = generate_studies(session, num_studies)
        
        generate_conditions(session, studies)
        generate_interventions(session, studies)
        generate_outcomes(session, studies)
        generate_sponsors(session, studies)
        generate_locations(session, studies)
        generate_study_design(session, studies)
        
        print("\n✓ Database population completed successfully!")
        
    except Exception as e:
        print(f"Error during database population: {e}")
        session.rollback()
    finally:
        session.close()


def load_studies_from_json(json_file: str):
    """Load studies from JSON file (ClinicalTrials.gov format)"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('studies', [])
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []


def fetch_studies_from_api(output_file: str, page_size: int = 100, max_pages: int | None = None, delay: float = 0.2):
    """Fetch studies from ClinicalTrials.gov API and save to a local JSON file.

    The API returns pages and a `nextPageToken` when more pages are available.
    This function iterates until there are no more pages or `max_pages` reached.
    """
    url = API_URL
    all_studies = []
    page_token = None
    page = 0

    while True:
        params = {"pageSize": page_size}
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"Error fetching from API: {e}")
            break

        studies = payload.get("studies") or payload.get("data") or []
        all_studies.extend(studies)

        page += 1
        print(f"Fetched page {page} - {len(studies)} studies (total {len(all_studies)})")

        next_token = payload.get("nextPageToken")
        if not next_token:
            break

        page_token = next_token
        if max_pages and page >= max_pages:
            print("Reached max_pages limit")
            break

        time.sleep(delay)

    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"studies": all_studies}, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_studies)} studies to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")



def extract_study_info(study_json: dict) -> dict:
    """Extract study information from JSON structure"""
    try:
        protocol = study_json.get('protocolSection', {})
        ident = protocol.get('identificationModule', {})
        status_mod = protocol.get('statusModule', {})
        design_mod = protocol.get('designModule', {})
        desc_mod = protocol.get('descriptionModule', {})
        
        # Extract dates
        start_date_struct = status_mod.get('startDateStruct', {})
        completion_date_struct = status_mod.get('completionDateStruct', {})
        primary_completion_struct = status_mod.get('primaryCompletionDateStruct', {})
        
        study_info = {
            'nct_id': ident.get('nctId', ''),
            'title': ident.get('briefTitle', ''),
            'acronym': ident.get('acronym', None),
            'status': status_mod.get('overallStatus', ''),
            'phase': design_mod.get('phases', [None])[0] if design_mod.get('phases') else None,
            'study_type': design_mod.get('studyType', ''),
            'start_date': parse_date_field(start_date_struct.get('date')),
            'completion_date': parse_date_field(completion_date_struct.get('date')),
            'primary_completion_date': parse_date_field(primary_completion_struct.get('date')),
            'enrollment': design_mod.get('enrollmentInfo', {}).get('count'),
            'enrollment_type': design_mod.get('enrollmentInfo', {}).get('type'),
            'brief_summary': desc_mod.get('briefSummary', ''),
            'eligibility_criteria': protocol.get('eligibilityModule', {}).get('eligibilityCriteria', ''),
            'gender': protocol.get('eligibilityModule', {}).get('sex', 'All'),
            'minimum_age': protocol.get('eligibilityModule', {}).get('minimumAge'),
            'maximum_age': protocol.get('eligibilityModule', {}).get('maximumAge'),
        }
        
        return study_info
    except Exception as e:
        print(f"Error extracting study info: {e}")
        return {}


def extract_conditions(study_json: dict) -> List[dict]:
    """Extract conditions from JSON"""
    try:
        conditions_mod = study_json.get('protocolSection', {}).get('conditionsModule', {})
        conditions_list = conditions_mod.get('conditions', [])
        return [{'condition_name': c} for c in conditions_list]
    except:
        return []


def extract_interventions(study_json: dict) -> List[dict]:
    """Extract interventions from JSON"""
    try:
        arms_mod = study_json.get('protocolSection', {}).get('armsInterventionsModule', {})
        interventions = arms_mod.get('interventions', [])
        
        result = []
        for intervention in interventions:
            result.append({
                'intervention_type': intervention.get('type'),
                'name': intervention.get('name'),
                'description': intervention.get('description'),
            })
        return result
    except:
        return []


def extract_outcomes(study_json: dict) -> List[dict]:
    """Extract outcomes from JSON"""
    try:
        outcomes_mod = study_json.get('protocolSection', {}).get('outcomesModule', {})
        outcomes_list = []
        
        # Primary outcomes
        for outcome in outcomes_mod.get('primaryOutcomes', []):
            outcomes_list.append({
                'outcome_type': 'Primary',
                'measure': outcome.get('measure'),
                'time_frame': outcome.get('timeFrame'),
                'description': outcome.get('description'),
            })
        
        # Secondary outcomes
        for outcome in outcomes_mod.get('secondaryOutcomes', []):
            outcomes_list.append({
                'outcome_type': 'Secondary',
                'measure': outcome.get('measure'),
                'time_frame': outcome.get('timeFrame'),
                'description': outcome.get('description'),
            })
        
        return outcomes_list
    except:
        return []


def extract_sponsors(study_json: dict) -> List[dict]:
    """Extract sponsors from JSON"""
    try:
        sponsors_mod = study_json.get('protocolSection', {}).get('sponsorCollaboratorsModule', {})
        
        sponsors_list = []
        
        # Lead sponsor
        lead = sponsors_mod.get('leadSponsor', {})
        if lead.get('name'):
            sponsors_list.append({
                'agency': lead.get('name'),
                'agency_class': lead.get('class'),
                'lead_or_collaborator': 'lead',
            })
        
        # Collaborators
        for collab in sponsors_mod.get('collaborators', []):
            sponsors_list.append({
                'agency': collab.get('name'),
                'agency_class': collab.get('class'),
                'lead_or_collaborator': 'collaborator',
            })
        
        return sponsors_list
    except:
        return []


def extract_locations(study_json: dict) -> List[dict]:
    """Extract locations from JSON"""
    try:
        contacts_mod = study_json.get('protocolSection', {}).get('contactsLocationsModule', {})
        locations = contacts_mod.get('locations', [])
        
        result = []
        for loc in locations:
            result.append({
                'facility': loc.get('facility', ''),
                'city': loc.get('city', ''),
                'state': loc.get('state', ''),
                'country': loc.get('country', ''),
            })
        return result
    except:
        return []


def extract_study_design(study_json: dict) -> dict:
    """Extract study design from JSON"""
    try:
        design_mod = study_json.get('protocolSection', {}).get('designModule', {})
        design_info = design_mod.get('designInfo', {})
        
        return {
            'allocation': design_info.get('allocation'),
            'intervention_model': design_info.get('interventionModel'),
            'masking': design_info.get('maskingInfo', {}).get('masking'),
            'primary_purpose': design_info.get('primaryPurpose'),
            'observational_model': design_info.get('observationalModel'),
            'time_perspective': design_info.get('timeframeOfObservation'),
        }
    except:
        return {}


def populate_from_json(session, json_file: str):
    """Populate database from JSON file"""
    studies_json = load_studies_from_json(json_file)
    
    if not studies_json:
        print(f"No studies found in {json_file}")
        return
    
    print(f"\nFound {len(studies_json)} studies in JSON file")
    print("=" * 70)
    
    loaded_count = 0
    errors = []
    
    for idx, study_json in enumerate(studies_json):
        try:
            # Extract all data
            study_info = extract_study_info(study_json)
            
            if not study_info.get('nct_id'):
                errors.append(f"Study {idx}: Missing NCT ID")
                continue
            
            # Validate with Pydantic
            study_data = StudySchema(**study_info)
            study_obj = Study(**study_data.model_dump())
            session.add(study_obj)
            session.flush()  # Get the study_id
            
            # Add conditions
            conditions_data = extract_conditions(study_json)
            for cond in conditions_data:
                try:
                    cond_schema = ConditionSchema(study_id=study_obj.study_id, **cond)
                    condition_obj = Condition(**cond_schema.model_dump())
                    session.add(condition_obj)
                except:
                    pass
            
            # Add interventions
            interventions_data = extract_interventions(study_json)
            for interv in interventions_data:
                try:
                    interv_schema = InterventionSchema(study_id=study_obj.study_id, **interv)
                    intervention_obj = Intervention(**interv_schema.model_dump())
                    session.add(intervention_obj)
                except:
                    pass
            
            # Add outcomes
            outcomes_data = extract_outcomes(study_json)
            for outcome in outcomes_data:
                try:
                    outcome_schema = OutcomeSchema(study_id=study_obj.study_id, **outcome)
                    outcome_obj = Outcome(**outcome_schema.model_dump())
                    session.add(outcome_obj)
                except:
                    pass
            
            # Add sponsors
            sponsors_data = extract_sponsors(study_json)
            for sponsor in sponsors_data:
                try:
                    sponsor_schema = SponsorSchema(study_id=study_obj.study_id, **sponsor)
                    sponsor_obj = Sponsor(**sponsor_schema.model_dump())
                    session.add(sponsor_obj)
                except:
                    pass
            
            # Add locations
            locations_data = extract_locations(study_json)
            for loc in locations_data:
                try:
                    loc_schema = LocationSchema(study_id=study_obj.study_id, **loc)
                    location_obj = Location(**loc_schema.model_dump())
                    session.add(location_obj)
                except:
                    pass
            
            # Add study design
            design_data = extract_study_design(study_json)
            if any(design_data.values()):
                try:
                    design_schema = StudyDesignSchema(study_id=study_obj.study_id, **design_data)
                    design_obj = StudyDesign(**design_schema.model_dump())
                    session.add(design_obj)
                except:
                    pass
            
            loaded_count += 1
            print(f"✓ Loaded: {study_info.get('nct_id')} - {study_info.get('title')[:60]}")
            
        except Exception as e:
            errors.append(f"Study {idx}: {str(e)}")
    
    session.commit()
    
    print("\n" + "=" * 70)
    print(f"✓ Successfully loaded {loaded_count} studies")
    if errors:
        print(f"⚠ Errors encountered: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error}")
    
    # Print data availability report
    print_data_availability_report()


def print_data_availability_report():
    """Print a report on available vs missing data fields"""
    print("\n" + "=" * 70)
    print("DATA AVAILABILITY REPORT")
    print("=" * 70)
    
    available_fields = {
        "Studies Table": [
            ("NCT ID", "✓ Always available"),
            ("Title", "✓ Usually available"),
            ("Acronym", "◐ Sometimes available"),
            ("Status", "✓ Usually available"),
            ("Phase", "◐ Sometimes available (null for observational)"),
            ("Study Type", "✓ Usually available"),
            ("Start Date", "✓ Usually available"),
            ("Completion Date", "◐ Sometimes available (ongoing studies)"),
            ("Primary Completion Date", "◐ Sometimes available"),
            ("Enrollment", "✓ Usually available"),
            ("Enrollment Type", "◐ Sometimes available"),
            ("Brief Summary", "✓ Usually available"),
            ("Eligibility Criteria", "✓ Usually available"),
            ("Gender", "✓ Usually available"),
            ("Min/Max Age", "✓ Usually available"),
        ],
        "Conditions Table": [
            ("Condition Name", "✓ Usually available"),
            ("MeSH Term", "✗ NOT available in JSON"),
        ],
        "Interventions Table": [
            ("Intervention Type", "✓ Usually available"),
            ("Name", "✓ Usually available"),
            ("Description", "✓ Usually available"),
        ],
        "Outcomes Table": [
            ("Outcome Type", "✓ Available (Primary/Secondary)"),
            ("Measure", "✓ Usually available"),
            ("Time Frame", "✓ Usually available"),
            ("Description", "✓ Usually available"),
        ],
        "Sponsors Table": [
            ("Agency Name", "✓ Usually available"),
            ("Agency Class", "✓ Usually available"),
            ("Lead/Collaborator", "✓ Available"),
        ],
        "Locations Table": [
            ("Facility", "✓ Usually available"),
            ("City", "✓ Usually available"),
            ("State", "◐ Sometimes available"),
            ("Country", "✓ Usually available"),
            ("Continent", "✗ NOT available in JSON (can be derived)"),
        ],
        "Study Design Table": [
            ("Allocation", "✓ Usually available"),
            ("Intervention Model", "✓ Usually available"),
            ("Masking", "✓ Usually available"),
            ("Primary Purpose", "✓ Usually available"),
            ("Observational Model", "◐ Sometimes available"),
            ("Time Perspective", "◐ Sometimes available"),
        ],
    }
    
    for table, fields in available_fields.items():
        print(f"\n{table}:")
        for field, availability in fields:
            print(f"  {availability:40} {field}")
    
    print("\n" + "=" * 70)
    print("LEGEND:")
    print("  ✓ = Reliably available in JSON")
    print("  ◐ = Sometimes available (may be null/empty)")
    print("  ✗ = Not available in JSON (needs external source)")
    print("=" * 70)


def main():
    """Main entry point"""
    print("Clinical Trials Database Generator")
    print("=" * 70)
    
    import sys
    
    # Check for JSON file argument
    if len(sys.argv) > 1 and sys.argv[1] in ("api", "--api"):
        # fetch from remote API and then populate
        json_file = "studies_api.json"
        max_pages = None
        if len(sys.argv) > 2:
            try:
                max_pages = int(sys.argv[2])
            except:
                max_pages = None
        print(f"Fetching studies from ClinicalTrials.gov API into {json_file} (max_pages={max_pages})")
        fetch_studies_from_api(json_file, page_size=100, max_pages=max_pages)
        print(f"\nLoading from JSON file: {json_file}")
        engine, Session = create_engine_and_session()
        
        if not engine or not Session:
            print("Failed to create database engine")
            return
        
        session = Session()
        
        try:
            print("Creating database schema...")
            create_schema(engine)
            print(f"\nLoading from JSON file: {json_file}")
            populate_from_json(session, json_file)
            print("\n✓ Database population from JSON completed successfully!")
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
        finally:
            session.close()
    else:
        # Use faker-generated data
        engine, Session = create_engine_and_session()
        
        if not engine or not Session:
            print("Failed to create database engine")
            return
        
        session = Session()
        
        try:
            print("Creating database schema...")
            create_schema(engine)
            
            print(f"Generating and populating sample studies...")
            studies = generate_studies(session, 50)
            
            generate_conditions(session, studies)
            generate_interventions(session, studies)
            generate_outcomes(session, studies)
            generate_sponsors(session, studies)
            generate_locations(session, studies)
            generate_study_design(session, studies)
            
            print("\n✓ Database population with sample data completed successfully!")
            print("\nTip: To load from JSON file, run: python main.py <path-to-json-file>")
            print("Tip: To fetch from ClinicalTrials.gov API and populate, run: python main.py api [max_pages]")
        except Exception as e:
            print(f"Error: {e}")
            session.rollback()
        finally:
            session.close()


if __name__ == "__main__":
    main()

