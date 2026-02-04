"""
Study Enrollment Success Calculation

Multiple approaches to calculate enrollment success metrics:
1. Enrollment Rate: Participants per month
2. Enrollment Velocity: Progress over study duration
3. Success Tier: Categorize as Successful/At-Risk/Failed
4. Percentile Ranking: Compared to peer studies
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pathlib import Path


def calculate_enrollment_rate(row):
    """
    Calculate enrollment rate: participants per month
    
    Formula: Enrollment / Trial Duration (months)
    Interpretation:
    - > 50/month = Excellent
    - 10-50/month = Good
    - 1-10/month = Adequate
    - < 1/month = Slow
    """
    enrollment = row['enrollment']
    start_date = row['start_date']
    completion_date = row['completion_date']
    
    if not enrollment or enrollment == 0:
        return None
    
    if not start_date or not completion_date:
        return None
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(completion_date)
    duration_days = (end - start).days
    duration_months = max(1, duration_days / 30.44)  # Account for month length
    
    rate = enrollment / duration_months
    return rate


def get_enrollment_rate_tier(rate):
    """Categorize enrollment rate"""
    if rate is None:
        return 'Unknown'
    elif rate > 50:
        return 'Excellent (>50/month)'
    elif rate >= 10:
        return 'Good (10-50/month)'
    elif rate >= 1:
        return 'Adequate (1-10/month)'
    else:
        return 'Slow (<1/month)'


def calculate_enrollment_success_pct(row, expected_monthly_rate=10):
    """
    Calculate enrollment success as percentage of expected
    
    Assumption: Typical clinical trial enrolls ~10 participants/month
    This is industry benchmark (adjustable)
    
    Formula: (Actual Enrollment) / (Expected Based on Duration) × 100%
    Interpretation:
    - > 100% = Exceeded expectations
    - 75-100% = Met expectations
    - 50-75% = Below expectations
    - < 50% = Significantly below
    """
    enrollment = row['enrollment']
    start_date = row['start_date']
    completion_date = row['completion_date']
    
    if not enrollment or enrollment == 0:
        return 0
    
    if not start_date or not completion_date:
        return None
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(completion_date)
    duration_days = (end - start).days
    duration_months = max(1, duration_days / 30.44)
    
    expected_enrollment = duration_months * expected_monthly_rate
    success_pct = (enrollment / max(1, expected_enrollment)) * 100
    
    return min(success_pct, 200)  # Cap at 200% for visualization


def get_success_tier(pct):
    """Categorize enrollment success"""
    if pct is None:
        return 'Unknown'
    elif pct >= 100:
        return 'Exceeded (≥100%)'
    elif pct >= 75:
        return 'Met (75-100%)'
    elif pct >= 50:
        return 'Below (50-75%)'
    else:
        return 'Significantly Below (<50%)'


def get_status_success_score(row):
    """
    Assign success score based on study status
    
    COMPLETED = Study finished (highest success)
    ACTIVE_NOT_RECRUITING = Study closed to enrollment (good)
    RECRUITING = Still enrolling (moderate)
    NOT_YET_RECRUITING = Not started (low)
    TERMINATED/WITHDRAWN = Failed (lowest)
    """
    status = row['status']
    
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
    
    return status_scores.get(status, 50)


def calculate_composite_success_score(row):
    """
    Combine multiple factors into single success metric (0-100)
    
    Factors:
    - Enrollment completeness (40%): How many participants relative to typical
    - Status success (30%): Did study complete successfully
    - Temporal efficiency (20%): How fast enrollment happened
    - Data completeness (10%): How rich the study metadata is
    """
    score = 0
    
    # Factor 1: Enrollment Completeness (40 points)
    enrollment = row['enrollment'] or 0
    if enrollment > 0:
        # Typical study: 100-500 participants
        # Max bonus: 500+ participants
        enroll_score = min(40, (enrollment / 500) * 40)
        score += enroll_score
    
    # Factor 2: Status Success (30 points)
    status_score = get_status_success_score(row)
    score += (status_score / 100) * 30
    
    # Factor 3: Temporal Efficiency (20 points)
    start_date = row['start_date']
    completion_date = row['completion_date']
    
    if start_date and completion_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(completion_date)
        duration_years = (end - start).days / 365.25
        
        # Efficient: Completed in < 3 years with good enrollment
        if duration_years > 0:
            rate_per_year = enrollment / duration_years
            if rate_per_year > 500:  # > 500/year = excellent pace
                score += 20
            elif rate_per_year > 100:  # > 100/year = good pace
                score += 15
            elif rate_per_year > 20:  # > 20/year = adequate pace
                score += 10
            else:
                score += 5
    
    # Factor 4: Data Completeness (10 points)
    non_null_fields = 0
    for field in ['title', 'status', 'phase', 'enrollment', 'brief_summary', 'eligibility_criteria']:
        val = row.get(field)
        if val and (not isinstance(val, str) or len(val) > 0):
            non_null_fields += 1
    
    data_score = min(10, (non_null_fields / 6) * 10)
    score += data_score
    
    return round(score, 1)


def run_enrollment_success_analysis(engine=None, output_dir=None):
    """
    Run complete enrollment success analysis
    
    Args:
        engine: SQLAlchemy engine (if None, will create from environment)
        output_dir: Directory to save results (if None, will use src/results)
    
    Returns:
        DataFrame with enrollment success metrics
    """
    
    # Use provided engine or create new one
    if engine is None:
        # Configuration
        load_dotenv()
        mysql_url = os.getenv('MYSQL_DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URL')

        if not mysql_url:
            mysql_user = os.getenv('MYSQL_USER') or os.getenv('DB_USER') or 'user'
            mysql_password = os.getenv('MYSQL_PASSWORD') or os.getenv('DB_PASSWORD') or 'pass'
            # For non-Docker execution, try localhost first, then fall back to mysql service
            mysql_host = os.getenv('MYSQL_HOST') or os.getenv('DB_HOST') or 'localhost'
            mysql_port = os.getenv('MYSQL_PORT') or os.getenv('DB_PORT') or '3306'
            mysql_db = os.getenv('MYSQL_DATABASE') or os.getenv('DB_NAME') or 'clinicaltrials'

            mysql_url = (
                f"mysql+pymysql://{mysql_user}:{mysql_password}@"
                f"{mysql_host}:{mysql_port}/{mysql_db}"
            )

        print(f'Using DB URL: {mysql_url.split("@")[-1]}\n')
        engine = create_engine(mysql_url)

    # Load data
    with engine.connect() as conn:
        studies_df = pd.read_sql_query(text('SELECT * FROM studies'), conn)

    if studies_df.empty:
        print("Warning: No studies found in database.")
        return pd.DataFrame()

    # Apply calculations to all studies
    studies_df['enrollment_rate_monthly'] = studies_df.apply(calculate_enrollment_rate, axis=1)
    studies_df['enrollment_rate_tier'] = studies_df['enrollment_rate_monthly'].apply(get_enrollment_rate_tier)
    studies_df['enrollment_success_pct'] = studies_df.apply(
        lambda row: calculate_enrollment_success_pct(row, expected_monthly_rate=10), 
        axis=1
    )
    studies_df['success_tier'] = studies_df['enrollment_success_pct'].apply(get_success_tier)
    studies_df['status_success_score'] = studies_df.apply(get_status_success_score, axis=1)
    studies_df['composite_success_score'] = studies_df.apply(calculate_composite_success_score, axis=1)

    # Percentile ranking (higher is better)
    studies_df['success_percentile'] = studies_df['composite_success_score'].rank(pct=True) * 100

    # Print analysis results
    print_enrollment_analysis(studies_df)
    
    # Export results
    export_results(studies_df, output_dir)
    
    return studies_df


def print_enrollment_analysis(studies_df):
    """Print detailed enrollment analysis results"""
    
    print("=" * 100)
    print("STUDY ENROLLMENT SUCCESS METRICS - COMPLETE DATABASE ANALYSIS")
    print("=" * 100)

    # Summary statistics
    print("\n1. OVERALL METRICS SUMMARY")
    print("-" * 100)
    print(f"Total Studies: {len(studies_df)}")
    print(f"Studies with enrollment data: {studies_df['enrollment'].notna().sum()}")
    print(f"\nEnrollment Statistics:")
    print(f"  Mean enrollment: {studies_df['enrollment'].mean():,.0f}")
    print(f"  Median enrollment: {studies_df['enrollment'].median():,.0f}")
    print(f"  Std Dev: {studies_df['enrollment'].std():,.0f}")
    print(f"  Min: {studies_df['enrollment'].min():,.0f}")
    print(f"  Max: {studies_df['enrollment'].max():,.0f}")

    print(f"\nEnrollment Rate (monthly) Statistics:")
    valid_rates = studies_df['enrollment_rate_monthly'].dropna()
    if len(valid_rates) > 0:
        print(f"  Mean rate: {valid_rates.mean():.2f} participants/month")
        print(f"  Median rate: {valid_rates.median():.2f} participants/month")

    print(f"\nComposite Success Score Statistics:")
    print(f"  Mean: {studies_df['composite_success_score'].mean():.1f}/100")
    print(f"  Median: {studies_df['composite_success_score'].median():.1f}/100")
    print(f"  Std Dev: {studies_df['composite_success_score'].std():.1f}")

    # Success tier distribution
    print(f"\n2. ENROLLMENT SUCCESS TIERS")
    print("-" * 100)
    tier_counts = studies_df['success_tier'].value_counts()
    for tier, count in tier_counts.items():
        pct = (count / len(studies_df)) * 100
        print(f"  {tier:30} {count:6} studies ({pct:5.1f}%)")

    # Enrollment rate tier distribution
    print(f"\n3. ENROLLMENT RATE TIERS (Participants/Month)")
    print("-" * 100)
    rate_tier_counts = studies_df['enrollment_rate_tier'].value_counts()
    for tier, count in rate_tier_counts.items():
        pct = (count / len(studies_df)) * 100
        print(f"  {tier:40} {count:6} studies ({pct:5.1f}%)")

    # Top 10 studies
    print(f"\n4. TOP 10 STUDIES BY COMPOSITE SUCCESS SCORE")
    print("-" * 100)
    top_studies = studies_df.nlargest(10, 'composite_success_score')[
        ['nct_id', 'title', 'enrollment', 'status', 'composite_success_score', 'enrollment_rate_monthly', 'enrollment_success_pct']
    ]

    for idx, (_, study) in enumerate(top_studies.iterrows(), 1):
        print(f"\n{idx}. {study['nct_id']}")
        print(f"   Title: {study['title'][:70]}")
        print(f"   Enrollment: {study['enrollment']:,.0f}")
        print(f"   Status: {study['status']}")
        print(f"   Composite Score: {study['composite_success_score']:.1f}/100")
        if pd.notna(study['enrollment_rate_monthly']):
            print(f"   Monthly Rate: {study['enrollment_rate_monthly']:.2f} participants/month")
        if pd.notna(study['enrollment_success_pct']):
            print(f"   Success vs. Benchmark: {study['enrollment_success_pct']:.1f}%")

    # Target study analysis
    print(f"\n5. TARGET STUDY ANALYSIS (NCT00158574)")
    print("-" * 100)
    target = studies_df[studies_df['nct_id'] == 'NCT00158574']

    if len(target) > 0:
        target = target.iloc[0]
        print(f"  NCT ID: {target['nct_id']}")
        print(f"  Title: {target['title']}")
        print(f"  Status: {target['status']}")
        print(f"  Enrollment: {target['enrollment']:,.0f}")
        print(f"  Composite Success Score: {target['composite_success_score']:.1f}/100")
        if pd.notna(target['enrollment_rate_monthly']):
            print(f"  Enrollment Rate: {target['enrollment_rate_monthly']:.2f} participants/month ({target['enrollment_rate_tier']})")
        if pd.notna(target['enrollment_success_pct']):
            print(f"  Success vs. Benchmark (10/mo): {target['enrollment_success_pct']:.1f}% ({target['success_tier']})")
        print(f"  Percentile Ranking: {target['success_percentile']:.1f}th percentile")
        print(f"  Interpretation: Better than {target['success_percentile']:.1f}% of studies in database")
    else:
        print("  Target study not found in database")

    print("=" * 100)
    print(f"Analysis Complete | {len(studies_df)} studies analyzed")
    print("=" * 100)


def export_results(studies_df, output_dir=None):
    """Export results to CSV file"""
    
    print(f"\n6. EXPORTING DETAILED RESULTS")
    print("-" * 100)

    export_df = studies_df[[
        'nct_id', 'title', 'status', 'phase', 'enrollment', 'start_date', 'completion_date',
        'enrollment_rate_monthly', 'enrollment_rate_tier', 
        'enrollment_success_pct', 'success_tier',
        'status_success_score', 'composite_success_score', 'success_percentile'
    ]].copy()

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        results_dir = project_root / 'src' / 'results'
    else:
        results_dir = Path(output_dir)
    
    csv_path = results_dir / 'enrollment_success_metrics.csv'

    # Ensure the results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    export_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")

    # Show sample
    print(f"\n  Sample output (first 5 studies):")
    print(export_df.head(5).to_string(max_cols=8))


if __name__ == "__main__":
    # When run as script, execute the analysis and print interpretation guide
    run_enrollment_success_analysis()
    
    # Print interpretation guide
    print(f"\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)

    print("""
METRIC 1: ENROLLMENT RATE (participants/month)
  Formula: Total Enrollment / Duration (months)
  Tiers:
    • Excellent: >50/month (very rapid enrollment)
    • Good: 10-50/month (typical academic trial)
    • Adequate: 1-10/month (slow but viable)
    • Slow: <1/month (very slow)
  Use For: Identifying enrollment velocity

METRIC 2: ENROLLMENT SUCCESS % (vs. 10/month benchmark)
  Formula: (Actual Enrollment) / (Expected Enrollment) × 100%
  Assumption: Typical trial enrolls 10 participants/month
  Tiers:
    • Exceeded: ≥100% (outperformed expectations)
    • Met: 75-100% (met expectations)
    • Below: 50-75% (underperformed)
    • Significantly Below: <50% (major shortfall)
  Use For: Comparing actual vs. expected performance

METRIC 3: STATUS SUCCESS SCORE (0-100)
  Weights:
    • COMPLETED: 100 (study finished successfully)
    • ACTIVE_NOT_RECRUITING: 85 (enrollment closed, study ongoing)
    • RECRUITING: 60 (still enrolling)
    • NOT_YET_RECRUITING: 30 (not started)
    • TERMINATED/WITHDRAWN: 5-10 (failed)
  Use For: Overall study viability

METRIC 4: COMPOSITE SUCCESS SCORE (0-100) ⭐ RECOMMENDED
  Combines:
    • Enrollment completeness (40%): Did they reach target volumes?
    • Status success (30%): Did study complete?
    • Temporal efficiency (20%): How fast was enrollment?
    • Data completeness (10%): Is metadata rich?
  Interpretation:
    • 80-100: Highly successful
    • 60-79: Successful
    • 40-59: Moderate success
    • 20-39: Below average
    • 0-19: Struggled significantly
  Use For: Single ranking metric across all studies

PERCENTILE RANKING:
  • 95th percentile = Better than 95% of studies
  • 50th percentile = Median performance
  • 5th percentile = Worse than 95% of studies
""")