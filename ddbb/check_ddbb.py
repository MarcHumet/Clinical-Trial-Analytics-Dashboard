import os
from dotenv import load_dotenv
import pymysql
from loguru import logger

# Load .env
load_dotenv()

# MySQL connection settings from environment
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
MYSQL_USER = os.getenv('MYSQL_USER', 'user')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'pass')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'clinicaltrials')

# Connect to MySQL
conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor,
)
cursor = conn.cursor()

# Get all tables in the current database
cursor.execute("SHOW TABLES")
# pymysql with DictCursor returns dict rows; fetch and normalize
raw_tables = cursor.fetchall()
tables = []
for row in raw_tables:
    # SHOW TABLES returns a dict with key like 'Tables_in_<db>'
    val = list(row.values())[0]
    tables.append((val,))

print("=" * 70)
print("DATABASE VERIFICATION REPORT")
print("=" * 70)
print(f"\nTables found: {len(tables)}\n")

for table in tables:
    table_name = table[0]
    cursor.execute(f"SELECT COUNT(*) AS cnt FROM `{table_name}`")
    count = cursor.fetchone()['cnt']
    # Get column info from information_schema
    cursor.execute(
        "SELECT COLUMN_NAME, COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
        (MYSQL_DATABASE, table_name),
    )
    columns = cursor.fetchall()
    
    print(f"\n✓ {table_name.upper()}")
    print(f"  Rows: {count}")
    print(f"  Columns: {len(columns)}")
    for col in columns:
        print(f"    - {col['COLUMN_NAME']}: {col['COLUMN_TYPE']}")

# Show sample data
print("\n" + "=" * 70)
print("SAMPLE DATA")
print("=" * 70)

print("\n✓ STUDIES (first 3 records):")
cursor.execute("SELECT study_id, nct_id, title, status, phase FROM studies LIMIT 3")
for row in cursor.fetchall():
    title = (row['title'] or '')[:50]
    print(f"  {row['nct_id']}: {title}... | Status: {row.get('status')} | Phase: {row.get('phase')}")

print("\n✓ CONDITIONS (sample):")
cursor.execute("SELECT c.condition_name, COUNT(*) as cnt FROM conditions c GROUP BY c.condition_name LIMIT 5")
for row in cursor.fetchall():
    print(f"  {row['condition_name']}: {row['cnt']} occurrences")

print("\n✓ INTERVENTIONS (sample):")
cursor.execute("SELECT intervention_type, COUNT(*) as cnt FROM interventions GROUP BY intervention_type")
for row in cursor.fetchall():
    print(f"  {row.get('intervention_type')}: {row['cnt']} records")

print("\n✓ OUTCOMES (sample):")
cursor.execute("SELECT outcome_type, COUNT(*) as cnt FROM outcomes GROUP BY outcome_type")
for row in cursor.fetchall():
    print(f"  {row.get('outcome_type')}: {row['cnt']} records")

print("\n✓ SPONSORS (sample):")
cursor.execute("SELECT agency_class, COUNT(*) as cnt FROM sponsors WHERE agency_class IS NOT NULL GROUP BY agency_class")
for row in cursor.fetchall():
    print(f"  {row.get('agency_class')}: {row['cnt']} records")

print("\n✓ LOCATIONS (by country):")
cursor.execute("SELECT country, COUNT(*) as cnt FROM locations WHERE country IS NOT NULL GROUP BY country LIMIT 5")
for row in cursor.fetchall():
    print(f"  {row.get('country')}: {row['cnt']} locations")

print("\n✓ INDEXES created:")
cursor.execute(
    "SELECT DISTINCT INDEX_NAME FROM INFORMATION_SCHEMA.STATISTICS WHERE TABLE_SCHEMA=%s",
    (MYSQL_DATABASE,)
)
for idx in cursor.fetchall():
    print(f"  - {idx['INDEX_NAME']}")

print("\n" + "=" * 70)

conn.close()
logger.success("✓ Database verification COMPLETE - Connection closed and checks finished")
print("=" * 70)