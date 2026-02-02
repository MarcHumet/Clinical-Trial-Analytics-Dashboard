import os
import json
import gzip
import pymysql
from loguru import logger
import requests
from tqdm import tqdm
from typing import Optional

DB_HOST = os.getenv("DB_HOST") or os.getenv("MYSQL_HOST") or "mysql"
DB_PORT = int(os.getenv("DB_PORT") or os.getenv("MYSQL_PORT") or 3306)
DB_USER = os.getenv("DB_USER") or os.getenv("MYSQL_USER") or "user"
DB_PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("MYSQL_PASSWORD") or "pass"
DB_NAME = os.getenv("DB_NAME") or os.getenv("MYSQL_DATABASE") or "clinicaltrials"

# If running on the host (not inside Docker), the service name "mysql" won't resolve.
if DB_HOST == "mysql" and not os.path.exists("/.dockerenv"):
    DB_HOST = "127.0.0.1"


logger.info("starting search_condition_name listing")
conn = pymysql.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    charset="utf8mb4",
)
try:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT condition_name FROM conditions;")
        rows = cur.fetchall()
        conditions = [r[0] for r in rows if r[0]]
        for name in conditions:
            logger.info(name)
finally:
    conn.close()


def get_mesh_term(condition_name: str) -> Optional[str]:
    """Fetch MeSH descriptor ID for a condition name."""
    if not condition_name:
        return None

    url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
    params = {
        "label": condition_name,
        "match": "contains",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json() or []
        if not results:
            return None
        resource = results[0].get("resource")
        if not resource:
            return None
        return resource.rsplit("/", 1)[-1]
    except Exception as e:
        logger.error(f"error:{e}")
        return None
dict_cond_name_to_mesh_term = {}
for condition in tqdm(conditions, desc="Fetching MeSH terms"):
    mesh_term = get_mesh_term(condition)
    if not mesh_term:
        dict_cond_name_to_mesh_term[condition] = None
        logger.warning(f"No MeSH term found for condition: {condition}")
        continue
    else:
        dict_cond_name_to_mesh_term[condition] = mesh_term
        logger.info(f"Condition: {condition} -> MeSH: {mesh_term}")
    # try:
    #     conn = pymysql.connect(
    #         host=DB_HOST,
    #         port=DB_PORT,
    #         user=DB_USER,
    #         password=DB_PASSWORD,
    #         database=DB_NAME,
    #         charset="utf8mb4",
    #     )
    #     with conn.cursor() as cur:
    #         sql = """
    #             UPDATE conditions
    #             SET mesh_term = %s
    #             WHERE condition_name = %s
    #         """
    #         cur.execute(sql, (mesh_term, condition))
    #     conn.commit()
    # except Exception as e:
    #     logger.error(f"error updating DB for condition {condition}: {e}")
    # finally:
    #     conn.close()

# Save dictionary in a compact gzip-compressed JSON file
output_path = "conditions_mesh_terms.json.gz"
with gzip.open(output_path, "wt", encoding="utf-8") as f:
    json.dump(dict_cond_name_to_mesh_term, f, ensure_ascii=False, separators=(",", ":"))
logger.info(f"Saved {len(dict_cond_name_to_mesh_term)} items to {output_path}")

# # Example: load it back
# with gzip.open(output_path, "rt", encoding="utf-8") as f:
#     loaded_dict = json.load(f)
# logger.info(f"Reloaded {len(loaded_dict)} items from {output_path}")
