import os
import unicodedata
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from loguru import logger

# Load .env (if present)
load_dotenv()

DB_HOST = os.getenv("DB_HOST") or os.getenv("MYSQL_HOST") or "mysql"
DB_PORT = int(os.getenv("DB_PORT") or os.getenv("MYSQL_PORT") or 3306)
DB_USER = os.getenv("DB_USER") or os.getenv("MYSQL_USER") or "user"
DB_PASSWORD = os.getenv("DB_PASSWORD") or os.getenv("MYSQL_PASSWORD") or "pass"
DB_NAME = os.getenv("DB_NAME") or os.getenv("MYSQL_DATABASE") or "clinicaltrials"

# Manual overrides for common country name variants
COUNTRY_OVERRIDES = {
    "United States": "US",
    "United States of America": "US",
    "Russia": "RU",
    "South Korea": "KR",
    "North Korea": "KP",
    "Ivory Coast": "CI",
    "Côte d'Ivoire": "CI",
    "Cote d'Ivoire": "CI",
    "Cote dIvoire": "CI",
    "Cote d Ivoire": "CI",
    "Venezuela": "VE",
    "Bolivia": "BO",
    "Tanzania": "TZ",
    "Syria": "SY",
    "Laos": "LA",
    "Moldova": "MD",
    "Czechia": "CZ",
    "Czech Republic": "CZ",
    "Iran": "IR",
    "Vietnam": "VN",
    "Brunei": "BN",
    "Palestine": "PS",
    "Palestinian Territories": "PS",
    "Taiwan": "TW",
    "Hong Kong": "HK",
    "Macau": "MO",
    "Congo": "CG",
    "Democratic Republic of the Congo": "CD",
    "Serbia and Montenegro": "RS",
    "North Macedonia": "MK",
    "The Gambia": "GM",
    "Reunion": "RE",
    "Guadeloupe": "GP",
    "French Polynesia": "PF",
    "New Caledonia": "NC",
    "Puerto Rico": "PR",
    "Turkey": "TR",
    "Turkiye": "TR",
    "Türkiye": "TR",
}

CONTINENT_NAMES = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
}


def _normalize_country_text(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    # Remove parenthetical notes like "Turkey (Türkiye)"
    if "(" in value and ")" in value:
        value = value.split("(")[0].strip()
    # Normalize unicode (fix mis-encoded characters)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    return value


def normalize_country_to_alpha2(country_name: str) -> str | None:
    if not country_name:
        return None
    country_name = _normalize_country_text(country_name)
    if not country_name:
        return None

    if country_name in COUNTRY_OVERRIDES:
        return COUNTRY_OVERRIDES[country_name]

    try:
        import pycountry

        country = pycountry.countries.lookup(country_name)
        return country.alpha_2
    except Exception as e:
        logger.error(f"error:{e}")
        return None


def get_continent(country_name: str) -> str | None:
    alpha2 = normalize_country_to_alpha2(country_name)
    if not alpha2:
        return None

    try:
        import pycountry_convert

        continent_code = pycountry_convert.country_alpha2_to_continent_code(alpha2)
        return CONTINENT_NAMES.get(continent_code)
    except Exception as e:
        logger.error(f"error:{e}")
        return None


def update_continents() -> dict:
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )
    except Error as e:
        print(f"Error connecting to database: {e}")
        return {"updated": 0, "skipped": 0, "error": str(e)}

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT country FROM locations WHERE country IS NOT NULL")
        countries = [row[0] for row in cursor.fetchall()]

        updated = 0
        skipped = 0
        unmapped = []

        for country in countries:
            continent = get_continent(country)
            if not continent:
                skipped += 1
                unmapped.append(country)
                continue
            cursor.execute(
                "UPDATE locations SET continent = %s WHERE country = %s",
                (continent, country),
            )
            updated += cursor.rowcount

        conn.commit()
        print(f"Updated rows: {updated}")
        print(f"Skipped countries (no mapping): {skipped}")
        if unmapped:
            print("Unmapped countries:")
            for c in sorted(set(unmapped)):
                print(f"  - {c}")
        return {"updated": updated, "skipped": skipped, "unmapped": sorted(set(unmapped))}

    except Error as e:
        print(f"Error updating continents: {e}")
        return {"updated": 0, "skipped": 0, "error": str(e)}
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    update_continents()
