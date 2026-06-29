import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:12345@localhost:5432/autoprepai",
)
# To create the database, run the following command in your terminal:
# psql -U postgres -h localhost -c "CREATE DATABASE autoprepai;" 
# pass = 12345 , change it if you have a different password for your postgres user


B2_KEY_ID = "0035909f3e49f7f0000000001"
B2_APPLICATION_KEY = "K003B3ulhBhTHUadjVuL9JuApAZuowc"
B2_BUCKET_NAME = "autoprepai-datasets"
B2_ENDPOINT_URL = "https://s3.eu-central-003.backblazeb2.com"  # EU Central region
