import os


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:12345@localhost:5432/autoprepai",
)

# To create the database, run the following command in your terminal:
# psql -U postgres -h localhost -c "CREATE DATABASE autoprepai;" 
# pass = 12345 , change it if you have a different password for your postgres user