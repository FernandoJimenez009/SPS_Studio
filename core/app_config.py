APP_NAME = "SPS Studio"
APP_VERSION = "2.21"
APP_SUBTITLE = "Structured Problem-Solving Studio"
CREATOR = "Fernando de Jesus Jimenez Solis"
CONTACT = "fjimenezsolis@sensata.com"

APP_ID = f"SPS.Studio.{APP_VERSION}"

WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION} - {APP_SUBTITLE}"
STARTUP_ERROR_TITLE = f"{APP_NAME} - Startup Error"

# ⚠️ OPCIÓN 1 (rápida - menos segura)
DB_CONFIG = {
    "driver": "ODBC Driver 17 for SQL Server",
    "server": "sagpdbsql02",
    "database": "PDE_management",
    "username": "PDEuser",
    "password": "Nj2nns5fmTv8",
    "encrypt": "yes",
    "trust_server_certificate": "yes",
}