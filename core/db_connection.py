import pyodbc
from core.app_config import DB_CONFIG


def get_connection():
    """
    Create a connection to the database.

    This function builds the connection string using the configuration
    defined in the application settings and attempts to establish a
    connection using pyodbc.

    Returns
    -------
    pyodbc.Connection or None
        Active database connection if successful, otherwise None.

    Notes
    -----
    - Uses encrypted connection.
    - TrustServerCertificate is enabled.
    - Timeout is set to 5 seconds.

    Raises
    ------
    None
        All exceptions are handled internally and return None.
    """
    try:
        conn_str = (
            f"DRIVER={{{DB_CONFIG['driver']}}};"
            f"SERVER={DB_CONFIG['server']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['username']};"
            f"PWD={DB_CONFIG['password']};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )

        connection = pyodbc.connect(conn_str, timeout=5)
        return connection

    except Exception:
        return None