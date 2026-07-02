from core.db_connection import get_connection
from core.version_guard import validate_version
from core.app_config import APP_NAME


def validate_startup():
    """
    Validate application startup conditions.

    This function ensures that:
    - A connection to the internal database is available.
    - The application version is valid and supported.

    It queries the latest active version record for the application
    and validates it using the version guard.

    Returns
    -------
    tuple of (bool, str or None, str or None)
        A tuple containing:
        - is_valid : bool
            Indicates whether the startup is valid.
        - message : str or None
            Error message to display if validation fails.
        - download_url : str or None
            URL to download the latest version if applicable.

    Notes
    -----
    Validation flow:

    1. Attempt to connect to the internal database.
    2. If connection fails:
       - Return error indicating network/VPN requirement.
    3. If connection succeeds:
       - Query latest active version record.
       - Validate version using `validate_version`.
    4. If version validation fails:
       - Return validation message and optional download URL.
    5. If everything is valid:
       - Return success.

    Raises
    ------
    None
        All exceptions are handled internally and converted
        into user-facing error messages.
    """
    conn = get_connection()

    if not conn:
        return False, (
            "Unable to connect to the internal database.\n"
            "Please connect to the internal network or VPN and try again."
        ), None

    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT TOP 1
                app_name,
                version_number,
                minimum_supported_version,
                download_url,
                is_active,
                message
            FROM dbo.AppVersionSPS
            WHERE app_name = ?
              AND is_active = 1
            ORDER BY created_at DESC
        """, APP_NAME)

        row = cursor.fetchone()

        valid, message, download_url = validate_version(row)

        if not valid:
            return False, message, download_url

        return True, None, None

    except Exception:
        return False, (
            "Unable to validate application version.\n"
            "Please connect to the internal Sensata network or VPN."
        ), None

    finally:
        if conn:
            conn.close()