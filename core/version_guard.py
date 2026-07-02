from core.app_config import APP_VERSION


def _version_to_tuple(version):
    """
    Convert a version string into a comparable tuple.

    Parameters
    ----------
    version : str or None
        Version string in the format "X.Y.Z".

    Returns
    -------
    tuple of int
        Tuple representation of the version (major, minor, patch).

    Notes
    -----
    - If the version is invalid or None, returns (0, 0, 0).
    - This allows safe comparison between versions.
    """
    try:
        if not version:
            return (0, 0, 0)

        return tuple(map(int, str(version).strip().split(".")))

    except Exception:
        return (0, 0, 0)


def validate_version(row):
    """
    Validate the application version against database configuration.

    This function compares the locally installed version against the
    latest version and minimum supported version retrieved from the database.

    Parameters
    ----------
    row : object or None
        Database row containing version configuration with attributes:
        - version_number
        - minimum_supported_version
        - is_active
        - download_url
        - message

    Returns
    -------
    tuple of (bool, str, str or None)
        A tuple containing:
        - is_valid : bool
            Indicates whether the current version is allowed to run.
        - message : str
            Informational or error message for the user.
        - download_url : str or None
            URL to download the latest version if required.

    Notes
    -----
    Validation rules:

    1. If no configuration is found → block execution.
    2. If application is inactive → block execution.
    3. If version < minimum supported → force update.
    4. If version != latest → recommend update.
    5. If version is valid → allow execution.

    Version comparison is performed using tuple conversion.

    Examples
    --------
    >>> _version_to_tuple("1.2.3")
    (1, 2, 3)
    >>> _version_to_tuple(None)
    (0, 0, 0)
    """
    if not row:
        return False, "Application configuration not found.", None

    db_version = row.version_number
    minimum_version = row.minimum_supported_version
    is_active = row.is_active
    download_url = row.download_url
    db_message = row.message

    if not is_active:
        return False, "Application is currently disabled.", None

    local_v = _version_to_tuple(APP_VERSION)
    db_v = _version_to_tuple(db_version)
    min_v = _version_to_tuple(minimum_version)

    # Version too old (blocked)
    if local_v < min_v:
        return False, (
            "This version of SPS Studio is no longer supported.\n"
            "You must update to continue.\n"
            f"[Download Latest Version]({download_url})"
        ), download_url

    # New version available (warning / soft block depending on UX)
    if local_v != db_v:
        return False, (
            "A newer version of SPS Studio is available.\n"
            f"Installed version: {APP_VERSION}\n"
            f"Latest version: {db_version}\n"
            "Please download the latest version:\n"
            f"[Download]({download_url})"
        ), download_url

    # Valid version
    return True, db_message, None