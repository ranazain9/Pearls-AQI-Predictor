"""Feature pipeline: fetch raw data, engineer features, persist locally and to Hopsworks."""
import logging
import os
import sys
from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FEATURES_CSV,
    HISTORICAL_CSV,
)
from src.features.engineering import clean_features_df, compute_features
from src.features.fetch_raw import (
    current_reading_to_row,
    fetch_aqicn_current,
    fetch_openmeteo_current_row,
)

load_dotenv()


def run_feature_pipeline(upload_to_hopsworks: bool = False) -> pd.DataFrame:
    """
    Run the hourly feature pipeline:
    1. Fetch current Open-Meteo weather + AQ (same provider as training)
    2. Optionally enrich with AQICN ground-truth PM2.5
    3. Append to historical dataset and compute features
    4. Remove NaNs and sanitize feature matrix
    5. Optionally upload to Hopsworks Feature Store
    """
    Path("data").mkdir(exist_ok=True)

    aqicn_data = None
    try:
        aqicn_data = fetch_aqicn_current()
    except Exception:
        pass

    meteo_row = fetch_openmeteo_current_row()
    new_row = current_reading_to_row(aqicn_data, meteo_row)

    if HISTORICAL_CSV.exists():
        history = pd.read_csv(HISTORICAL_CSV, parse_dates=["timestamp"])
        combined = pd.concat([history, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    else:
        combined = new_row

    featured = clean_features_df(combined)
    featured.to_csv(FEATURES_CSV, index=False)
    featured.to_csv(HISTORICAL_CSV, index=False)

    if upload_to_hopsworks:
            _upload_to_hopsworks(featured)

    return featured


logger = logging.getLogger(__name__)

_HOPSWORKS_UPLOAD_MAX_ATTEMPTS = 5
_HOPSWORKS_UPLOAD_BASE_DELAY_SECONDS = 5
_TRANSIENT_HOPSWORKS_TRANSPORT_SIGNATURES = (
    "generic hdfsobjectstore error",
    "io error occurred while communicating with hdfs",
    "rpc listener disconnected",
)
_NATIVE_HOPSWORKS_TRANSPORT_SIGNATURES = (
    "libgssapi_krb5",
    "loading kerberos libraries are not supported",
    *_TRANSIENT_HOPSWORKS_TRANSPORT_SIGNATURES,
)


def _prepare_hopsworks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return the clean, ordered upload frame with Hopsworks timestamp precision."""
    clean = clean_features_df(df)
    cols_to_upload = ["timestamp", TARGET_COLUMN] + [
        column for column in FEATURE_COLUMNS if column in clean.columns
    ]
    clean = clean[cols_to_upload].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).astype("datetime64[us]")

    bigint_cols = ["aqi", "week_of_year"]
    for col in bigint_cols:
        if col in clean.columns:
            clean[col] = clean[col].round().astype("int64")

    int32_cols = ["hour", "day", "day_of_week", "month"]
    for col in int32_cols:
        if col in clean.columns:
            clean[col] = clean[col].round().astype("int32")

    return clean


def _exception_chain_messages(error: BaseException) -> str:
    """Return normalized messages from an exception and its causal chain."""
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__

    return "\n".join(messages)


def _is_transient_hopsworks_transport_error(error: BaseException) -> bool:
    """Identify a known temporary Hopsworks Delta/HDFS transport failure."""
    combined_message = _exception_chain_messages(error)
    return any(
        signature in combined_message
        for signature in _TRANSIENT_HOPSWORKS_TRANSPORT_SIGNATURES
    )


def _is_native_hopsworks_transport_error(error: BaseException) -> bool:
    """Identify the unsupported native Windows Delta/HDFS transport failure."""
    if os.name != "nt":
        return False

    combined_message = _exception_chain_messages(error)
    return any(
        signature in combined_message
        for signature in _NATIVE_HOPSWORKS_TRANSPORT_SIGNATURES
    )


def _close_hopsworks_project(project: object | None) -> None:
    """Close a Hopsworks project without masking an upload result or error."""
    if project is None:
        return

    close = getattr(project, "close", None)
    if not callable(close):
        return

    try:
        close()
    except Exception:
        logger.warning("Failed to close the Hopsworks project cleanly.", exc_info=True)


def _uses_windows_hopsworks_transport() -> bool:
    """Return True when local Delta/HDFS writes are unavailable."""
    return sys.platform == "win32"


def _feature_group_description() -> str:
    return (
        "Lahore AQI features — 17 clean weather/AQ features and "
        "Open-Meteo us_aqi target."
    )


def _resolve_feature_group(fs: object):
    """Return the persisted feature group, creating metadata if needed."""
    try:
        return fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    except Exception:
        return fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            primary_key=["timestamp"],
            description=_feature_group_description(),
            event_time="timestamp",
            online_enabled=True,
        )


def _windows_kafka_write_options() -> dict[str, object]:
    return {
        "internal_kafka": False,
        "wait_for_online_ingestion": True,
        "online_ingestion_options": {"timeout": 600},
    }


def _windows_server_job_write_options() -> dict[str, object]:
    return {
        "wait_for_job": True,
        "internal_kafka": False,
    }


def _upload_via_server_ingestion_job(fg: object, clean: pd.DataFrame) -> None:
    """Upload through Hopsworks REST + cluster ingestion job (no local HDFS)."""
    from hsfs.engine.python import Engine

    write_options = _windows_server_job_write_options()
    Engine()._legacy_save_dataframe(
        fg,
        clean,
        "upsert",
        fg.online_enabled,
        None,
        write_options.copy(),
        write_options.copy(),
    )


def _upload_via_online_kafka(fg: object, clean: pd.DataFrame) -> None:
    """Stream rows over Kafka and materialize offline storage on Hopsworks."""
    fg.insert(
        clean,
        storage="online",
        write_options=_windows_kafka_write_options(),
        wait=True,
    )
    try:
        materialization_job = fg.materialization_job
        logger.info("Starting server-side offline materialization job.")
        materialization_job.run(await_termination=True)
    except Exception:
        logger.info("No materialization job configured. Skipping.")


def _upload_on_windows(fg: object, clean: pd.DataFrame) -> None:
    """Upload without using the unsupported native Windows Delta/HDFS client."""
    if fg.online_enabled:
        logger.info(
            "Windows detected: uploading via Kafka (skipping local HDFS/Delta)."
        )
        _upload_via_online_kafka(fg, clean)
        return

    logger.info(
        "Windows detected: uploading via server-side ingestion job "
        "(skipping local HDFS/Delta)."
    )
    _upload_via_server_ingestion_job(fg, clean)


def _upload_to_hopsworks(df: pd.DataFrame) -> None:
    """Insert clean, direct-AQI features into the Hopsworks Feature Store."""
    import hopsworks

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in .env")

    clean = _prepare_hopsworks_dataframe(df)
    use_windows_transport = _uses_windows_hopsworks_transport()

    for attempt in range(1, _HOPSWORKS_UPLOAD_MAX_ATTEMPTS + 1):
        project = None
        retry_delay = None
        try:
            project = hopsworks.login(api_key_value=api_key)
            fs = project.get_feature_store()
            fg = _resolve_feature_group(fs)

            if use_windows_transport:
                _upload_on_windows(fg, clean)
            else:
                fg.insert(clean)
            return
        except Exception as error:
            if _is_native_hopsworks_transport_error(error):
                raise RuntimeError(
                    "Local feature and historical CSV files were saved, but "
                    "Hopsworks synchronization failed because its Delta/HDFS "
                    "transport is not supported by the installed native Windows "
                    "packages. Run the upload under Linux or WSL with a "
                    "Hopsworks-supported Python version (preferably Python 3.11 "
                    "or 3.12)."
                ) from error

            if not _is_transient_hopsworks_transport_error(error):
                raise

            if attempt == _HOPSWORKS_UPLOAD_MAX_ATTEMPTS:
                raise

            retry_delay = _HOPSWORKS_UPLOAD_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Transient Hopsworks HDFS/RPC failure on attempt %s/%s: %s. "
                "Retrying with a fresh session in %s seconds.",
                attempt,
                _HOPSWORKS_UPLOAD_MAX_ATTEMPTS,
                error,
                retry_delay,
            )
        finally:
            _close_hopsworks_project(project)

        if retry_delay is not None:
            time.sleep(retry_delay)

