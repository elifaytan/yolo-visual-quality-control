import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


DATABASE_PATH = Path("quality_control.db")


def create_connection() -> sqlite3.Connection:
    """SQLite veritabanı bağlantısı oluşturur."""

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """Kontrol kayıtları tablosunu oluşturur."""

    with create_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_time TEXT NOT NULL,
                profile_name TEXT NOT NULL,
                result TEXT NOT NULL,
                error_count INTEGER NOT NULL,
                detected_objects TEXT NOT NULL,
                expected_objects TEXT NOT NULL,
                forbidden_objects TEXT NOT NULL,
                position_rules TEXT NOT NULL,
                errors TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )

        connection.commit()


def save_inspection(
    profile_name: str,
    is_pass: bool,
    detected_objects: dict[str, int],
    expected_objects: dict[str, int],
    forbidden_objects: list[str],
    position_rules: dict[str, str],
    errors: list[str],
    confidence: float,
) -> int:
    """Tek bir kalite kontrol sonucunu veritabanına kaydeder."""

    inspection_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    result = "PASS" if is_pass else "FAIL"

    with create_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                inspection_time,
                profile_name,
                result,
                error_count,
                detected_objects,
                expected_objects,
                forbidden_objects,
                position_rules,
                errors,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_time,
                profile_name,
                result,
                len(errors),
                json.dumps(
                    detected_objects,
                    ensure_ascii=False,
                ),
                json.dumps(
                    expected_objects,
                    ensure_ascii=False,
                ),
                json.dumps(
                    forbidden_objects,
                    ensure_ascii=False,
                ),
                json.dumps(
                    position_rules,
                    ensure_ascii=False,
                ),
                json.dumps(
                    errors,
                    ensure_ascii=False,
                ),
                confidence,
            ),
        )

        connection.commit()

        return int(cursor.lastrowid)


def get_inspection_history() -> pd.DataFrame:
    """Bütün kontrol kayıtlarını yeni kayıttan eskiye doğru getirir."""

    with create_connection() as connection:
        query = """
            SELECT
                id,
                inspection_time,
                profile_name,
                result,
                error_count,
                detected_objects,
                errors,
                confidence
            FROM inspections
            ORDER BY id DESC
        """

        dataframe = pd.read_sql_query(
            query,
            connection,
        )

    if dataframe.empty:
        return dataframe

    dataframe = dataframe.rename(
        columns={
            "id": "Kontrol No",
            "inspection_time": "Tarih ve Saat",
            "profile_name": "Ürün Profili",
            "result": "Sonuç",
            "error_count": "Uygunsuzluk Sayısı",
            "detected_objects": "Algılanan Nesneler",
            "errors": "Hata Açıklamaları",
            "confidence": "Güven Eşiği",
        }
    )

    dataframe["Algılanan Nesneler"] = dataframe[
        "Algılanan Nesneler"
    ].apply(format_json_value)

    dataframe["Hata Açıklamaları"] = dataframe[
        "Hata Açıklamaları"
    ].apply(format_json_value)

    return dataframe


def format_json_value(value: str) -> str:
    """JSON metnini tabloda okunabilir hâle getirir."""

    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return str(value)

    if isinstance(parsed_value, dict):
        if not parsed_value:
            return "-"

        return ", ".join(
            f"{key}: {item_value}"
            for key, item_value in parsed_value.items()
        )

    if isinstance(parsed_value, list):
        if not parsed_value:
            return "-"

        return " | ".join(
            str(item)
            for item in parsed_value
        )

    return str(parsed_value)


def get_summary_statistics() -> dict[str, float | int]:
    """Kontrol geçmişine ait özet istatistikleri hesaplar."""

    with create_connection() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_count,
                SUM(
                    CASE WHEN result = 'PASS'
                    THEN 1 ELSE 0 END
                ) AS pass_count,
                SUM(
                    CASE WHEN result = 'FAIL'
                    THEN 1 ELSE 0 END
                ) AS fail_count
            FROM inspections
            """
        ).fetchone()

    total_count = int(row["total_count"] or 0)
    pass_count = int(row["pass_count"] or 0)
    fail_count = int(row["fail_count"] or 0)

    pass_rate = (
        (pass_count / total_count) * 100
        if total_count > 0
        else 0.0
    )

    return {
        "total_count": total_count,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_rate, 2),
    }


def delete_all_inspections() -> None:
    """Bütün kontrol geçmişini siler."""

    with create_connection() as connection:
        connection.execute(
            "DELETE FROM inspections"
        )

        connection.commit()