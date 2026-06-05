from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ingest.validate import validate_matches


def make_canonical() -> pd.DataFrame:
    """A canonical-shaped frame spanning two seasons with two malformed rows.

    m3 has missing goals; m5 has a missing date. The rest are clean.
    """
    df = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5"],
            "date": ["2023-08-12", "2023-08-19", "2023-08-26", "2024-08-17", None],
            "season": [2023, 2023, 2023, 2024, 2024],
            "home_team": ["A", "C", "A", "D", "B"],
            "away_team": ["B", "D", "C", "B", "A"],
            "home_goals": [2, 1, None, 0, 3],
            "away_goals": [0, 0, None, 0, 1],
            "result": ["HW", "HW", None, "D", "HW"],
        }
    )
    # Match the dtypes build_canonical produces before validation runs.
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["season"] = df["season"].astype("Int64")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    for col in ["match_id", "home_team", "away_team", "result"]:
        df[col] = df[col].astype("string")
    return df


class TestIngestValidation:
    """Tests for the data-quality validation step (Sprint 1)."""

    def test_schema_missing_column_raises(self, tmp_path):
        """AT1: a missing required column is a structural failure."""
        df = make_canonical().drop(columns=["home_goals"])
        with pytest.raises(ValueError):
            validate_matches(
                df,
                quarantine_dir=tmp_path / "quarantine",
                report_path=tmp_path / "report.json",
            )

    def test_malformed_rows_quarantined(self, tmp_path):
        """AT2: malformed rows leave the clean frame and land in a per-season quarantine file."""
        quarantine = tmp_path / "quarantine"
        clean, _ = validate_matches(
            make_canonical(),
            quarantine_dir=quarantine,
            report_path=tmp_path / "report.json",
        )

        # m3 (missing goals) and m5 (missing date) are gone from the clean data.
        kept_ids = set(clean["match_id"])
        assert "m3" not in kept_ids
        assert "m5" not in kept_ids

        # m3 belongs to season 2023; the quarantine file exists and contains it.
        q_2023 = quarantine / "dropped_2023.parquet"
        assert q_2023.exists()
        assert "m3" in set(pd.read_parquet(q_2023)["match_id"])

    def test_dropped_count_logged(self, tmp_path):
        """AT2: the report logs the dropped count and a per-reason breakdown."""
        _, report = validate_matches(
            make_canonical(),
            quarantine_dir=tmp_path / "quarantine",
            report_path=tmp_path / "report.json",
        )

        assert report["rows_dropped"] == 2
        assert report["dropped_by_reason"]["missing_goals"] == 1
        assert report["dropped_by_reason"]["missing_date"] == 1

    def test_clean_rows_retained(self, tmp_path):
        """AT2: all valid rows survive validation."""
        clean, _ = validate_matches(
            make_canonical(),
            quarantine_dir=tmp_path / "quarantine",
            report_path=tmp_path / "report.json",
        )

        assert set(clean["match_id"]) == {"m1", "m2", "m4"}
        assert not clean["home_goals"].isna().any()

    def test_report_written(self, tmp_path):
        """AT3: a validation report is written with the expected keys."""
        report_path = tmp_path / "report.json"
        validate_matches(
            make_canonical(),
            quarantine_dir=tmp_path / "quarantine",
            report_path=report_path,
        )

        assert report_path.exists()
        payload = json.loads(report_path.read_text())
        for key in ["total_rows", "rows_kept", "rows_dropped", "by_season"]:
            assert key in payload
        assert payload["total_rows"] == 5
        assert payload["rows_kept"] == 3

    def test_no_malformed_no_quarantine(self, tmp_path):
        """A clean dataset drops nothing and writes no quarantine files."""
        clean_input = make_canonical()
        clean_input = clean_input[clean_input["match_id"].isin(["m1", "m2", "m4"])]
        quarantine = tmp_path / "quarantine"

        clean, report = validate_matches(
            clean_input,
            quarantine_dir=quarantine,
            report_path=tmp_path / "report.json",
        )

        assert report["rows_dropped"] == 0
        assert len(clean) == 3
        # No malformed rows -> no quarantine files written.
        assert not (quarantine.exists() and any(quarantine.glob("dropped_*.parquet")))
