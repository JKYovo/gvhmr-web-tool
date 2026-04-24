import json
import sqlite3
import threading
from pathlib import Path

from hmr4d.service.common import ensure_dir, terminal_job_states, utc_now_iso, write_json


class SQLiteJobStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path).expanduser().resolve()
        ensure_dir(self.db_path.parent)
        self._lock = threading.RLock()
        self._init_db()
        self._recover_unfinished_records()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    batch_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _recover_unfinished_records(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM jobs WHERE status IN ('queued', 'running')").fetchall()
            for row in rows:
                job = json.loads(row["payload_json"])
                job["status"] = "failed"
                job["finished_at"] = utc_now_iso()
                job["error_summary"] = "Service restarted before the job completed."
                job["updated_at"] = utc_now_iso()
                self._save_job(conn, job)
            conn.commit()

    def _save_job(self, conn, job):
        conn.execute(
            """
            INSERT INTO jobs (id, status, batch_id, created_at, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                batch_id=excluded.batch_id,
                updated_at=excluded.updated_at,
                payload_json=excluded.payload_json
            """,
            (
                job["job_id"],
                job["status"],
                job.get("batch_id"),
                job["submitted_at"],
                job["updated_at"],
                json.dumps(job, ensure_ascii=False),
            ),
        )
        self._write_job_manifest(job)

    def _save_batch(self, conn, batch):
        conn.execute(
            """
            INSERT INTO batches (id, status, created_at, updated_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                updated_at=excluded.updated_at,
                payload_json=excluded.payload_json
            """,
            (
                batch["batch_id"],
                batch["status"],
                batch["submitted_at"],
                batch["updated_at"],
                json.dumps(batch, ensure_ascii=False),
            ),
        )
        self._write_batch_manifest(batch)

    def _write_job_manifest(self, job):
        output_dir = Path(job["output_dir"])
        ensure_dir(output_dir)
        write_json(output_dir / "job.json", self._public_job(job))

    def _write_batch_manifest(self, batch):
        batch_dir = Path(batch["batch_dir"])
        ensure_dir(batch_dir)
        write_json(batch_dir / "batch.json", self._public_batch(batch))

    def _public_job(self, job):
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "input_video": job["input_video"],
            "submitted_at": job["submitted_at"],
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "static_cam": job["static_cam"],
            "f_mm": job["f_mm"],
            "save_intermediate": job["save_intermediate"],
            "generate_preview": job["generate_preview"],
            "output_dir": job["output_dir"],
            "artifacts": job.get("artifacts", {}),
            "error_summary": job.get("error_summary"),
        }

    def _public_batch(self, batch):
        return {
            "batch_id": batch["batch_id"],
            "submitted_at": batch["submitted_at"],
            "job_ids": batch["job_ids"],
            "total": batch["total"],
            "queued": batch["queued"],
            "running": batch["running"],
            "succeeded": batch["succeeded"],
            "failed": batch["failed"],
            "cancelled": batch["cancelled"],
        }

    def create_job(self, job):
        with self._lock, self._connect() as conn:
            self._save_job(conn, job)
            conn.commit()
        return job

    def save_job(self, job):
        with self._lock, self._connect() as conn:
            self._save_job(conn, job)
            conn.commit()
        return job

    def get_job(self, job_id):
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def list_jobs(self, limit=50, batch_id=None):
        query = "SELECT payload_json FROM jobs"
        params = []
        if batch_id:
            query += " WHERE batch_id = ?"
            params.append(batch_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def create_batch(self, batch):
        with self._lock, self._connect() as conn:
            self._save_batch(conn, batch)
            conn.commit()
        return batch

    def save_batch(self, batch):
        with self._lock, self._connect() as conn:
            self._save_batch(conn, batch)
            conn.commit()
        return batch

    def get_batch(self, batch_id):
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def list_batches(self, limit=20):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM batches ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def update_batch_counts(self, batch_id):
        batch = self.get_batch(batch_id)
        if batch is None:
            return None

        jobs = self.list_jobs(limit=max(batch["total"], 1_000_000), batch_id=batch_id)
        counts = {state: 0 for state in ["queued", "running", "succeeded", "failed", "cancelled"]}
        for job in jobs:
            counts[job["status"]] = counts.get(job["status"], 0) + 1

        batch.update(counts)
        batch["updated_at"] = utc_now_iso()
        if counts["running"] > 0:
            batch["status"] = "running"
        elif counts["queued"] > 0:
            batch["status"] = "queued"
        elif counts["failed"] > 0:
            batch["status"] = "failed"
        elif counts["cancelled"] == batch["total"]:
            batch["status"] = "cancelled"
        else:
            batch["status"] = "succeeded"
        return self.save_batch(batch)

    def terminal_jobs(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM jobs").fetchall()
        jobs = [json.loads(row["payload_json"]) for row in rows]
        return [job for job in jobs if job["status"] in terminal_job_states()]
