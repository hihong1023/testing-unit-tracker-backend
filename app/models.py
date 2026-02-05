# app/models.py
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, JSON
from uuid import uuid4

# ---------- USERS / TOKENS ----------

class User(SQLModel, table=False):
    id: str
    name: str
    role: str

class Token(SQLModel, table=True):
    token: str = Field(primary_key=True)
    user_id: str
    name: str
    role: str
    issued_at: datetime = Field(default_factory=datetime.utcnow)


# ---------- STEPS (still static, no table needed) ----------
# keep your TestStep as normal Pydantic/SQLModel table=False

class TestStep(SQLModel, table=False):
    id: int
    name: str
    order: int
    required: bool = True


# ---------- CORE ENTITIES ----------

class Unit(SQLModel, table=True):
    id: str = Field(primary_key=True)
    sku: Optional[str] = None
    rev: Optional[str] = None
    lot: Optional[str] = None
    status: str = "IN_PROGRESS"
    current_step_id: Optional[int] = None


class Assignment(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    unit_id: str = Field(foreign_key="unit.id")
    step_id: int
    tester_id: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    status: str = "PENDING"          # "PENDING" | "RUNNING" | "DONE"
    prev_passed: bool = False        # your existing field
    skipped: bool = False            # 🔹 NEW: this step is not tested for this unit
    actual_start_at: Optional[datetime] = Field(default=None, nullable=True)
    actual_end_at: Optional[datetime] = Field(default=None, nullable=True)
    
    sub_checks: Optional[Dict[str, bool]] = Field(
        default=None,
        sa_column=Column(JSON),
    )
    remark: Optional[str] = None
    
class AssignmentUpdate(SQLModel):
    tester_id: Optional[str] = None          # can be set to null to unassign
    start_at: Optional[datetime] = None      # can be set to null to clear
    end_at: Optional[datetime] = None        # can be set to null to clear
    status: Optional[str] = None             # PENDING/RUNNING/DONE/PASS/FAIL
    skipped: Optional[bool] = None           # mark as Not Tested / back
    remark: Optional[str] = None

class Result(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    unit_id: str = Field(index=True, foreign_key="unit.id")
    step_id: int = Field(index=True)
    passed: bool
    metrics: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    files: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    submitted_by: Optional[str] = None
    finished_at: Optional[datetime] = Field(default=None, nullable=True)


class FileMeta(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    unit_id: str = Field(index=True, foreign_key="unit.id")
    step_id: int = Field(index=True)
    result_id: str = Field(index=True, foreign_key="result.id")
    orig_name: str
    stored_name: str
    stored_path: str
    sha256: str = Field(index=True)
    size: int


class Notification(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    tester_id: str = Field(index=True)
    unit_id: str = Field(index=True, foreign_key="unit.id")
    from_step_id: int
    to_step_id: int
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    read: bool = False






