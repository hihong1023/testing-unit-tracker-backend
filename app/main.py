from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
from pathlib import Path
import hashlib
import io
import re

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Font

from sqlmodel import Session, select

# 🔹 IMPORTANT: import everything you need from db.py here
from app.db import (
    engine,
    get_session,
    download_db_from_blob_if_needed,
    upload_db_to_blob,
    init_db,
)
# 🔹 Your models
from app.models import (
    Unit,
    Assignment,
    Result,
    FileMeta,
    Notification,
    Token,
    User,
    TestStep,
    AssignmentUpdate,
)

# =====================================================
# App & CORS
# =====================================================

app = FastAPI(title="Testing Unit Tracker")

origins = [
    "https://proud-sand-0ed440210.3.azurestaticapps.net",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # you can later tighten to just your frontend origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # 1) Ensure we have the latest DB from blob
    download_db_from_blob_if_needed()

    # 2) Ensure tables exist
    init_db()

    print("[APP] Startup complete.")


@app.on_event("shutdown")
def on_shutdown():
    # Upload the current DB to blob so it persists
    upload_db_to_blob()
    print("[APP] Shutdown complete.")

# =====================================================
# Auth (simple token-based)
# =====================================================

class LoginRequest(BaseModel):
    name: str  # username from frontend

class LoginResponse(BaseModel):
    access_token: str
    role: str
    user: User

security = HTTPBearer()
TOKEN_TTL = timedelta(hours=12)

async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session),
) -> User:
    token_str = creds.credentials
    token_row = session.get(Token, token_str)

    if not token_row:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if datetime.utcnow() - token_row.issued_at > TOKEN_TTL:
        session.delete(token_row)
        session.commit()
        raise HTTPException(status_code=401, detail="Token expired")

    return User(id=token_row.user_id, name=token_row.name, role=token_row.role)

def require_role(required_role: str):
    async def dep(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            # hide existence of the route from wrong-role callers
            raise HTTPException(status_code=404, detail="Not found")
        return user
    return dep

# Preset accounts
PRESET_TESTERS = [
    "alex", "brian", "ge fan", "jimmy", "kae", "krish",
    "nicholas", "sunny", "yew meng", "yubo", "zhen yang",
]
PRESET_SUPERVISORS = ["kian siang", "alban", "hai hong"]

@app.post("/auth/login", response_model=LoginResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    username = body.name.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Name is required")

    key = username.lower()

    if key in PRESET_SUPERVISORS:
        role = "supervisor"
    elif key in PRESET_TESTERS:
        role = "tester"
    else:
        raise HTTPException(status_code=401, detail="Unknown user")

    user = User(id=str(uuid4()), name=username, role=role)

    token_str = str(uuid4())
    token_row = Token(
        token=token_str,
        user_id=user.id,
        name=user.name,
        role=user.role,
    )
    session.add(token_row)
    session.commit()

    return LoginResponse(access_token=token_str, role=role, user=user)

@app.get("/testers", response_model=List[str])
def list_testers(supervisor: User = Depends(require_role("supervisor"))):
    """Scheduler uses this to populate tester dropdown (supervisor only)."""
    return PRESET_TESTERS

# =====================================================
# Static Test Steps
# =====================================================

STEPS: List[TestStep] = [
    TestStep(id=1, name="Connectivity Test", order=1),
    TestStep(id=2, name="Functionality Test", order=2),
    TestStep(id=3, name="EIRP Determination & Stability Calibration", order=3),
    TestStep(id=4, name="Pre-Vibration Physical Layer Test", order=4),
    TestStep(id=5, name="Vibration Test", order=5),
    TestStep(id=6, name="Post-Vibration Physical Layer Test", order=6),
    TestStep(id=7, name="Thermal Cycling", order=7),
    TestStep(id=8, name="Post-Thermal Cycling Physical Layer Test", order=8),
    TestStep(id=9, name="Burn-in Test", order=9),
    TestStep(id=10, name="EMI/EMC Test", order=10),
    TestStep(id=11, name="Post-EMI/EMC Physical Layer Test", order=11),
    TestStep(id=12, name="BGAN Network Emulator Test", order=12),
    TestStep(id=13, name="Over-the-Air Test", order=13),
]

STEP_BY_ID: Dict[int, TestStep] = {s.id: s for s in STEPS}
STEP_IDS_ORDERED = [s.id for s in sorted(STEPS, key=lambda s: s.order)]

@app.get("/steps", response_model=List[TestStep])
def list_steps(user: User = Depends(get_current_user)):
    return STEPS

# =====================================================
# Units
# =====================================================

class CreateUnitRequest(BaseModel):
    unit_id: str
    sku: Optional[str] = None
    rev: Optional[str] = None
    lot: Optional[str] = None

class UnitSummary(BaseModel):
    unit_id: str
    status: str
    progress_percent: float
    passed_steps: int
    total_steps: int
    next_step_id: Optional[int]
    next_step_name: Optional[str]

class UnitDetails(BaseModel):
    unit: Unit
    assignments: List[Assignment]
    results: List[Result]

@app.post("/units", response_model=Unit)
def create_unit(
    body: CreateUnitRequest,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    unit_id = body.unit_id.strip()
    if not unit_id:
        raise HTTPException(status_code=400, detail="unit_id required")

    existing = session.get(Unit, unit_id)
    if existing:
        raise HTTPException(status_code=400, detail="Unit already exists")

    unit = Unit(
        id=unit_id,
        sku=body.sku,
        rev=body.rev,
        lot=body.lot,
        current_step_id=STEP_IDS_ORDERED[0] if STEP_IDS_ORDERED else None,
    )
    session.add(unit)

    # create assignments for all steps
    for idx, step_id in enumerate(STEP_IDS_ORDERED):
        a = Assignment(
            unit_id=unit_id,
            step_id=step_id,
            prev_passed=(idx == 0),  # first step unblocked
        )
        session.add(a)

    session.commit()
    session.refresh(unit)
    return unit

@app.delete("/units/{unit_id}", status_code=200)
def delete_unit(
    unit_id: str,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Delete a unit and all related data.
    Idempotent: if unit doesn't exist, returns 200 with deleted=False.
    """
    unit = session.get(Unit, unit_id)
    if not unit:
        return {"ok": True, "deleted": False}

    # delete file metadata + physical files
    files = session.exec(
        select(FileMeta).where(FileMeta.unit_id == unit_id)
    ).all()
    for f in files:
        if f.stored_path:
            p = Path(f.stored_path)
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        session.delete(f)

    # delete results
    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()
    for r in results:
        session.delete(r)

    # delete assignments
    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    for a in assignments:
        session.delete(a)

    # delete notifications for this unit
    notes = session.exec(
        select(Notification).where(Notification.unit_id == unit_id)
    ).all()
    for n in notes:
        session.delete(n)

    session.delete(unit)
    session.commit()

    return {"ok": True, "deleted": True}

class RenameUnitRequest(BaseModel):
    new_unit_id: str


@app.patch("/units/{unit_id}/rename", response_model=Unit)
def rename_unit(
    unit_id: str,
    body: RenameUnitRequest,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Rename a unit and update all related records (assignments, results,
    files, notifications) to use the new unit_id.

    Frontend:
      PATCH /units/{unit_id}/rename
      { "new_unit_id": "NEW_ID" }
    """
    new_id = body.new_unit_id.strip()
    if not new_id:
        raise HTTPException(status_code=400, detail="new_unit_id required")

    if new_id == unit_id:
        # nothing to do
        unit = session.get(Unit, unit_id)
        if not unit:
            raise HTTPException(status_code=404, detail="Unit not found")
        return unit

    # Check if target id already exists
    existing = session.get(Unit, new_id)
    if existing:
        raise HTTPException(status_code=400, detail="Target unit_id already exists")

    # Load the unit we are renaming
    unit = session.get(Unit, unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    # --- Update related tables first ---

    # Assignments
    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    for a in assignments:
        a.unit_id = new_id
        session.add(a)

    # Results
    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()
    for r in results:
        r.unit_id = new_id
        session.add(r)

    # File metadata
    files = session.exec(
        select(FileMeta).where(FileMeta.unit_id == unit_id)
    ).all()
    for f in files:
        f.unit_id = new_id
        session.add(f)
        # (Optional) we are NOT renaming physical file paths here;
        # stored_path can still contain the old unit id in the filename.

    # Notifications (if any)
    notes = session.exec(
        select(Notification).where(Notification.unit_id == unit_id)
    ).all()
    for n in notes:
        n.unit_id = new_id
        session.add(n)

    # --- Finally update the Unit primary key ---
    unit.id = new_id
    session.add(unit)

    session.commit()
    session.refresh(unit)
    return unit

@app.get("/units/summary", response_model=List[UnitSummary])
def get_units_summary(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    units = session.exec(select(Unit)).all()
    summaries: List[UnitSummary] = []

    for u in units:
        # All assignments for this unit, sorted by step order
        assigns = session.exec(
            select(Assignment).where(Assignment.unit_id == u.id)
        ).all()
        assigns.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

        # 🔹 Only count non-skipped steps in progress
        effective_assigns = [a for a in assigns if not a.skipped]
        total_steps = len(effective_assigns)

        # All results for this unit
        results = session.exec(
            select(Result).where(Result.unit_id == u.id)
        ).all()
        result_map = {(r.unit_id, r.step_id): r for r in results}

        # 🔹 Passed steps = number of effective steps with a PASS result
        passed_steps = sum(
            1
            for a in effective_assigns
            if (u.id, a.step_id) in result_map
            and result_map[(u.id, a.step_id)].passed
        )

        # 🔹 Progress percentage (0–100)
        if total_steps > 0:
            progress = (passed_steps / total_steps) * 100.0
        else:
            progress = 0.0

        # 🔹 Next step: first effective step that does NOT have a result
        next_step_id: Optional[int] = None
        next_step_name: Optional[str] = None
        for a in effective_assigns:
            if (u.id, a.step_id) not in result_map:
                next_step_id = a.step_id
                next_step_name = STEP_BY_ID[a.step_id].name
                break

        # 🔹 Status
        if total_steps == 0:
            status = "EMPTY"
        else:
            # How many effective steps actually have a result?
            completed_effective = sum(
                1 for a in effective_assigns if (u.id, a.step_id) in result_map
            )
            if completed_effective == total_steps:
                status = "COMPLETED"
            else:
                status = "IN_PROGRESS"

        summaries.append(
            UnitSummary(
                unit_id=u.id,
                status=status,
                progress_percent=progress,
                passed_steps=passed_steps,
                total_steps=total_steps,
                next_step_id=next_step_id,
                next_step_name=next_step_name,
            )
        )

    return summaries


@app.get("/units/{unit_id}/details", response_model=UnitDetails)
def get_unit_details(
    unit_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Details view for a single unit.
    Used by UnitsPage details route: /units/{unit_id}/details
    """
    unit = session.get(Unit, unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()

    return UnitDetails(unit=unit, assignments=assignments, results=results)

# =====================================================
# Results & Uploads
# =====================================================
def recompute_unit_status(session: Session, unit_id: str) -> None:
    """
    Helper: set unit.status to COMPLETED when all assignments are
    either skipped or finished (DONE/PASS/FAIL). Otherwise mark as IN_PROGRESS.
    """
    unit = session.get(Unit, unit_id)
    if not unit:
        return

    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()

    if not assignments:
        # no assignments, keep whatever status it has
        return

    all_done = all(
        a.skipped or a.status in ("DONE", "PASS", "FAIL")
        for a in assignments
    )

    if all_done:
        unit.status = "COMPLETED"
    else:
        # only override if previously completed; otherwise keep existing
        if unit.status == "COMPLETED":
            unit.status = "IN_PROGRESS"

    session.add(unit)
    
class ResultIn(BaseModel):
    unit_id: str
    step_id: int
    metrics: Optional[Dict[str, Any]] = None
    passed: bool
    finished_at: Optional[datetime] = None

class ResultOut(BaseModel):
    id: str
    unit_id: str
    step_id: int
    passed: bool
    metrics: Dict[str, Any]
    files: List[str]
    submitted_by: Optional[str]
    finished_at: datetime

@app.post("/results", response_model=ResultOut)
def create_or_update_result(
    body: ResultIn,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if body.step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")

    unit = session.get(Unit, body.unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    unit_id = body.unit_id
    step_id = body.step_id
    passed = body.passed
    metrics = body.metrics or {}
    finished = body.finished_at or datetime.utcnow()

    # ----- create / update Result -----
    existing_result = session.exec(
        select(Result).where(
            Result.unit_id == unit_id,
            Result.step_id == step_id,
        )
    ).first()

    if existing_result:
        existing_result.metrics = metrics
        existing_result.passed = passed
        existing_result.finished_at = finished
        res = existing_result
    else:
        res = Result(
            unit_id=unit_id,
            step_id=step_id,
            passed=passed,
            metrics=metrics,
            files=[],
            submitted_by=user.id,
            finished_at=finished,
        )
        session.add(res)

    # ----- update Assignment for this step -----
    a = session.exec(
        select(Assignment).where(
            Assignment.unit_id == unit_id,
            Assignment.step_id == step_id,
        )
    ).first()
    if a:
        a.tester_id = user.name
        # store PASS/FAIL so scheduler + matrix see same state
        a.status = "PASS" if passed else "FAIL"
        session.add(a)

    # ----- chain prev_passed to next step -----
    if step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_step_id = STEP_IDS_ORDERED[idx + 1]
            nxt = session.exec(
                select(Assignment).where(
                    Assignment.unit_id == unit_id,
                    Assignment.step_id == next_step_id,
                )
            ).first()
            if nxt:
                nxt.prev_passed = passed
                session.add(nxt)

    # ----- recompute unit.status (COMPLETED / IN_PROGRESS) -----
    recompute_unit_status(session, unit_id)

    session.commit()
    session.refresh(res)

    # ----- create notification for next tester (same as before) -----
    if passed and step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_step_id = STEP_IDS_ORDERED[idx + 1]
            next_assign = session.exec(
                select(Assignment).where(
                    Assignment.unit_id == unit_id,
                    Assignment.step_id == next_step_id,
                )
            ).first()
            if next_assign and next_assign.tester_id:
                dup = session.exec(
                    select(Notification).where(
                        Notification.tester_id == next_assign.tester_id,
                        Notification.unit_id == unit_id,
                        Notification.from_step_id == step_id,
                        Notification.to_step_id == next_step_id,
                    )
                ).first()
                if not dup:
                    note = Notification(
                        tester_id=next_assign.tester_id,
                        unit_id=unit_id,
                        from_step_id=step_id,
                        to_step_id=next_step_id,
                        message=(
                            f"Unit {unit_id} is ready for "
                            f"{STEP_BY_ID[next_step_id].name} (previous step passed)."
                        ),
                    )
                    session.add(note)
                    session.commit()

    return res



# Storage config for uploads
STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(exist_ok=True)

ALLOWED_EXT = {".zip", ".csv", ".pdf", ".png"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

@app.post("/uploads")
async def upload_evidence(
    unit_id: str = Form(...),
    step_id: int = Form(...),
    result_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    unit = session.get(Unit, unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    if step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")

    res = session.get(Result, result_id)
    if not res:
        raise HTTPException(status_code=404, detail="Result not found")

    if res.unit_id != unit_id or res.step_id != step_id:
        raise HTTPException(
            status_code=400,
            detail="result_id does not match unit/step",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    content = await file.read()
    size = len(content)
    if size > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large")

    sha = hashlib.sha256(content).hexdigest()

    existing = session.exec(
        select(FileMeta).where(
            FileMeta.sha256 == sha,
            FileMeta.unit_id == unit_id,
            FileMeta.step_id == step_id,
        )
    ).first()

    if existing:
        if existing.id not in res.files:
            res.files.append(existing.id)
            session.add(res)
            session.commit()
        return {"file_id": existing.id, "deduplicated": True}

    bucket = sha[:2]
    bucket_dir = STORAGE_ROOT / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:6]
    server_name = f"{unit_id}_{step_id}_{timestamp}_{unique_suffix}{ext}"
    stored_path = bucket_dir / server_name

    with open(stored_path, "wb") as f:
        f.write(content)

    meta = FileMeta(
        unit_id=unit_id,
        step_id=step_id,
        result_id=result_id,
        orig_name=file.filename,
        stored_name=server_name,
        stored_path=str(stored_path),
        sha256=sha,
        size=size,
    )
    session.add(meta)
    session.commit()
    session.refresh(meta)

    res.files.append(meta.id)
    session.add(res)
    session.commit()

    return {"file_id": meta.id, "deduplicated": False}

@app.get("/reports/unit/{unit_id}/traveller.xlsx")
def export_traveller_log(
    unit_id: str,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Export a traveller log for one unit as Excel.

    Layout:
    Row 1:  step headers  (2. Functionality Test, 3. EIRP..., etc.)
    Row 2:  tester names  (Tester1, Tester2, ...)
    Row 3:  finish dates  (20-Nov, 21-Nov, ...)
    Row 4:  results       (Pass / Fail, coloured)
    """
    unit = session.get(Unit, unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    # All assignments & results for this unit
    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()

    # Maps by step_id for quick lookup
    assignments_by_step = {a.step_id: a for a in assignments}
    results_by_step = {r.step_id: r for r in results}

    # Use your static STEP order
    ordered_steps = [STEP_BY_ID[sid] for sid in STEP_IDS_ORDERED]

    # --- Build Excel workbook ---
    wb = Workbook()
    ws = wb.active
    ws.title = "Traveller"

    # Helper for formatting dates like "20-Nov"
    def fmt_date(dt):
        if not dt:
            return ""
        return dt.strftime("%d-%b")

    # Row 1: step headers
    ws.cell(row=1, column=1, value="Unit")
    for col_idx, step in enumerate(ordered_steps, start=2):
        ws.cell(
            row=1,
            column=col_idx,
            value=f"{step.order}. {step.name}",
        )

    # Row 2: tester names
    ws.cell(row=2, column=1, value="Tester")
    for col_idx, step in enumerate(ordered_steps, start=2):
        a = assignments_by_step.get(step.id)
        ws.cell(row=2, column=col_idx, value=a.tester_id if a else "")

    # Row 3: finish dates
    ws.cell(row=3, column=1, value="Date")
    for col_idx, step in enumerate(ordered_steps, start=2):
        r = results_by_step.get(step.id)
        ws.cell(row=3, column=col_idx, value=fmt_date(r.finished_at) if r else "")

    # Row 4: Pass / Fail with colours
    ws.cell(row=4, column=1, value="Result")
    for col_idx, step in enumerate(ordered_steps, start=2):
        r = results_by_step.get(step.id)
        if not r:
            continue

        val = "Pass" if r.passed else "Fail"
        cell = ws.cell(row=4, column=col_idx, value=val)

        if r.passed:
            cell.fill = PatternFill("solid", fgColor="C6EFCE")  # light green
            cell.font = Font(color="006100")
        else:
            cell.fill = PatternFill("solid", fgColor="FFC7CE")  # light red
            cell.font = Font(color="9C0006")

    # Column widths & simple alignment
    for col_idx in range(1, len(ordered_steps) + 2):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 20

    # Top-left cell with unit id
    ws.cell(row=1, column=1, value=unit_id)

    # --- Return as downloadable file ---
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"{unit_id}_traveller.xlsx"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }

    return StreamingResponse(
        buf,
        media_type=(
            "application/"
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        headers=headers,
    )


# =====================================================
# Tester: Assignments view (tester home page)
# =====================================================

@app.get("/tester/assignments", response_model=List[Assignment])
def get_tester_assignments(
    tester_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    List assignments that a tester can see / work on.

    Frontend calls:
      GET /tester/assignments?tester_id=<name>

    Rules:
    - Tester can only query their own assignments (tester_id must match login name)
    - Supervisor can query assignments for any tester
    - Only steps that are PENDING/RUNNING, whose previous step passed,
      and that do NOT already have a Result are returned.
    """
    # Security: testers may only query themselves
    if user.role == "tester" and user.name != tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Load all assignments and results
    assignments = session.exec(select(Assignment)).all()
    results = session.exec(select(Result)).all()
    result_set = {(r.unit_id, r.step_id) for r in results}

    visible: List[Assignment] = []

    for a in assignments:
        # ignore completed steps
        if a.status not in ("PENDING", "RUNNING"):
            continue

        # ignore steps that already have a result
        if (a.unit_id, a.step_id) in result_set:
            continue

        # previous step must be passed
        if not a.prev_passed:
            continue

        # tester can see:
        # - steps assigned to them
        # - unassigned steps (None)
        if a.tester_id not in (None, tester_id):
            continue

        visible.append(a)

    # Sort nicely for UI: by unit, then by step order
    visible.sort(key=lambda x: (x.unit_id, STEP_BY_ID[x.step_id].order))
    return visible

@app.get("/tester/schedule", response_model=List[Assignment])
def get_tester_schedule(
    tester_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Full schedule view for a tester.

    Used by 'Upcoming Tests (Tester)' page.
    - Shows ALL assignments assigned to this tester
      (PENDING/RUNNING/DONE/SKIPPED, regardless of prev_passed or Result).
    """

    # Security: testers may only query themselves
    if user.role == "tester" and user.name != tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    assignments = session.exec(
        select(Assignment).where(Assignment.tester_id == tester_id)
    ).all()

    # Sort nicely: by start date, then unit, then step order
    assignments.sort(
        key=lambda a: (
            a.start_at or datetime.max,
            a.unit_id,
            STEP_BY_ID[a.step_id].order,
        )
    )
    return assignments

# =====================================================
# Scheduling (Supervisor)
# =====================================================

class DuplicateRequest(BaseModel):
    source_unit_id: str
    new_unit_ids: List[str]
    day_shift: int = 1

@app.post("/schedule/duplicate")
def duplicate_schedule(
    body: DuplicateRequest,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Duplicate the schedule (assignments) from one source unit to
    one or more new units, shifting dates by `day_shift` days.

    Used by the Scheduler "Duplicate Schedule" modal.
    """
    src_id = body.source_unit_id.strip()
    if not src_id:
        raise HTTPException(status_code=400, detail="source_unit_id required")

    # 1) Source unit must exist
    src_unit = session.get(Unit, src_id)
    if not src_unit:
        raise HTTPException(status_code=404, detail="Source unit not found")

    # 2) Load and sort source assignments by step order
    src_assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == src_id)
    ).all()
    src_assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

    def shift(dt: Optional[datetime]) -> Optional[datetime]:
        if not dt:
            return None
        return dt + timedelta(days=body.day_shift)

    created_units: List[str] = []

    for new_id_raw in body.new_unit_ids:
        new_id = new_id_raw.strip()
        if not new_id:
            continue

        # Don't allow overwrite of existing units
        existing = session.get(Unit, new_id)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Unit {new_id} already exists",
            )

        # 3) Create the new unit, copying SKU/REV/LOT from source
        new_unit = Unit(
            id=new_id,
            sku=src_unit.sku,
            rev=src_unit.rev,
            lot=src_unit.lot,
            status="IN_PROGRESS",
            current_step_id=STEP_IDS_ORDERED[0] if STEP_IDS_ORDERED else None,
        )
        session.add(new_unit)

        # 4) Clone all assignments with shifted dates
        for src_a in src_assignments:
            new_a = Assignment(
                unit_id=new_id,
                step_id=src_a.step_id,
                tester_id=src_a.tester_id,
                start_at=shift(src_a.start_at),
                end_at=shift(src_a.end_at),
                status="PENDING",
                prev_passed=(src_a.step_id == STEP_IDS_ORDERED[0]),
            )
            session.add(new_a)

        created_units.append(new_id)

    session.commit()
    return {"ok": True, "created_units": created_units}

class AssignmentPatch(BaseModel):
    tester_id: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    status: Optional[str] = None
    skipped: Optional[bool] = None

def overlaps(
    a_start: Optional[datetime],
    a_end: Optional[datetime],
    b_start: Optional[datetime],
    b_end: Optional[datetime],
) -> bool:
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return (a_start <= b_end) and (b_start <= a_end)

@app.get("/assignments/schedule", response_model=List[Assignment])
def get_schedule(
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """Return all assignments for the scheduler page (Gantt-like view)."""
    assignments = session.exec(select(Assignment)).all()
    assignments.sort(key=lambda a: (a.unit_id, STEP_BY_ID[a.step_id].order))
    return assignments

@app.patch("/assignments/{assign_id}", response_model=Assignment)
def patch_assignment(
    assign_id: str,
    body: AssignmentPatch,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    a = session.get(Assignment, assign_id)
    if not a:
        raise HTTPException(status_code=404, detail="Assignment not found")

    sent_fields = body.__fields_set__

    # --- if skipping, clear tester & dates and mark status ---
    if body.skipped is True:
        a.skipped = True
        a.tester_id = None
        a.start_at = None
        a.end_at = None
        a.status = "SKIPPED"
    else:
        # ✅ explicit updates (including None to clear)
        if "tester_id" in sent_fields:
            a.tester_id = body.tester_id  # None → unassign

        if "start_at" in sent_fields:
            a.start_at = body.start_at    # None → clear

        if "end_at" in sent_fields:
            a.end_at = body.end_at        # None → clear

        prev_status = a.status
        if "status" in sent_fields and body.status is not None:
            a.status = body.status

        # --- sync PASS / FAIL into Result + next.prev_passed ---
        if "status" in sent_fields and body.status in ("PASS", "FAIL"):
            passed = body.status == "PASS"

            # upsert Result row for this unit / step
            r = session.exec(
                select(Result).where(
                    Result.unit_id == a.unit_id,
                    Result.step_id == a.step_id,
                )
            ).first()
            now = datetime.utcnow()
            if r:
                r.passed = passed
                r.finished_at = r.finished_at or now
                session.add(r)
            else:
                r = Result(
                    unit_id=a.unit_id,
                    step_id=a.step_id,
                    passed=passed,
                    metrics={},
                    files=[],
                    submitted_by=None,
                    finished_at=now,
                )
                session.add(r)

            # propagate prev_passed to next step
            if a.step_id in STEP_IDS_ORDERED:
                idx = STEP_IDS_ORDERED.index(a.step_id)
                if idx + 1 < len(STEP_IDS_ORDERED):
                    next_step_id = STEP_IDS_ORDERED[idx + 1]
                    nxt = session.exec(
                        select(Assignment).where(
                            Assignment.unit_id == a.unit_id,
                            Assignment.step_id == next_step_id,
                        )
                    ).first()
                    if nxt:
                        nxt.prev_passed = passed
                        session.add(nxt)

        # if un-skipping
        if body.skipped is False:
            a.skipped = False
            if a.status == "SKIPPED":
                a.status = "PENDING"

    # --- validate overlaps only if NOT skipped ---
    if not a.skipped:
        new_start = a.start_at
        new_end = a.end_at
        if new_start and new_end and new_end < new_start:
            raise HTTPException(
                status_code=400,
                detail="End date cannot be before start date",
            )

        others = session.exec(
            select(Assignment).where(Assignment.unit_id == a.unit_id)
        ).all()
        for other in others:
            if other.id == a.id or other.skipped:
                continue
            if overlaps(new_start, new_end, other.start_at, other.end_at):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Unit '{a.unit_id}' already has another test scheduled "
                        f"from {other.start_at} to {other.end_at}"
                    ),
                )

    # --- recompute unit.status after any change ---
    recompute_unit_status(session, a.unit_id)

    session.add(a)
    session.commit()
    session.refresh(a)
    return a


# =====================================================
# Traveller Log
# =====================================================
def safe_sheet_title(unit_id: str) -> str:
    # Excel sheet title must be <=31 chars and not contain [ ] : * ? / \
    title = re.sub(r'[\[\]\:\*\?\/\\]', "_", unit_id)
    if len(title) > 31:
        title = title[:31]
    return title or "Unit"


def add_traveller_sheet_for_unit(
    wb: Workbook,
    unit_id: str,
    session: Session,
    first_sheet: bool = False,
):
    unit = session.get(Unit, unit_id)
    if not unit:
        # silently skip missing units; you can also raise if you prefer
        return

    # assignments + results
    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()

    assignments_by_step = {a.step_id: a for a in assignments}
    results_by_step = {r.step_id: r for r in results}

    # use your known ordered steps
    ordered_steps: list[TestStep] = [STEP_BY_ID[sid] for sid in STEP_IDS_ORDERED]

    # create / reuse sheet
    sheet_name = safe_sheet_title(unit_id)
    if first_sheet:
        ws = wb.active
        ws.title = sheet_name
    else:
        ws = wb.create_sheet(title=sheet_name)

    def fmt_date(dt):
        if not dt:
            return ""
        # 20-Nov format
        return dt.strftime("%d-%b")

    # Row 1: step headers
    ws.cell(row=1, column=1, value=unit_id)
    for col_idx, step in enumerate(ordered_steps, start=2):
        ws.cell(
            row=1,
            column=col_idx,
            value=f"{step.order}. {step.name}",
        )

    # Row 2: testers
    ws.cell(row=2, column=1, value="Tester")
    for col_idx, step in enumerate(ordered_steps, start=2):
        a = assignments_by_step.get(step.id)
        ws.cell(row=2, column=col_idx, value=a.tester_id if a else "")

    # Row 3: dates
    ws.cell(row=3, column=1, value="Date")
    for col_idx, step in enumerate(ordered_steps, start=2):
        r = results_by_step.get(step.id)
        ws.cell(row=3, column=col_idx, value=fmt_date(r.finished_at) if r else "")

    # Row 4: result
    ws.cell(row=4, column=1, value="Result")
    for col_idx, step in enumerate(ordered_steps, start=2):
        r = results_by_step.get(step.id)
        if not r:
            continue
        val = "Pass" if r.passed else "Fail"
        cell = ws.cell(row=4, column=col_idx, value=val)
        if r.passed:
            cell.fill = PatternFill("solid", fgColor="C6EFCE")
            cell.font = Font(color="006100")
        else:
            cell.fill = PatternFill("solid", fgColor="FFC7CE")
            cell.font = Font(color="9C0006")

    # column widths
    for col_idx in range(1, len(ordered_steps) + 2):
        ws.column_dimensions[get_column_letter(col_idx)].width = 20

class BulkTravellerRequest(BaseModel):
    unit_ids: List[str]


@app.post("/reports/traveller/bulk.xlsx")
def export_traveller_bulk_xlsx(
    payload: BulkTravellerRequest,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    if not payload.unit_ids:
        raise HTTPException(status_code=400, detail="No unit_ids provided")

    wb = Workbook()
    first = True
    for unit_id in payload.unit_ids:
        add_traveller_sheet_for_unit(wb, unit_id, session, first_sheet=first)
        first = False

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="traveller_logs.xlsx"'
    }

    return StreamingResponse(
        buf,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers=headers,
    )

# =====================================================
# Root
# =====================================================

@app.get("/")
def root():
    return {"message": "Testing Unit Tracker API running"}




















