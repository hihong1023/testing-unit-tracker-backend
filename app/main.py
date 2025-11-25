from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import uuid4
from pathlib import Path
import hashlib
import json
import zipfile
import os
import re
from sqlmodel import Session, select
from app.db import init_db, get_session
from app.models import (
    Unit, Assignment, Result, FileMeta, Notification, Token, User, TestStep
)


app = FastAPI(title="Testing Unit Tracker")

@app.on_event("startup")
def on_startup():
    init_db()
    

origins = [
    "https://proud-sand-0ed440210.3.azurestaticapps.net",
    "http://localhost:5173",  # keep for local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Auth (simple token-based)
# -------------------------


class LoginRequest(BaseModel):
    name: str  # we only send the username from frontend now

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
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return dep

# Preset accounts
PRESET_TESTERS = ["alex","bryan", "ge fan","jimmy", "kae","krish", "nicholas" ,  "sunny",  "yew meng", "yubo",  "zhen yang",  ]
PRESET_SUPERVISORS = ["kian siang", "alban","hai hong"]

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
    """
    Scheduler uses this to populate tester dropdown.
    Only supervisors may fetch tester list.
    """
    return PRESET_TESTERS

# -------------------------
# Static Test Steps
# -------------------------


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


'''
@app.get("/results/{result_id}/files", response_model=List[FileMetaOut])
def list_result_files(result_id: str, user: User = Depends(get_current_user)):
    res = RESULTS.get(result_id)
    if not res:
        raise HTTPException(status_code=404, detail="Result not found")

    out: List[FileMetaOut] = []
    for fid in res.files:
        fm = FILES.get(fid)
        if not fm:
            continue
        out.append(
            FileMetaOut(
                id=fm.id,
                unit_id=fm.unit_id,
                step_id=fm.step_id,
                result_id=fm.result_id,
                orig_name=fm.orig_name,
            )
        )
    return out

@app.get("/files/{file_id}")
def download_file(file_id: str, user: User = Depends(get_current_user)):
    fm = FILES.get(file_id)
    if not fm:
        raise HTTPException(status_code=404, detail="File not found")

    path = Path(fm.stored_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    return FileResponse(path, filename=fm.orig_name)

@app.delete("/files/{file_id}")
def delete_file(file_id: str, supervisor: User = Depends(require_role("supervisor"))):
    fm = FILES.get(file_id)
    if not fm:
        raise HTTPException(status_code=404, detail="File not found")

    res = RESULTS.get(fm.result_id)
    if res:
        res.files = [fid for fid in res.files if fid != file_id]
        RESULTS[fm.result_id] = res

    p = Path(fm.stored_path)
    try:
        p.unlink()
    except FileNotFoundError:
        pass

    del FILES[file_id]

    return {"ok": True}
'''
    
# -------------------------
# Units Endpoints
# -------------------------

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

    for idx, step_id in enumerate(STEP_IDS_ORDERED):
        a = Assignment(
            unit_id=unit_id,
            step_id=step_id,
            prev_passed=(idx == 0),
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
    This is idempotent: if the unit doesn't exist, we still return 200.
    Frontend (UnitsPage handleDelete) calls this.
    """
    # 1) Try to load the unit
    unit = session.get(Unit, unit_id)

    # If already missing, treat as "already deleted"
    if not unit:
        return {"ok": True, "deleted": False}

    # 2) Delete file metadata + physical files for this unit
    files = session.exec(
        select(FileMeta).where(FileMeta.unit_id == unit_id)
    ).all()
    for f in files:
        if f.stored_path:
            p = Path(f.stored_path)
            try:
                p.unlink()
            except FileNotFoundError:
                # file already gone, ignore
                pass
        session.delete(f)

    # 3) Delete results for this unit
    results = session.exec(
        select(Result).where(Result.unit_id == unit_id)
    ).all()
    for r in results:
        session.delete(r)

    # 4) Delete assignments for this unit
    assignments = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()
    for a in assignments:
        session.delete(a)

    # 5) Delete notifications for this unit (if Notification has unit_id)
    notes = session.exec(
        select(Notification).where(Notification.unit_id == unit_id)
    ).all()
    for n in notes:
        session.delete(n)

    # 6) Delete the unit itself
    session.delete(unit)
    session.commit()

    return {"ok": True, "deleted": True}

@app.get("/units/summary", response_model=List[UnitSummary])
def get_units_summary(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    units = session.exec(select(Unit)).all()
    summaries: List[UnitSummary] = []

    for u in units:
        assigns = session.exec(
            select(Assignment).where(Assignment.unit_id == u.id)
        ).all()
        assigns.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

        total_steps = len(assigns)

        results = session.exec(
            select(Result).where(Result.unit_id == u.id)
        ).all()
        result_map = {(r.unit_id, r.step_id): r for r in results}

        passed_steps = sum(
            1 for a in assigns
            if (u.id, a.step_id) in result_map and result_map[(u.id, a.step_id)].passed
        )
        progress = (passed_steps / total_steps * 100) if total_steps else 0.0

        next_step_id = None
        next_step_name = None
        for a in assigns:
            if (u.id, a.step_id) not in result_map:
                next_step_id = a.step_id
                next_step_name = STEP_BY_ID[a.step_id].name
                break

        if total_steps == 0:
            status = "EMPTY"
        elif len(result_map) == total_steps:
            status = "COMPLETED"
        else:
            status = "IN_PROGRESS"

        summaries.append(UnitSummary(
            unit_id=u.id,
            status=status,
            progress_percent=progress,
            passed_steps=passed_steps,
            total_steps=total_steps,
            next_step_id=next_step_id,
            next_step_name=next_step_name,
        ))

    return summaries

'''
@app.get("/units/{unit_id}/details", response_model=UnitDetails)
def get_unit_details(unit_id: str, user: User = Depends(get_current_user)):
    unit = UNITS.get(unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")
    assignments = [a for a in ASSIGNMENTS.values() if a.unit_id == unit_id]
    assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)
    results = [r for r in RESULTS.values() if r.unit_id == unit_id]
    return UnitDetails(unit=unit, assignments=assignments, results=results)
'''

# -------------------------
# Tester Queue & Upcoming
# -------------------------

class TesterTask(BaseModel):
    assignment: Assignment
    step: TestStep
    reasons_blocked: List[str] = []

class TesterQueueResponse(BaseModel):
    ready: List[TesterTask]
    blocked: List[TesterTask]

def environment_ok_stub(
    unit_id: str, step_id: int
) -> Tuple[bool, Optional[str]]:
    return True, None

def calibration_ok_stub(
    unit_id: str, step_id: int
) -> Tuple[bool, Optional[str]]:
    return True, None

'''
@app.get("/tester/queue", response_model=TesterQueueResponse)
def get_tester_queue(
    tester_id: str,
    user: User = Depends(get_current_user),
):
    ready: List[TesterTask] = []
    blocked: List[TesterTask] = []

    my_assignments: List[Assignment] = []
    for a in ASSIGNMENTS.values():
        if not (a.tester_id == tester_id or a.tester_id is None):
            continue
        if a.status not in ("PENDING", "RUNNING"):
            continue
        key = (a.unit_id, a.step_id)
        if key in RESULT_BY_UNIT_STEP:
            continue
        my_assignments.append(a)

    for a in my_assignments:
        reasons: List[str] = []

        if not a.prev_passed:
            reasons.append("Previous step not passed")

        ok_env, env_reason = environment_ok_stub(a.unit_id, a.step_id)
        if not ok_env:
            reasons.append(env_reason or "Environment out-of-range")

        ok_cal, cal_reason = calibration_ok_stub(a.unit_id, a.step_id)
        if not ok_cal:
            reasons.append(cal_reason or "Calibration expired")

        step = STEP_BY_ID[a.step_id]
        task = TesterTask(assignment=a, step=step, reasons_blocked=reasons)

        if reasons:
            blocked.append(task)
        else:
            ready.append(task)

    return TesterQueueResponse(ready=ready, blocked=blocked)

@app.get("/tester/assignments", response_model=List[Assignment])
def get_tester_assignments(
    tester_id: str,
    user: User = Depends(get_current_user),
):
    out = []

    for a in ASSIGNMENTS.values():
        # ignore done steps
        if a.status not in ("PENDING", "RUNNING"):
            continue

        # ignore steps with existing results
        if (a.unit_id, a.step_id) in RESULT_BY_UNIT_STEP:
            continue

        # must only allow steps whose previous step passed
        if not a.prev_passed:
            continue

        # tester can see:
        # - steps already assigned to them
        # - steps NOT assigned (unassigned = ready to test)
        if a.tester_id not in (None, tester_id):
            continue

        out.append(a)

    return out



# === NOTIFICATIONS: endpoints ========================

@app.get("/tester/notifications", response_model=List[Notification])
def get_tester_notifications(
    tester_id: str,
    unread_only: bool = False,
    user: User = Depends(get_current_user),
):
    """
    Return notifications for a tester.
    - Tester can only see their own.
    - Supervisor can see for any tester.
    Sorted by created_at (newest first).
    """
    if user.role == "tester" and user.name != tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    ids = TESTER_NOTIF_INDEX.get(tester_id, [])
    notes = [NOTIFICATIONS[nid] for nid in ids if nid in NOTIFICATIONS]

    if unread_only:
        notes = [n for n in notes if not n.read]

    # newest first
    notes.sort(key=lambda n: n.created_at, reverse=True)
    return notes


@app.post("/tester/notifications/{notif_id}/read")
def mark_notification_read(
    notif_id: str,
    user: User = Depends(get_current_user),
):
    """
    Mark a single notification as read.
    - Tester can only mark their own notifications.
    - Supervisor can mark any.
    """
    notif = NOTIFICATIONS.get(notif_id)
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")

    if user.role == "tester" and user.name != notif.tester_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    notif.read = True
    NOTIFICATIONS[notif_id] = notif
    return {"ok": True}
'''

# -------------------------
# Results & Uploads
# -------------------------

class ResultIn(BaseModel):
    unit_id: str
    step_id: int
    metrics: Optional[Dict[str, Any]] = None  # optional
    passed: bool  # tester must decide
    finished_at: Optional[datetime] = None    # <- NEW


class ResultOut(BaseModel):
    id: str
    unit_id: str
    step_id: int
    passed: bool
    metrics: Dict[str, Any]
    files: List[str]
    submitted_by: Optional[str]
    finished_at: datetime




# === NOTIFICATIONS: helper ==========================



@app.post("/results", response_model=ResultOut)
def create_or_update_result(
    body: ResultIn,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # ---- 1) Validate step exists ----
    if body.step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")

    # ---- 2) Validate unit exists ----
    unit = session.get(Unit, body.unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    unit_id = body.unit_id
    step_id = body.step_id
    passed = body.passed
    metrics = body.metrics or {}
    finished = body.finished_at or datetime.utcnow()

    # ---- 3) Idempotent result lookup by (unit_id, step_id) ----
    existing_result = session.exec(
        select(Result).where(
            Result.unit_id == unit_id,
            Result.step_id == step_id
        )
    ).first()

    if existing_result:
        # ---- 4a) Update existing result ----
        existing_result.metrics = metrics
        existing_result.passed = passed
        existing_result.finished_at = finished
        res = existing_result
    else:
        # ---- 4b) Create new result ----
        res = Result(
            unit_id=unit_id,
            step_id=step_id,
            passed=passed,
            metrics=metrics,
            files=[],                      # same as your old logic
            submitted_by=user.id,
            finished_at=finished,
        )
        session.add(res)

    # ---- 5) Whoever submits becomes tester for this step ----
    a = session.exec(
        select(Assignment).where(
            Assignment.unit_id == unit_id,
            Assignment.step_id == step_id
        )
    ).first()

    if a:
        a.tester_id = user.name
        a.status = "DONE"
        session.add(a)

    # ---- 6) Chain prev_passed to next step ----
    if step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_step_id = STEP_IDS_ORDERED[idx + 1]

            nxt = session.exec(
                select(Assignment).where(
                    Assignment.unit_id == unit_id,
                    Assignment.step_id == next_step_id
                )
            ).first()

            if nxt:
                nxt.prev_passed = passed
                session.add(nxt)

    # ---- 7) Update unit status if all assignments DONE ----
    assigns = session.exec(
        select(Assignment).where(Assignment.unit_id == unit_id)
    ).all()

    if assigns and all(x.status == "DONE" for x in assigns):
        unit.status = "COMPLETED"
        session.add(unit)

    session.commit()
    session.refresh(res)

    # ---- 8) If passed -> notify next tester (dedupe) ----
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
                # check duplicate notification
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

    return ResultOut(**res.dict())


'''
@app.delete("/results/{unit_id}/{step_id}")
def delete_result_for_step(
    unit_id: str,
    step_id: int,
    supervisor: User = Depends(require_role("supervisor")),
):
    key = (unit_id, step_id)
    rid = RESULT_BY_UNIT_STEP.get(key)
    if not rid:
        return {"ok": True, "deleted": False}

    # remove result
    del RESULTS[rid]
    del RESULT_BY_UNIT_STEP[key]

    # reset assignment
    for a in ASSIGNMENTS.values():
        if a.unit_id == unit_id and a.step_id == step_id:
            a.status = "PENDING"
            ASSIGNMENTS[a.id] = a
            break

    # also reset prev_passed of next step to False (since chain breaks)
    if step_id in STEP_IDS_ORDERED:
        idx = STEP_IDS_ORDERED.index(step_id)
        if idx + 1 < len(STEP_IDS_ORDERED):
            next_id = STEP_IDS_ORDERED[idx + 1]
            for a in ASSIGNMENTS.values():
                if a.unit_id == unit_id and a.step_id == next_id:
                    a.prev_passed = False
                    ASSIGNMENTS[a.id] = a
                    break

    return {"ok": True, "deleted": True}
'''

# Storage config
STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(exist_ok=True)

ALLOWED_EXT = {".zip", ".csv", ".pdf", ".png"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

def sha256_fileobj(fileobj) -> str:
    h = hashlib.sha256()
    while True:
        chunk = fileobj.read(8192)
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()

@app.post("/uploads")
async def upload_evidence(
    unit_id: str = Form(...),
    step_id: int = Form(...),
    result_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # ---- 1) Unit must exist ----
    unit = session.get(Unit, unit_id)
    if not unit:
        raise HTTPException(status_code=404, detail="Unit not found")

    # ---- 2) Step must exist ----
    if step_id not in STEP_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown step_id")

    # ---- 3) Result must exist ----
    res = session.get(Result, result_id)
    if not res:
        raise HTTPException(status_code=404, detail="Result not found")

    # ---- 4) Safety: result must match unit+step ----
    if res.unit_id != unit_id or res.step_id != step_id:
        raise HTTPException(
            status_code=400,
            detail="result_id does not match unit/step"
        )

    # ---- 5) file extension check ----
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    content = await file.read()
    size = len(content)
    if size > MAX_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large")

    # ---- 6) hash for dedupe ----
    sha = hashlib.sha256(content).hexdigest()

    # ---- 7) dedupe search in DB ----
    existing = session.exec(
        select(FileMeta).where(
            FileMeta.sha256 == sha,
            FileMeta.unit_id == unit_id,
            FileMeta.step_id == step_id
        )
    ).first()

    if existing:
        if existing.id not in res.files:
            res.files.append(existing.id)
            session.add(res)
            session.commit()
        return {"file_id": existing.id, "deduplicated": True}

    # ---- 8) new file store to disk ----
    bucket = sha[:2]
    bucket_dir = STORAGE_ROOT / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:6]
    server_name = f"{unit_id}_{step_id}_{timestamp}_{unique_suffix}{ext}"
    stored_path = bucket_dir / server_name

    with open(stored_path, "wb") as f:
        f.write(content)

    # ---- 9) create FileMeta row ----
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

    # ---- 10) attach to Result.files ----
    res.files.append(meta.id)
    session.add(res)
    session.commit()

    return {"file_id": meta.id, "deduplicated": False}



# -------------------------
# Scheduling (Supervisor)
# -------------------------


class DuplicateRequest(BaseModel):
    source_unit_id: str
    new_unit_ids: List[str]
    day_shift: int = 1

'''
@app.post("/schedule/duplicate")
def duplicate_schedule(
    body: DuplicateRequest,
    supervisor: User = Depends(require_role("supervisor"))
):
    src = body.source_unit_id

    if src not in UNITS:
        raise HTTPException(status_code=404, detail="Source unit not found")

    src_assignments = [
        a for a in ASSIGNMENTS.values() if a.unit_id == src
    ]
    src_assignments.sort(key=lambda a: STEP_BY_ID[a.step_id].order)

    created_units = []

    for new_unit in body.new_unit_ids:
        if new_unit in UNITS:
            raise HTTPException(status_code=400, detail=f"Unit {new_unit} already exists")

        UNITS[new_unit] = Unit(
            id=new_unit,
            sku=UNITS[src].sku,
            rev=UNITS[src].rev,
            lot=UNITS[src].lot,
            status="IN_PROGRESS",
            current_step_id=STEP_IDS_ORDERED[0],
        )

        for src_a in src_assignments:

            def shift(dt):
                if not dt:
                    return None
                return dt + timedelta(days=body.day_shift)

            new_a = Assignment(
                id=str(uuid4()),
                unit_id=new_unit,
                step_id=src_a.step_id,
                tester_id=src_a.tester_id,
                start_at=shift(src_a.start_at),
                end_at=shift(src_a.end_at),
                status="PENDING",
                prev_passed=(src_a.step_id == STEP_IDS_ORDERED[0])
            )

            ASSIGNMENTS[new_a.id] = new_a

        created_units.append(new_unit)

    return {"ok": True, "created_units": created_units}
'''

class AssignmentPatch(BaseModel):
    tester_id: Optional[str] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    status: Optional[str] = None


def overlaps(
    a_start: Optional[datetime],
    a_end: Optional[datetime],
    b_start: Optional[datetime],
    b_end: Optional[datetime],
) -> bool:
    """
    Check if two date ranges overlap.
    We treat them as full-day intervals, so any intersection counts as overlap.
    """
    if not a_start or not a_end or not b_start or not b_end:
        return False
    # inclusive overlap: [a_start, a_end] vs [b_start, b_end]
    return (a_start <= b_end) and (b_start <= a_end)

@app.get("/assignments/schedule", response_model=List[Assignment])
def get_schedule(
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Return all assignments for the scheduler page.
    Frontend uses this to render the Gantt-like view.
    """
    assignments = session.exec(select(Assignment)).all()
    # Optional: sort by unit then by step order
    assignments.sort(key=lambda a: (a.unit_id, STEP_BY_ID[a.step_id].order))
    return assignments


@app.patch("/assignments/{assign_id}", response_model=Assignment)
def patch_assignment(
    assign_id: str,
    body: AssignmentPatch,
    supervisor: User = Depends(require_role("supervisor")),
    session: Session = Depends(get_session),
):
    """
    Update a single assignment (tester, start/end dates, status).
    Used by SchedulerPage.
    """
    a = session.get(Assignment, assign_id)
    if not a:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # New values (dates only, but still stored as datetime)
    new_tester = body.tester_id if body.tester_id is not None else a.tester_id
    new_start = body.start_at if body.start_at is not None else a.start_at
    new_end = body.end_at if body.end_at is not None else a.end_at

    # Basic sanity: end cannot be before start
    if new_start and new_end and new_end < new_start:
        raise HTTPException(
            status_code=400,
            detail="End date cannot be before start date",
        )

    # Prevent overlapping tests for the same unit
    others = session.exec(
        select(Assignment).where(Assignment.unit_id == a.unit_id)
    ).all()

    for other in others:
        if other.id == a.id:
            continue
        if overlaps(new_start, new_end, other.start_at, other.end_at):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unit '{a.unit_id}' already has another test scheduled "
                    f"from {other.start_at} to {other.end_at}"
                ),
            )

    # Apply changes
    if body.tester_id is not None:
        a.tester_id = body.tester_id
    if body.start_at is not None:
        a.start_at = body.start_at
    if body.end_at is not None:
        a.end_at = body.end_at
    if body.status is not None:
        a.status = body.status

    session.add(a)
    session.commit()
    session.refresh(a)
    return a
'''
@app.get("/assignments/schedule", response_model=List[Assignment])
def get_schedule(supervisor: User = Depends(require_role("supervisor"))):
    # Just return all assignments; frontend can compute conflicts or show Gantt
    return list(ASSIGNMENTS.values())


@app.patch("/assignments/{assign_id}", response_model=Assignment)
def patch_assignment(
    assign_id: str,
    body: AssignmentPatch,
    supervisor: User = Depends(require_role("supervisor")),
):
    a = ASSIGNMENTS.get(assign_id)
    if not a:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # New values (dates only, but still stored as datetime)
    new_tester = body.tester_id if body.tester_id is not None else a.tester_id
    new_start = body.start_at if body.start_at is not None else a.start_at
    new_end = body.end_at if body.end_at is not None else a.end_at

    # Basic sanity: end cannot be before start
    if new_start and new_end and new_end < new_start:
        raise HTTPException(
            status_code=400,
            detail="End date cannot be before start date",
        )

    # (ONLY) prevent the **same unit** from having overlapping tests.
    # Tester is allowed to have multiple tests on the same day.
    for other in ASSIGNMENTS.values():
        if other.id == assign_id:
            continue
        if other.unit_id != a.unit_id:
            continue

        if overlaps(new_start, new_end, other.start_at, other.end_at):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unit '{a.unit_id}' already has another test scheduled "
                    f"from {other.start_at} to {other.end_at}"
                ),
            )

    # Apply changes
    if body.tester_id is not None:
        a.tester_id = body.tester_id
    if body.start_at is not None:
        a.start_at = body.start_at
    if body.end_at is not None:
        a.end_at = body.end_at
    if body.status is not None:
        a.status = body.status

    ASSIGNMENTS[assign_id] = a
    return a

@app.post("/steps/sync")
def sync_steps(supervisor: User = Depends(require_role("supervisor"))):
    for unit_id in UNITS.keys():
        existing_step_ids = {a.step_id for a in ASSIGNMENTS.values() if a.unit_id == unit_id}

        for idx, step_id in enumerate(STEP_IDS_ORDERED):
            if step_id in existing_step_ids:
                continue

            new_a = Assignment(
                id=str(uuid4()),
                unit_id=unit_id,
                step_id=step_id,
                prev_passed=(step_id == STEP_IDS_ORDERED[0]),
            )
            ASSIGNMENTS[new_a.id] = new_a

    return {"ok": True}
'''

# -------------------------
# Evidence Export (ZIP)
# -------------------------

ZIP_ROOT = Path("zips")
ZIP_ROOT.mkdir(exist_ok=True)

def step_folder_name(unit_id: str, step_id: int) -> str:
    step = STEP_BY_ID.get(step_id)
    if step:
        order = step.order
        name = step.name
    else:
        order = step_id
        name = f"Step {step_id}"
    return f"{order}. {unit_id}#{name}"

def sanitize_step_folder(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_")
    name = name.strip().strip(".")
    return name or "step"

'''
@app.get("/reports/unit/{unit_id}/zip")
def export_unit_zip(
    unit_id: str,
    user: User = Depends(get_current_user),
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")

    unit_results = [r for r in RESULTS.values() if r.unit_id == unit_id]
    unit_files = [f for f in FILES.values() if f.unit_id == unit_id]

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_name = f"{unit_id}_logs_{timestamp}.zip"
    zip_path = ZIP_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        results_json = [r.dict() for r in unit_results]
        zf.writestr("results.json", json.dumps(results_json, default=str, indent=2))

        lines = []
        for f in unit_files:
            lines.append(
                f"{f.id} | {f.unit_id} | {f.step_id} | {f.orig_name} | {f.stored_name}"
            )
        zf.writestr("manifest.txt", "\n".join(lines))

        for f in unit_files:
            path = Path(f.stored_path)
            if not path.exists():
                continue

            step = STEP_BY_ID.get(f.step_id)
            if step:
                folder_name = sanitize_step_folder(f"{step.order}. {unit_id}#{step.name}")
            else:
                folder_name = f"step_{f.step_id}"

            arcname = os.path.join(folder_name, f.orig_name)
            zf.write(path, arcname=arcname)

    return FileResponse(
        path=zip_path,
        filename=zip_name,
        media_type="application/zip",
    )

@app.get("/reports/unit/{unit_id}/step/{step_id}/zip")
def export_step_zip(
    unit_id: str,
    step_id: int,
    user: User = Depends(),
):
    if unit_id not in UNITS:
        raise HTTPException(status_code=404, detail="Unit not found")
    if step_id not in STEP_BY_ID:
        raise HTTPException(status_code=404, detail="Step not found")

    step_files = [
        f
        for f in FILES.values()
        if f.unit_id == unit_id and f.step_id == step_id
    ]
    if not step_files:
        raise HTTPException(status_code=404, detail="No files for this step")

    folder = step_folder_name(unit_id, step_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_name = f"{folder}_logs_{timestamp}.zip"
    zip_path = ZIP_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in step_files:
            path = Path(f.stored_path)
            if not path.exists():
                continue
            arcname = os.path.join(folder, f.orig_name)
            zf.write(path, arcname=arcname)

    return FileResponse(
        path=zip_path,
        filename=zip_name,
        media_type="application/zip",
    )
'''

# -------------------------
# Root
# -------------------------

@app.get("/")
def root():
    return {"message": "Testing Unit Tracker API running"}











