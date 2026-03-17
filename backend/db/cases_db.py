"""
db/cases_db.py — Cases + Referrals + Patient Passports
"""
import sqlite3, json, uuid, secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path("database/retina_cases.db")

def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY, patient_id TEXT DEFAULT 'Unknown',
            created_at TEXT NOT NULL, image_name TEXT,
            dr_grade INTEGER, dr_label TEXT, dr_confidence REAL, dr_refer INTEGER,
            quality_score REAL, quality_adequate INTEGER, risk_level TEXT,
            full_result TEXT NOT NULL, status TEXT DEFAULT 'completed')""")

        conn.execute("""CREATE TABLE IF NOT EXISTS referrals (
            id TEXT PRIMARY KEY, case_id TEXT NOT NULL, patient_id TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            referring_dr TEXT DEFAULT '', specialist TEXT DEFAULT '',
            clinic TEXT DEFAULT '', reason TEXT DEFAULT '',
            urgency TEXT DEFAULT 'routine', status TEXT DEFAULT 'pending',
            notes TEXT DEFAULT '', outcome TEXT DEFAULT '',
            dr_grade INTEGER, dr_label TEXT,
            FOREIGN KEY (case_id) REFERENCES cases(id))""")

        conn.execute("""CREATE TABLE IF NOT EXISTS passports (
            token TEXT PRIMARY KEY, case_id TEXT NOT NULL,
            patient_id TEXT NOT NULL, created_at TEXT NOT NULL,
            expires_at TEXT, views INTEGER DEFAULT 0, active INTEGER DEFAULT 1,
            FOREIGN KEY (case_id) REFERENCES cases(id))""")

        conn.commit()

# ── Cases ──────────────────────────────────────────────────────────────────

def save_case(result_dict: Dict, patient_id: str = "Unknown", image_name: str = "") -> str:
    init_db()
    case_id = result_dict.get("image_id") or str(uuid.uuid4())[:8]
    dr = result_dict.get("dr_grading", {})
    quality = result_dict.get("quality", {})
    with get_connection() as conn:
        conn.execute("""INSERT OR REPLACE INTO cases
            (id,patient_id,created_at,image_name,dr_grade,dr_label,
             dr_confidence,dr_refer,quality_score,quality_adequate,full_result,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (case_id, patient_id, datetime.utcnow().isoformat(), image_name,
             dr.get("grade",-1), dr.get("label",""),
             dr.get("confidence",0.0), 1 if dr.get("refer") else 0,
             quality.get("score",0.0), 1 if quality.get("adequate",True) else 0,
             json.dumps(result_dict), "completed"))
        conn.commit()
    return case_id

def get_cases(limit=50, offset=0, patient_id=None, dr_grade=None, refer_only=False):
    init_db()
    q, p = "SELECT * FROM cases WHERE 1=1", []
    if patient_id: q += " AND patient_id LIKE ?"; p.append(f"%{patient_id}%")
    if dr_grade is not None: q += " AND dr_grade = ?"; p.append(dr_grade)
    if refer_only: q += " AND dr_refer = 1"
    q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"; p.extend([limit, offset])
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(q, p).fetchall()]

def get_case(case_id: str) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if not row: return None
    c = dict(row); c["full_result"] = json.loads(c["full_result"]); return c

def get_stats() -> Dict:
    init_db()
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        today = conn.execute("SELECT COUNT(*) FROM cases WHERE created_at >= date('now')").fetchone()[0]
        week  = conn.execute("SELECT COUNT(*) FROM cases WHERE created_at >= date('now','-7 days')").fetchone()[0]
        ref   = conn.execute("SELECT COUNT(*) FROM cases WHERE dr_refer = 1").fetchone()[0]
        dist  = conn.execute("SELECT dr_grade, COUNT(*) FROM cases GROUP BY dr_grade").fetchall()
    return {"total_cases":total,"today":today,"this_week":week,"referable_cases":ref,
            "dr_grade_distribution":{str(r[0]):r[1] for r in dist if r[0]>=0}}

def delete_case(case_id: str) -> bool:
    init_db()
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM cases WHERE id = ?", (case_id,)); conn.commit()
        return cur.rowcount > 0

# ── Referrals ──────────────────────────────────────────────────────────────

REFERRAL_STATUSES = ["pending","sent","acknowledged","seen","completed","cancelled"]
URGENCY_LEVELS    = ["urgent","priority","routine"]

def create_referral(case_id:str, patient_id:str, referring_dr:str="",
                    specialist:str="", clinic:str="", reason:str="",
                    urgency:str="routine", notes:str="",
                    dr_grade:Optional[int]=None, dr_label:str="") -> Dict:
    init_db()
    rid = f"ref-{str(uuid.uuid4())[:8]}"
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""INSERT INTO referrals
            (id,case_id,patient_id,created_at,updated_at,
             referring_dr,specialist,clinic,reason,urgency,status,notes,dr_grade,dr_label)
            VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?,?)""",
            (rid,case_id,patient_id,now,now,
             referring_dr,specialist,clinic,reason,urgency,notes,dr_grade,dr_label))
        conn.commit()
    return get_referral(rid)

def get_referral(rid: str) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM referrals WHERE id = ?", (rid,)).fetchone()
    return dict(row) if row else None

def update_referral(rid:str, status:str, notes:str="", outcome:str="") -> Optional[Dict]:
    if status not in REFERRAL_STATUSES: raise ValueError(f"Invalid status: {status}")
    with get_connection() as conn:
        conn.execute("""UPDATE referrals SET status=?,updated_at=?,
            notes=CASE WHEN ?!='' THEN ? ELSE notes END,
            outcome=CASE WHEN ?!='' THEN ? ELSE outcome END WHERE id=?""",
            (status, datetime.utcnow().isoformat(), notes,notes, outcome,outcome, rid))
        conn.commit()
    return get_referral(rid)

def get_referrals(patient_id=None, case_id=None, status=None, limit=50) -> List[Dict]:
    init_db()
    q, p = "SELECT * FROM referrals WHERE 1=1", []
    if patient_id: q += " AND patient_id LIKE ?"; p.append(f"%{patient_id}%")
    if case_id:    q += " AND case_id = ?";       p.append(case_id)
    if status:     q += " AND status = ?";         p.append(status)
    q += " ORDER BY created_at DESC LIMIT ?"; p.append(limit)
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(q, p).fetchall()]

def get_referral_stats() -> Dict:
    init_db()
    with get_connection() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM referrals").fetchone()[0]
        by_s    = conn.execute("SELECT status,COUNT(*) FROM referrals GROUP BY status").fetchall()
        urgent  = conn.execute("SELECT COUNT(*) FROM referrals WHERE urgency='urgent' AND status NOT IN ('completed','cancelled')").fetchone()[0]
    return {"total":total,"urgent_open":urgent,"by_status":{r[0]:r[1] for r in by_s}}

# ── Patient Passports ──────────────────────────────────────────────────────

def create_passport(case_id:str, patient_id:str, expires_days:Optional[int]=None) -> str:
    init_db()
    token   = secrets.token_urlsafe(20)
    expires = (datetime.utcnow()+timedelta(days=expires_days)).isoformat() if expires_days else None
    with get_connection() as conn:
        conn.execute("INSERT INTO passports (token,case_id,patient_id,created_at,expires_at,views,active) VALUES (?,?,?,?,?,0,1)",
                     (token,case_id,patient_id,datetime.utcnow().isoformat(),expires))
        conn.commit()
    return token

def get_passport(token:str) -> Optional[Dict]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM passports WHERE token=? AND active=1",(token,)).fetchone()
        if not row: return None
        passport = dict(row)
        if passport.get("expires_at") and datetime.fromisoformat(passport["expires_at"]) < datetime.utcnow():
            return None
        conn.execute("UPDATE passports SET views=views+1 WHERE token=?",(token,)); conn.commit()
    case = get_case(passport["case_id"])
    return {"passport":passport,"case":case} if case else None

def revoke_passport(token:str) -> bool:
    init_db()
    with get_connection() as conn:
        cur = conn.execute("UPDATE passports SET active=0 WHERE token=?",(token,)); conn.commit()
        return cur.rowcount > 0

def get_passports_for_case(case_id:str) -> List[Dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM passports WHERE case_id=? ORDER BY created_at DESC",(case_id,)).fetchall()
    return [dict(r) for r in rows]
