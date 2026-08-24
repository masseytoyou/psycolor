
import os
import uuid
import hashlib
import hmac
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="Psycolor User", page_icon="🧠", layout="wide")

INDEX_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=0"
SUBTEST_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=978787284"
MODEL_NAME = "gpt-4o-mini"

ROLE_LABELS = {
    "general": "일반 이용자",
    "senior": "시니어 이용자",
    "expert": "전문가",
    "admin": "관리자",
}

# 스트림릿 버전에 따라 st.fragment / st.experimental_fragment 중 있는 것을 사용.
# fragment로 감싼 함수는 그 안의 위젯을 조작할 때 앱 전체가 아니라 그 부분만 재실행되어
# 다른 탭의 DB 조회 등이 매번 다시 실행되는 걸 막아줌 (버튼 누를 때 화면 전체가
# 흐려지는 현상의 주 원인).
_fragment_decorator = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if _fragment_decorator is None:
    def _fragment_decorator(func):
        return func

SELECTION = {
    "K-WPPSI-IV_A": {
        "VCI": ["RV", "PN"],
        "PSI": ["BD", "OA"],
        "FSIQ": []
    },
    "K-WPPSI-IV_B": {
        "VCI": ["VC", "IN", "SI", "CO", "RV", "PN"],
        "VSI": ["BD", "OA"],
        "FRI": ["MR", "PC"],
        "WMI": ["PM", "ZL"],
        "PSI": ["BS", "CA", "CAR", "CAS", "AC"],
        "FSIQ": [],
    },
    "K-WISC-V": {
        "VCI": ["SI", "VC", "IN", "CO"],
        "VSI": ["BD", "VP"],
        "WMI": ["DS", "AR", "LN"],
        "PSI": ["CD", "SS", "CA"],
        "FSIQ": [],
    },
    "K-WAIS-IV": {
        "VCI": ["SI", "VC", "IN", "CO"],
        "PRI": ["BD", "MR", "VP", "FW", "PCm"],
        "WMI": ["DS", "AR", "LN"],
        "PSI": ["CD", "SS", "CA"],
        "FSIQ": [],
    },
}


# =========================================================
# 공통 유틸
# =========================================================
def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_text(value: Optional[str]) -> str:
    return (value or "").strip()


def get_api_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def get_db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다.")


def get_connection():
    return get_connection_pool().getconn()


def release_connection(conn) -> None:
    """conn.close() 대체 함수: 커넥션을 실제로 끊지 않고 풀에 반납합니다."""
    try:
        get_connection_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


@st.cache_resource
def get_connection_pool():
    """DB 커넥션 풀. 앱 세션 전체에서 재사용되며 매 인터랙션마다 새로 연결하지 않습니다."""
    return psycopg2.pool.SimpleConnectionPool(1, 10, get_db_url())


def password_hash(password: str) -> str:
    salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}${hashed.hex()}"


def password_verify(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        current = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(current, expected)
    except Exception:
        return False


def role_badge(role: str) -> str:
    if role == "expert":
        return "전문가"
    if role == "admin":
        return "관리자"
    if role == "senior":
        return "시니어"
    return "일반"


def init_session_state() -> None:
    defaults = {
        "logged_in": False,
        "user_id": None,
        "username": "",
        "nickname": "",
        "role": "",
        "last_generated_test_id": None,
        "last_generated_report": "",
        "last_generated_prompt": "",
        "last_generated_test_type": "",
        "last_generated_pdf": None,
        "last_generated_txt": None,
        "pending_payment": False,
        "payment_notice_ack": False,
        "dash_history_df": None,
        "dash_radar_png": None,
        "dash_trend_png": None,
        "dash_pdf_with_chart": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout() -> None:
    for key in [
        "logged_in", "user_id", "username", "nickname", "role",
        "last_generated_test_id", "last_generated_report",
        "last_generated_prompt", "last_generated_test_type",
        "last_generated_pdf", "last_generated_txt",
        "last_generated_index_scores", "last_generated_profile_png",
        "pending_payment", "payment_notice_ack",
        "dash_history_df", "dash_radar_png", "dash_trend_png", "dash_pdf_with_chart",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def normalize_binary_data(value):
    if value is None:
        return None
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    return None


# =========================================================
# Lookup / 보고서 생성 유틸
# =========================================================
@st.cache_data
def load_lookup_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_i = pd.read_csv(INDEX_CSV_URL)
    df_s = pd.read_csv(SUBTEST_CSV_URL)
    return df_i, df_s


@st.cache_data
def get_test_frames(test_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_i, df_s = load_lookup_tables()
    return (
        df_i[df_i["test_type"] == test_type].copy(),
        df_s[df_s["test_type"] == test_type].copy(),
    )


def flatten_subtests(test_type: str) -> List[str]:
    subtests: List[str] = []
    for items in SELECTION[test_type].values():
        subtests.extend(items)
    return subtests


def put_index_cla_and_com(test_type: str, index_scores: Dict[str, int]) -> Dict[str, Dict[str, str]]:
    index_df, _ = get_test_frames(test_type)
    result: Dict[str, Dict[str, str]] = {}

    for index_code, score in index_scores.items():
        matched_rows = index_df[index_df["index_code"] == index_code]
        for _, row in matched_rows.iterrows():
            if row["min_composite_score"] <= score <= row["max_composite_score"]:
                result[index_code] = {
                    str(row["classification"]): str(row["comment"])
                }
                break
    return result


def put_subtest_cla_and_com(test_type: str, subtest_scores: Dict[str, int]) -> Dict[str, Dict[str, str]]:
    _, subtest_df = get_test_frames(test_type)
    result: Dict[str, Dict[str, str]] = {}

    for subtest_code, score in subtest_scores.items():
        matched_rows = subtest_df[subtest_df["subtest_code"] == subtest_code]
        for _, row in matched_rows.iterrows():
            if row["min_scaled_score"] <= score <= row["max_scaled_score"]:
                result[subtest_code] = {
                    str(row["classification"]): str(row["comment"])
                }
                break
    return result


def build_prompt(
    test_type: str,
    index_cla_com: Dict[str, Dict[str, str]],
    subtest_cla_com: Dict[str, Dict[str, str]],
    examinee_info: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append("너는 심리검사 보고서 문장 정리 도우미다.")
    lines.append("반드시 제공된 정보만 사용하라.")
    lines.append("없는 사실을 추론하지 마라.")
    lines.append("진단명이나 치료 권고를 임의로 추가하지 마라.")
    lines.append("공식적이고 자연스러운 한국어 보고서 문체로 작성하라.")
    lines.append(f"검사 유형은 {test_type}이다.")
    lines.append("수검자 정보는 참고하되, 입력되지 않은 항목은 언급하지 마라.")
    lines.append("지표 결과와 소검사 결과는 각각 다른 문단으로 작성하라.")
    lines.append("")
    lines.append("[수검자 정보]")
    for k, v in examinee_info.items():
        if safe_text(v):
            lines.append(f"- {k}: {safe_text(v)}")
    lines.append("")
    lines.append("[지표 결과]")
    for key, value in index_cla_com.items():
        cla, com = next(iter(value.items()))
        lines.append(f"- {key}: {cla} / {com}")

    lines.append("")
    lines.append("[소검사 결과]")
    for key, value in subtest_cla_com.items():
        cla, com = next(iter(value.items()))
        lines.append(f"- {key}: {cla} / {com}")

    lines.append("")
    lines.append("위 정보를 바탕으로 전체 결과를 3000자 이내의 자연스러운 한국어 보고서 문단 2개로 작성하라.")
    lines.append("첫 번째 문단은 지표 결과 중심, 두 번째 문단은 소검사 결과 중심으로 작성하라.")
    lines.append("단, '***' 같은 기호는 사용하지 말 것.")
    return "\n".join(lines)


def generate_report(prompt: str, model_name: str = MODEL_NAME) -> str:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    return response.output_text


def validate_scores(index_scores: Dict[str, int], subtest_scores: Dict[str, int]) -> List[str]:
    errors: List[str] = []

    for key, value in index_scores.items():
        if not (40 <= value <= 200):
            errors.append(f"지표점수 {key}는 40~200 사이여야 합니다.")

    for key, value in subtest_scores.items():
        if not (1 <= value <= 19):
            errors.append(f"환산점수 {key}는 1~19 사이여야 합니다.")

    return errors


# =========================================================
# 다운로드 유틸
# =========================================================
def make_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def make_pdf_bytes(title: str, lines: List[str], chart_images: Optional[List[bytes]] = None) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))

    width, height = A4
    left_margin = 50
    top_margin = 50
    bottom_margin = 50
    y = height - top_margin
    line_height = 18
    usable_width = width - (left_margin * 2)

    def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
        text = str(text)
        if not text:
            return [""]

        wrapped_lines: List[str] = []
        current = ""

        for ch in text:
            trial = current + ch
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                if current:
                    wrapped_lines.append(current)
                current = ch

        if current:
            wrapped_lines.append(current)

        return wrapped_lines or [""]

    c.setFont("HYSMyeongJo-Medium", 14)
    c.drawString(left_margin, y, title)
    y -= 30

    c.setFont("HYSMyeongJo-Medium", 11)
    for raw_line in lines:
        split_lines = str(raw_line).splitlines() or [""]
        for split_line in split_lines:
            wrapped = wrap_text(split_line, "HYSMyeongJo-Medium", 11, usable_width)
            for line in wrapped:
                if y < bottom_margin:
                    c.showPage()
                    c.setFont("HYSMyeongJo-Medium", 11)
                    y = height - top_margin

                c.drawString(left_margin, y, line)
                y -= line_height

    if chart_images:
        for img_bytes in chart_images:
            if not img_bytes:
                continue
            c.showPage()
            img = ImageReader(BytesIO(img_bytes))
            img_w, img_h = img.getSize()
            aspect = img_h / img_w

            draw_w = usable_width
            draw_h = draw_w * aspect
            max_h = height - top_margin - bottom_margin
            if draw_h > max_h:
                draw_h = max_h
                draw_w = draw_h / aspect

            x = left_margin + (usable_width - draw_w) / 2
            y_img = height - top_margin - draw_h
            c.drawImage(img, x, y_img, width=draw_w, height=draw_h)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# =========================================================
# DB 초기화
# =========================================================
def table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = %s
            )
            """,
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def column_exists(conn, table_name: str, column_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                  AND column_name = %s
            )
            """,
            (table_name, column_name),
        )
        return bool(cur.fetchone()[0])


def ensure_users_table(conn) -> None:
    with conn.cursor() as cur:
        if table_exists(conn, "users") and not column_exists(conn, "users", "user_id"):
            backup_name = f"users_legacy_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cur.execute(f'ALTER TABLE users RENAME TO {backup_name}')

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGSERIAL PRIMARY KEY,
            username TEXT,
            password_hash TEXT,
            role TEXT,
            nickname TEXT,
            created_at TEXT
        )
        """)

        required_columns = {
            "username": "TEXT",
            "password_hash": "TEXT",
            "role": "TEXT",
            "nickname": "TEXT",
            "created_at": "TEXT",
        }
        for col_name, col_type in required_columns.items():
            if not column_exists(conn, "users", col_name):
                cur.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")

        cur.execute("UPDATE users SET created_at = COALESCE(created_at, %s)", (now_str(),))
        cur.execute("UPDATE users SET role = COALESCE(role, 'general')")

        cur.execute("ALTER TABLE users ALTER COLUMN username SET NOT NULL")
        cur.execute("ALTER TABLE users ALTER COLUMN password_hash SET NOT NULL")
        cur.execute("ALTER TABLE users ALTER COLUMN role SET NOT NULL")
        cur.execute("ALTER TABLE users ALTER COLUMN created_at SET NOT NULL")
        cur.execute("ALTER TABLE users ALTER COLUMN nickname DROP NOT NULL")

        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'users_role_check'
                ) THEN
                    ALTER TABLE users
                    ADD CONSTRAINT users_role_check
                    CHECK (role IN ('general', 'senior', 'expert', 'admin'));
                END IF;
            END $$;
            """
        )

        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_unique ON users(username)")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_nickname_unique ON users(nickname) WHERE nickname IS NOT NULL")


def init_db() -> None:
    conn = get_connection()
    lock_key = 20260331
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (lock_key,))

        ensure_users_table(conn)

        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_run (
                test_id TEXT PRIMARY KEY,
                expert_user_id BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
                test_type TEXT NOT NULL,
                examinee_name TEXT,
                date_of_birth TEXT,
                sex TEXT,
                examiner TEXT,
                test_date TEXT,
                created_at TEXT NOT NULL
            )
            """)

            if not column_exists(conn, "test_run", "expert_user_id"):
                cur.execute("ALTER TABLE test_run ADD COLUMN expert_user_id BIGINT")

            cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'fk_test_run_expert_user'
                ) THEN
                    ALTER TABLE test_run
                    ADD CONSTRAINT fk_test_run_expert_user
                    FOREIGN KEY (expert_user_id) REFERENCES users(user_id) ON DELETE SET NULL;
                END IF;
            END $$;
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_result (
                result_id BIGSERIAL PRIMARY KEY,
                test_id TEXT NOT NULL REFERENCES test_run(test_id) ON DELETE CASCADE,
                result_type TEXT NOT NULL,
                result_name TEXT NOT NULL,
                raw_score INTEGER NOT NULL,
                classification TEXT NOT NULL,
                comment TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS final_report (
                report_id BIGSERIAL PRIMARY KEY,
                test_id TEXT NOT NULL UNIQUE REFERENCES test_run(test_id) ON DELETE CASCADE,
                prompt TEXT,
                final_report TEXT NOT NULL,
                model_name TEXT,
                created_at TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS community_post (
                post_id BIGSERIAL PRIMARY KEY,
                board_type TEXT NOT NULL CHECK (board_type IN ('public', 'anonymous')),
                author_user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                image_bytes BYTEA,
                image_name TEXT,
                image_mime TEXT,
                approval_status TEXT NOT NULL CHECK (approval_status IN ('pending', 'approved', 'rejected')),
                approved_by BIGINT REFERENCES users(user_id) ON DELETE SET NULL,
                approved_at TEXT,
                created_at TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS post_like (
                like_id BIGSERIAL PRIMARY KEY,
                post_id BIGINT NOT NULL REFERENCES community_post(post_id) ON DELETE CASCADE,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                UNIQUE (post_id, user_id)
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS post_comment (
                comment_id BIGSERIAL PRIMARY KEY,
                post_id BIGINT NOT NULL REFERENCES community_post(post_id) ON DELETE CASCADE,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS inbox_message (
                message_id BIGSERIAL PRIMARY KEY,
                sender_user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                receiver_user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                message_text TEXT,
                file_name TEXT,
                file_mime TEXT,
                file_bytes BYTEA,
                created_at TEXT NOT NULL
            )
            """)

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        except Exception:
            pass
        release_connection(conn)


# =========================================================
# 사용자 / 인증
# =========================================================
def create_user(username: str, password: str, role: str, nickname: Optional[str] = None) -> None:
    username = safe_text(username)
    nickname = safe_text(nickname)
    hashed = password_hash(password)

    if role != "admin" and not nickname:
        raise ValueError("닉네임은 필수입니다.")

    if role == "admin":
        nickname = None

    conn = get_connection()
    cur = conn.cursor()
    try:
        if nickname:
            cur.execute("SELECT 1 FROM users WHERE nickname = %s", (nickname,))
            if cur.fetchone():
                raise ValueError("이미 사용 중인 닉네임입니다.")

        cur.execute("""
        INSERT INTO users (username, password_hash, role, nickname, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """, (username, hashed, role, nickname if nickname else None, now_str()))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_connection(conn)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, str]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT user_id, username, password_hash, role, COALESCE(nickname, '')
    FROM users
    WHERE username = %s
    """, (safe_text(username),))
    row = cur.fetchone()
    cur.close()
    release_connection(conn)

    if not row:
        return None

    user_id, username_db, pw_hash, role, nickname = row
    if not password_verify(password, pw_hash):
        return None

    return {
        "user_id": user_id,
        "username": username_db,
        "role": role,
        "nickname": nickname,
    }


def get_user_by_nickname(nickname: str) -> Optional[Dict[str, str]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    SELECT user_id, username, role, COALESCE(nickname, '')
    FROM users
    WHERE nickname = %s
    """, (safe_text(nickname),))
    row = cur.fetchone()
    cur.close()
    release_connection(conn)

    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "role": row[2],
        "nickname": row[3],
    }


def search_general_users_by_nickname(keyword: str) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT user_id, username, nickname, created_at
    FROM users
    WHERE role = 'general'
      AND nickname ILIKE %s
    ORDER BY nickname ASC
    LIMIT 20
    """
    df = pd.read_sql_query(query, conn, params=(f"%{safe_text(keyword)}%",))
    release_connection(conn)
    return df


# =========================================================
# 보고서 저장 / 조회
# =========================================================
def save_test_run(
    expert_user_id: int,
    test_type: str,
    examinee_info: Dict[str, str],
    index_scores: Dict[str, int],
    subtest_scores: Dict[str, int],
    index_cla_com: Dict[str, Dict[str, str]],
    subtest_cla_com: Dict[str, Dict[str, str]],
    prompt: str,
    final_report_text: str,
    model_name: str = MODEL_NAME,
) -> str:
    test_id = uuid.uuid4().hex
    now = now_str()

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
        INSERT INTO test_run (
            test_id, expert_user_id, test_type, examinee_name, date_of_birth, sex, examiner, test_date, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            test_id,
            expert_user_id,
            test_type,
            examinee_info.get("이름", ""),
            examinee_info.get("생년월일", ""),
            examinee_info.get("성별", ""),
            examinee_info.get("검사자", ""),
            examinee_info.get("검사일", ""),
            now,
        ))

        for result_name, raw_score in index_scores.items():
            matched = index_cla_com.get(result_name)
            if not matched:
                continue
            classification, comment = next(iter(matched.items()))
            cur.execute("""
            INSERT INTO test_result (
                test_id, result_type, result_name, raw_score, classification, comment
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (test_id, "index", result_name, raw_score, classification, comment))

        for result_name, raw_score in subtest_scores.items():
            matched = subtest_cla_com.get(result_name)
            if not matched:
                continue
            classification, comment = next(iter(matched.items()))
            cur.execute("""
            INSERT INTO test_result (
                test_id, result_type, result_name, raw_score, classification, comment
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (test_id, "subtest", result_name, raw_score, classification, comment))

        cur.execute("""
        INSERT INTO final_report (
            test_id, prompt, final_report, model_name, created_at
        ) VALUES (%s, %s, %s, %s, %s)
        """, (test_id, prompt, final_report_text, model_name, now))

        conn.commit()
        return test_id
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_connection(conn)


# =========================================================
# 누적 결과 대시보드 (이력 조회 + 시각화)
# =========================================================
@st.cache_data(ttl=60)
def get_examinee_test_history(
    examinee_name: str,
    date_of_birth: str,
    test_type: Optional[str] = None,
) -> pd.DataFrame:
    """이름 + 생년월일로 동일 수검자의 회차별 검사 결과를 조회합니다."""
    conn = get_connection()
    try:
        query = """
        SELECT
            tr.test_id,
            tr.test_type,
            tr.test_date,
            tr.created_at,
            trres.result_type,
            trres.result_name,
            trres.raw_score,
            trres.classification
        FROM test_run tr
        JOIN test_result trres ON trres.test_id = tr.test_id
        WHERE tr.examinee_name = %s AND tr.date_of_birth = %s
        """
        params: List[str] = [examinee_name, date_of_birth]
        if test_type:
            query += " AND tr.test_type = %s"
            params.append(test_type)
        query += " ORDER BY tr.created_at ASC"

        return pd.read_sql(query, conn, params=params)
    finally:
        release_connection(conn)


def build_index_radar_chart(df_history: pd.DataFrame) -> Optional[bytes]:
    """가장 최근 회차의 지표점수로 레이더차트 PNG bytes를 생성합니다."""
    df_idx = df_history[df_history["result_type"] == "index"]
    if df_idx.empty:
        return None

    latest_test_id = df_idx.sort_values("created_at").iloc[-1]["test_id"]
    latest = df_idx[df_idx["test_id"] == latest_test_id].drop_duplicates("result_name")

    labels = latest["result_name"].tolist()
    values = [float(v) for v in latest["raw_score"].tolist()]

    if len(labels) < 3:
        return None  # 레이더차트는 축이 3개 이상이어야 의미가 있음

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, linewidth=2, color="#4C72B0")
    ax.fill(angles_closed, values_closed, alpha=0.25, color="#4C72B0")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(40, 160)
    ax.set_title("지표점수 프로파일 (최근 회차)", fontsize=12, pad=20)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_index_profile_chart(index_scores: Dict[str, int], title: str = "인지능력 지표 프로파일") -> Optional[bytes]:
    """현재 검사 회차의 지표점수를 웩슬러식 프로파일 형태의 PNG bytes로 생성합니다.

    실제 웩슬러 검사지를 복제하지 않고, 일반적인 심리검사 점수 프로파일
    (영역별 표준점수 + 평균 범위 표시)에 가까운 형태로 시각화합니다.
    """
    if not index_scores:
        return None

    labels = list(index_scores.keys())
    values = [float(index_scores[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    # 지표점수의 일반적인 평균 구간(90~109)을 배경으로 표시
    ax.axhspan(90, 109, alpha=0.12, zorder=0)
    ax.axhline(100, linewidth=1.2, linestyle="--", alpha=0.65, zorder=1)

    x = np.arange(len(labels))
    ax.plot(x, values, marker="o", markersize=8, linewidth=2.2, zorder=3)

    for xi, value in zip(x, values):
        ax.annotate(
            f"{int(value)}",
            (xi, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("지표점수", fontsize=10)
    ax.set_ylim(40, 160)
    ax.set_yticks([40, 60, 80, 90, 100, 110, 120, 140, 160])
    ax.set_title(title, fontsize=14, pad=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 평균선/범례 설명
    ax.text(
        0.99, 0.03,
        "음영: 일반적인 평균 범위(90~109)  |  점선: 100",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5, alpha=0.75,
    )

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_index_trend_chart(df_history: pd.DataFrame) -> Optional[bytes]:
    """회차별 지표점수 추이를 라인차트 PNG bytes로 생성합니다."""
    df_idx = df_history[df_history["result_type"] == "index"].copy()
    if df_idx.empty:
        return None

    session_order = df_idx.drop_duplicates("test_id").sort_values("created_at")["test_id"].tolist()
    if len(session_order) < 2:
        return None  # 추이 비교는 2회 이상부터 의미가 있음

    df_idx["session_label"] = df_idx["test_date"].where(
        df_idx["test_date"].fillna("").str.strip() != "", df_idx["created_at"]
    )

    pivot = df_idx.pivot_table(
        index="test_id", columns="result_name", values="raw_score", aggfunc="first"
    ).reindex(session_order)

    session_labels = (
        df_idx.drop_duplicates("test_id").set_index("test_id").loc[session_order, "session_label"].tolist()
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for col in pivot.columns:
        ax.plot(range(len(session_labels)), pivot[col], marker="o", label=col)

    ax.set_xticks(range(len(session_labels)))
    ax.set_xticklabels(session_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("지표점수")
    ax.set_title("회차별 지표점수 추이", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@_fragment_decorator
def render_cumulative_dashboard() -> None:
    st.subheader("누적 검사 결과 대시보드")
    st.caption("동일 수검자의 이름 + 생년월일 기준으로 회차별 결과를 비교합니다. (동명이인 주의)")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        search_name = st.text_input("수검자 이름", key="dash_search_name")
    with col2:
        search_dob = st.text_input("생년월일", key="dash_search_dob")
    with col3:
        search_test_type = st.selectbox(
            "검사 유형(선택)", options=[""] + list(SELECTION.keys()), key="dash_search_test_type"
        )

    search_clicked = st.button("조회", key="dash_search_button")

    if search_clicked:
        if not safe_text(search_name) or not safe_text(search_dob):
            st.error("이름과 생년월일을 입력해주세요.")
            return

        df_history = get_examinee_test_history(
            search_name.strip(), search_dob.strip(), search_test_type or None
        )
        st.session_state["dash_history_df"] = df_history

    df_history = st.session_state.get("dash_history_df")
    if df_history is None:
        return

    if df_history.empty:
        st.info("해당 수검자의 검사 기록이 없습니다.")
        return

    n_sessions = df_history["test_id"].nunique()
    st.success(f"총 {n_sessions}회 검사 기록을 찾았습니다.")

    radar_png = build_index_radar_chart(df_history)
    trend_png = build_index_trend_chart(df_history)

    col_a, col_b = st.columns(2)
    with col_a:
        if radar_png:
            st.image(radar_png, caption="최근 회차 지표점수 프로파일")
        else:
            st.info("레이더차트는 지표점수가 3개 이상일 때 표시됩니다.")
    with col_b:
        if trend_png:
            st.image(trend_png, caption="회차별 지표점수 추이")
        else:
            st.info("추이차트는 검사 기록이 2회 이상일 때 표시됩니다.")

    st.session_state["dash_radar_png"] = radar_png
    st.session_state["dash_trend_png"] = trend_png

    with st.expander("원본 데이터 보기"):
        st.dataframe(
            df_history[["test_date", "test_type", "result_type", "result_name", "raw_score", "classification"]],
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.get("last_generated_report"):
        st.divider()
        if st.button("현재 보고서에 차트 포함해서 PDF 재생성", key="dash_pdf_with_chart_btn"):
            chart_imgs = [img for img in [radar_png, trend_png] if img]
            if not chart_imgs:
                st.warning("포함할 차트가 없습니다. (지표점수 3개 이상 또는 검사 2회 이상 필요)")
            else:
                pdf_bytes = make_pdf_bytes(
                    "Psycolor Report",
                    st.session_state["last_generated_report"].splitlines(),
                    chart_images=chart_imgs,
                )
                st.session_state["dash_pdf_with_chart"] = pdf_bytes
                st.success("차트 포함 PDF가 생성되었습니다.")

        if st.session_state.get("dash_pdf_with_chart"):
            st.download_button(
                "차트 포함 PDF 다운로드",
                data=st.session_state["dash_pdf_with_chart"],
                file_name="psycolor_report_with_chart.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dash_pdf_with_chart_download",
            )


# =========================================================
# 커뮤니티 / 메일
# =========================================================
def create_post(
    board_type: str,
    author_user_id: int,
    title: str,
    content: str,
    image_bytes: Optional[bytes],
    image_name: Optional[str],
    image_mime: Optional[str],
) -> None:
    approval_status = "approved" if board_type == "anonymous" else "pending"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO community_post (
        board_type, author_user_id, title, content, image_bytes, image_name, image_mime,
        approval_status, created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        board_type,
        author_user_id,
        safe_text(title),
        safe_text(content),
        psycopg2.Binary(image_bytes) if image_bytes else None,
        image_name,
        image_mime,
        approval_status,
        now_str(),
    ))
    conn.commit()
    cur.close()
    release_connection(conn)
    _load_posts_cached.clear()


def toggle_like(post_id: int, user_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT like_id FROM post_like WHERE post_id = %s AND user_id = %s", (post_id, user_id))
    row = cur.fetchone()
    if row:
        cur.execute("DELETE FROM post_like WHERE like_id = %s", (row[0],))
    else:
        cur.execute("""
        INSERT INTO post_like (post_id, user_id, created_at)
        VALUES (%s, %s, %s)
        """, (post_id, user_id, now_str()))
    conn.commit()
    cur.close()
    release_connection(conn)
    _load_posts_cached.clear()


def add_comment(post_id: int, user_id: int, content: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO post_comment (post_id, user_id, content, created_at)
    VALUES (%s, %s, %s, %s)
    """, (post_id, user_id, safe_text(content), now_str()))
    conn.commit()
    cur.close()
    release_connection(conn)
    _load_comments_cached.clear()
    _load_posts_cached.clear()  # 댓글 수 표시가 갱신되도록 게시글 목록 캐시도 무효화


def load_posts(board_type: str, include_pending: bool = False) -> pd.DataFrame:
    return _load_posts_cached(board_type, include_pending)


@st.cache_data(ttl=20)
def _load_posts_cached(board_type: str, include_pending: bool) -> pd.DataFrame:
    conn = get_connection()
    base_query = """
    SELECT
        p.post_id,
        p.board_type,
        p.title,
        p.content,
        p.image_bytes,
        p.image_name,
        p.image_mime,
        p.approval_status,
        p.created_at,
        u.user_id AS author_user_id,
        u.role AS author_role,
        COALESCE(u.nickname, u.username, '-') AS author_name,
        COALESCE(l.like_count, 0) AS like_count,
        COALESCE(c.comment_count, 0) AS comment_count
    FROM community_post p
    JOIN users u
        ON p.author_user_id = u.user_id
    LEFT JOIN (
        SELECT post_id, COUNT(*) AS like_count
        FROM post_like
        GROUP BY post_id
    ) l
        ON p.post_id = l.post_id
    LEFT JOIN (
        SELECT post_id, COUNT(*) AS comment_count
        FROM post_comment
        GROUP BY post_id
    ) c
        ON p.post_id = c.post_id
    WHERE p.board_type = %s
    """
    params = [board_type]
    if not include_pending:
        base_query += " AND p.approval_status = 'approved' "
    base_query += " ORDER BY p.created_at DESC "
    try:
        df = pd.read_sql_query(base_query, conn, params=params)
    finally:
        release_connection(conn)

    # PostgreSQL BYTEA는 psycopg2에서 memoryview로 반환될 수 있습니다.
    # st.cache_data는 memoryview가 들어 있는 DataFrame을 직렬화하지 못해
    # UnserializableReturnValueError가 발생할 수 있으므로 캐시 반환 전에
    # 반드시 일반 Python bytes/None으로 정규화합니다.
    if "image_bytes" in df.columns:
        df["image_bytes"] = df["image_bytes"].map(normalize_binary_data)

    return df


def load_comments(post_id: int) -> pd.DataFrame:
    return _load_comments_cached(post_id)


@st.cache_data(ttl=20)
def _load_comments_cached(post_id: int) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT
        c.comment_id,
        c.content,
        c.created_at,
        u.role,
        COALESCE(u.nickname, u.username, '-') AS author_name
    FROM post_comment c
    JOIN users u
        ON c.user_id = u.user_id
    WHERE c.post_id = %s
    ORDER BY c.created_at ASC
    """
    df = pd.read_sql_query(query, conn, params=(post_id,))
    release_connection(conn)
    return df


def user_liked_post(post_id: int, user_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM post_like WHERE post_id = %s AND user_id = %s", (post_id, user_id))
    found = cur.fetchone() is not None
    cur.close()
    release_connection(conn)
    return found


def send_report_message(
    sender_user_id: int,
    receiver_user_id: int,
    title: str,
    message_text: str,
    file_name: str,
    file_mime: str,
    file_bytes: bytes,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO inbox_message (
        sender_user_id, receiver_user_id, title, message_text,
        file_name, file_mime, file_bytes, created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        sender_user_id,
        receiver_user_id,
        safe_text(title),
        safe_text(message_text),
        file_name,
        file_mime,
        psycopg2.Binary(file_bytes),
        now_str(),
    ))
    conn.commit()
    cur.close()
    release_connection(conn)
    _load_received_messages_cached.clear()


def load_received_messages(receiver_user_id: int) -> pd.DataFrame:
    return _load_received_messages_cached(receiver_user_id)


@st.cache_data(ttl=20)
def _load_received_messages_cached(receiver_user_id: int) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT
        m.message_id,
        m.title,
        m.message_text,
        m.file_name,
        m.file_mime,
        m.file_bytes,
        m.created_at,
        COALESCE(u.nickname, u.username, '-') AS sender_name,
        u.role AS sender_role
    FROM inbox_message m
    JOIN users u
        ON m.sender_user_id = u.user_id
    WHERE m.receiver_user_id = %s
    ORDER BY m.created_at DESC
    """
    df = pd.read_sql_query(query, conn, params=(receiver_user_id,))
    release_connection(conn)
    return df


# =========================================================
# 화면 유틸
# =========================================================
def render_post_image(image_bytes, image_name: Optional[str], image_mime: Optional[str]) -> None:
    image_bytes = normalize_binary_data(image_bytes)
    if not image_bytes:
        return
    if image_mime and not str(image_mime).startswith("image/"):
        return
    try:
        st.image(image_bytes, caption=image_name, use_container_width=True)
    except Exception:
        st.caption("이미지 표시 중 오류가 발생했습니다.")


def render_post_detail_expander(
    row,
    allow_like: bool,
    allow_comment: bool,
    comment_form_prefix: str,
    comments_title: str = "댓글",
    anonymous_meta: bool = False,
) -> None:
    post_id = int(row["post_id"])
    title = str(row["title"])

    with st.expander(title, expanded=False):
        if anonymous_meta:
            st.caption(f"익명 게시글 · 작성일: {row['created_at']}")
        else:
            st.caption(
                f"작성자: {row['author_name']} · 유형: {role_badge(row['author_role'])} · 작성일: {row['created_at']}"
            )

        st.write(row["content"])
        render_post_image(row.get("image_bytes"), row.get("image_name"), row.get("image_mime"))

        if allow_like:
            cols = st.columns([1, 1, 4])
            liked = user_liked_post(post_id, int(st.session_state["user_id"]))
            like_label = "좋아요 취소" if liked else "좋아요"
            with cols[0]:
                if st.button(f"{like_label} ({int(row['like_count'])})", key=f"{comment_form_prefix}_like_{post_id}"):
                    toggle_like(post_id, int(st.session_state["user_id"]))
                    st.rerun()
            with cols[1]:
                st.write(f"댓글 {int(row['comment_count'])}")
        else:
            st.caption(f"댓글 {int(row['comment_count'])}")

        comments = load_comments(post_id)
        if not comments.empty:
            st.markdown(comments_title)
            for _, c_row in comments.iterrows():
                badge = " [전문가]" if c_row["role"] == "expert" else ""
                st.write(f"- {c_row['author_name']}{badge}: {c_row['content']} ({c_row['created_at']})")

        if allow_comment:
            with st.form(key=f"{comment_form_prefix}_comment_form_{post_id}", clear_on_submit=True):
                comment_text = st.text_input("댓글 작성", key=f"{comment_form_prefix}_comment_{post_id}")
                submitted = st.form_submit_button("댓글 등록", use_container_width=True)
            if submitted:
                if not safe_text(comment_text):
                    st.warning("댓글 내용을 입력해주세요.")
                else:
                    add_comment(post_id, int(st.session_state["user_id"]), comment_text)
                    st.success("댓글이 등록되었습니다.")
                    st.rerun()


# =========================================================
# 화면 렌더링
# =========================================================
def render_auth_page() -> None:
    st.title("🧠 Psycolor")
    st.caption("회원 유형별 로그인 / 회원가입")

    tab1, tab2 = st.tabs(["로그인", "회원가입"])

    with tab1:
        st.subheader("로그인")
        login_username = st.text_input("아이디", key="login_username")
        login_password = st.text_input("비밀번호", type="password", key="login_password")

        if st.button("로그인", type="primary", use_container_width=True):
            user = authenticate_user(login_username, login_password)
            if not user:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
                return

            st.session_state["logged_in"] = True
            st.session_state["user_id"] = user["user_id"]
            st.session_state["username"] = user["username"]
            st.session_state["nickname"] = user["nickname"]
            st.session_state["role"] = user["role"]
            st.success("로그인되었습니다.")
            st.rerun()

    with tab2:
        st.subheader("회원가입")
        sign_role = st.selectbox(
            "회원 유형",
            options=["general", "senior", "expert"],
            format_func=lambda x: ROLE_LABELS[x],
            key="signup_role",
        )
        sign_username = st.text_input("아이디", key="signup_username")
        sign_password = st.text_input("비밀번호", type="password", key="signup_password")
        sign_password2 = st.text_input("비밀번호 확인", type="password", key="signup_password2")
        sign_nickname = st.text_input("닉네임", key="signup_nickname")
        st.caption("닉네임은 일반 이용자 / 시니어 이용자 / 전문가 전체에서 중복 없이 사용됩니다.")

        if st.button("회원가입", use_container_width=True):
            try:
                if not safe_text(sign_username):
                    raise ValueError("아이디를 입력해주세요.")
                if not sign_password:
                    raise ValueError("비밀번호를 입력해주세요.")
                if sign_password != sign_password2:
                    raise ValueError("비밀번호 확인이 일치하지 않습니다.")
                create_user(
                    username=sign_username,
                    password=sign_password,
                    role=sign_role,
                    nickname=sign_nickname,
                )
                st.success("회원가입이 완료되었습니다. 로그인 해주세요.")
            except Exception as e:
                st.error(f"회원가입 중 오류가 발생했습니다: {e}")


def render_sidebar() -> None:
    with st.sidebar:
        st.write(f"로그인 사용자: {st.session_state['username']}")
        if st.session_state["nickname"]:
            st.write(f"닉네임: {st.session_state['nickname']}")
        st.write(f"권한: {ROLE_LABELS.get(st.session_state['role'], st.session_state['role'])}")
        if st.button("로그아웃", use_container_width=True):
            logout()
            st.rerun()


@_fragment_decorator
def render_general_public_community() -> None:
    st.subheader("공개 커뮤니티")

    with st.expander("게시글 작성", expanded=False):
        title = st.text_input("제목", key="public_title")
        content = st.text_area("내용", key="public_content", height=160)
        uploaded = st.file_uploader(
            "사진 업로드",
            type=["png", "jpg", "jpeg", "webp"],
            key="public_file",
        )
        if st.button("공개 커뮤니티 글 등록", key="public_submit", use_container_width=True):
            try:
                if not safe_text(title) or not safe_text(content):
                    raise ValueError("제목과 내용을 입력해주세요.")

                image_bytes = uploaded.getvalue() if uploaded else None
                image_name = uploaded.name if uploaded else None
                image_mime = uploaded.type if uploaded else None

                create_post(
                    board_type="public",
                    author_user_id=st.session_state["user_id"],
                    title=title,
                    content=content,
                    image_bytes=image_bytes,
                    image_name=image_name,
                    image_mime=image_mime,
                )
                st.success("게시글이 등록되었습니다. 관리자 승인 후 공개됩니다.")
                st.rerun()
            except Exception as e:
                st.error(f"게시글 등록 중 오류가 발생했습니다: {e}")

    posts = load_posts("public", include_pending=False)
    if posts.empty:
        st.info("아직 승인된 공개 게시글이 없습니다.")
        return

    for _, row in posts.iterrows():
        render_post_detail_expander(
            row=row,
            allow_like=True,
            allow_comment=True,
            comment_form_prefix="public",
            comments_title="댓글",
            anonymous_meta=False,
        )


@_fragment_decorator
def render_general_anonymous_write_only() -> None:
    st.subheader("익명 커뮤니티")
    st.caption("익명 커뮤니티 게시글은 관리자 승인 없이 바로 등록되며, 전문가만 열람 및 댓글 작성이 가능합니다.")

    with st.form("general_anonymous_post_form", clear_on_submit=True):
        title = st.text_input("익명 게시글 제목", key="anon_title_general")
        content = st.text_area("익명 게시글 내용", key="anon_content_general", height=160)
        uploaded = st.file_uploader(
            "사진 업로드",
            type=["png", "jpg", "jpeg", "webp"],
            key="anon_file_general",
        )
        submitted = st.form_submit_button("익명 커뮤니티 글 등록", use_container_width=True)

    if submitted:
        try:
            if not safe_text(title) or not safe_text(content):
                raise ValueError("제목과 내용을 입력해주세요.")
            image_bytes = uploaded.getvalue() if uploaded else None
            image_name = uploaded.name if uploaded else None
            image_mime = uploaded.type if uploaded else None
            create_post(
                board_type="anonymous",
                author_user_id=st.session_state["user_id"],
                title=title,
                content=content,
                image_bytes=image_bytes,
                image_name=image_name,
                image_mime=image_mime,
            )
            st.success("익명 게시글이 등록되었습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"익명 게시글 등록 중 오류가 발생했습니다: {e}")

    st.divider()
    st.markdown("내가 작성한 익명 게시글")
    my_posts = load_posts("anonymous", include_pending=False)
    my_posts = my_posts[my_posts["author_user_id"] == int(st.session_state["user_id"])]

    if my_posts.empty:
        st.info("아직 작성한 익명 게시글이 없습니다.")
        return

    for _, row in my_posts.iterrows():
        render_post_detail_expander(
            row=row,
            allow_like=False,
            allow_comment=False,
            comment_form_prefix="general_my_anonymous",
            comments_title="전문가 댓글",
            anonymous_meta=True,
        )


@_fragment_decorator
def render_general_inbox() -> None:
    st.subheader("보고서 수신함")

    df = load_received_messages(int(st.session_state["user_id"]))
    if df.empty:
        st.info("수신된 보고서가 없습니다.")
        return

    for _, row in df.iterrows():
        with st.expander(str(row["title"]), expanded=False):
            sender_badge = "전문가" if row["sender_role"] == "expert" else role_badge(row["sender_role"])
            st.caption(f"발신자: {row['sender_name']} ({sender_badge}) · 수신일: {row['created_at']}")
            if safe_text(row["message_text"]):
                st.write(row["message_text"])
            file_bytes = normalize_binary_data(row["file_bytes"])
            if file_bytes:
                st.download_button(
                    label=f"첨부파일 다운로드: {row['file_name']}",
                    data=file_bytes,
                    file_name=row["file_name"],
                    mime=row["file_mime"] or "application/octet-stream",
                    key=f"inbox_download_{row['message_id']}",
                    use_container_width=True,
                )


@_fragment_decorator
def render_senior_page() -> None:
    st.title("시니어 사용자")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("식단 추천", use_container_width=True):
            st.info("추후 서비스 오픈 예정")
    with col2:
        if st.button("비대면 정기 체크", use_container_width=True):
            st.info("추후 서비스 오픈 예정")


@_fragment_decorator
def render_expert_report_generator() -> None:
    st.subheader("보고서 생성")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("수검자 정보")
        examinee_name = st.text_input("이름", key="exp_examinee_name")
        date_of_birth = st.text_input("생년월일", key="exp_dob")
        sex = st.selectbox("성별", options=["", "남", "여", "기타"], key="exp_sex")
        examiner = st.text_input("검사자", value=st.session_state.get("nickname") or st.session_state.get("username"), key="exp_examiner")
        test_date = st.text_input("검사일", key="exp_test_date")

        examinee_info = {
            "이름": examinee_name,
            "생년월일": date_of_birth,
            "성별": sex,
            "검사자": examiner,
            "검사일": test_date,
        }

        test_type = st.selectbox("검사 유형", options=list(SELECTION.keys()), key="exp_test_type")

        st.markdown("지표점수 입력")
        index_scores: Dict[str, int] = {}
        for index_code in SELECTION[test_type].keys():
            value = st.number_input(
                f"{index_code} 지표점수",
                min_value=40,
                max_value=200,
                value=None,
                step=1,
                placeholder="미시행 시 건너뛰기",
                key=f"exp_index_{test_type}_{index_code}",
            )
            if value is not None:
                index_scores[index_code] = int(value)

        st.markdown("소검사 환산점수 입력")
        subtest_scores: Dict[str, int] = {}
        for subtest_code in flatten_subtests(test_type):
            value = st.number_input(
                f"{subtest_code} 환산점수",
                min_value=1,
                max_value=19,
                value=None,
                step=1,
                placeholder="미시행 시 건너뛰기",
                key=f"exp_subtest_{test_type}_{subtest_code}",
            )
            if value is not None:
                subtest_scores[subtest_code] = int(value)

        generate_clicked = st.button("보고서 생성 및 저장", type="primary", use_container_width=True, key="expert_generate")

    with right:
        st.markdown("생성 결과")
        if st.session_state.get("last_generated_report"):
            st.text_area(
                "최종 보고서",
                st.session_state["last_generated_report"],
                height=450,
                key="expert_generated_report_view",
            )
            profile_png = st.session_state.get("last_generated_profile_png")
            if profile_png:
                st.markdown("#### 영역별 지표점수 프로파일")
                st.image(profile_png, caption="현재 검사 회차 지표점수 프로파일", use_container_width=True)
        else:
            st.info("아직 생성된 보고서가 없습니다.")

    if generate_clicked:
        if not index_scores and not subtest_scores:
            st.error("최소 1개 이상의 점수를 입력해주세요.")
            return

        errors = validate_scores(index_scores, subtest_scores)
        if errors:
            for err in errors:
                st.error(err)
            return

        index_cla_com = put_index_cla_and_com(test_type, index_scores)
        subtest_cla_com = put_subtest_cla_and_com(test_type, subtest_scores)
        prompt = build_prompt(test_type, index_cla_com, subtest_cla_com, examinee_info)

        try:
            with st.spinner("보고서를 생성하는 중입니다..."):
                final_report_text = generate_report(prompt)

            saved_test_id = save_test_run(
                expert_user_id=int(st.session_state["user_id"]),
                test_type=test_type,
                examinee_info=examinee_info,
                index_scores=index_scores,
                subtest_scores=subtest_scores,
                index_cla_com=index_cla_com,
                subtest_cla_com=subtest_cla_com,
                prompt=prompt,
                final_report_text=final_report_text,
                model_name=MODEL_NAME,
            )

            txt_content = "\n".join(["[최종 보고서]", final_report_text])

            # 방금 입력한 현재 회차의 지표점수 프로파일을 PDF에도 함께 첨부합니다.
            profile_png = build_index_profile_chart(index_scores)
            pdf_chart_images = [profile_png] if profile_png else None
            pdf_bytes = make_pdf_bytes(
                "Psycolor Report",
                final_report_text.splitlines(),
                chart_images=pdf_chart_images,
            )
            txt_bytes = make_txt_bytes(txt_content)

            st.session_state["last_generated_index_scores"] = dict(index_scores)
            st.session_state["last_generated_profile_png"] = profile_png

            st.session_state["last_generated_test_id"] = saved_test_id
            st.session_state["last_generated_report"] = final_report_text
            st.session_state["last_generated_prompt"] = prompt
            st.session_state["last_generated_test_type"] = test_type
            st.session_state["last_generated_pdf"] = pdf_bytes
            st.session_state["last_generated_txt"] = txt_bytes
            st.session_state["pending_payment"] = True
            st.session_state["payment_notice_ack"] = False

            st.success("보고서가 저장되었습니다.")
            st.rerun()

        except Exception as e:
            st.error(f"생성 중 오류가 발생했습니다: {e}")

    if st.session_state.get("last_generated_report"):
        st.divider()
        st.subheader("결제")

        if st.session_state.get("pending_payment", False):
            if st.button("결제", type="primary", use_container_width=True, key="expert_payment_button"):
                st.session_state["payment_notice_ack"] = True
                st.rerun()

        if st.session_state.get("payment_notice_ack", False):
            with st.container(border=True):
                st.warning("현재 지원되지 않는 서비스 입니다.")
                if st.button("확인", use_container_width=True, key="expert_payment_notice_confirm"):
                    st.session_state["pending_payment"] = False
                    st.session_state["payment_notice_ack"] = False
                    st.rerun()

        if not st.session_state.get("pending_payment", False) and not st.session_state.get("payment_notice_ack", False):
            st.success("다운로드가 가능합니다.")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="보고서 TXT 다운로드",
                    data=st.session_state["last_generated_txt"],
                    file_name="psycolor_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="expert_txt_download",
                )
            with col2:
                st.download_button(
                    label="보고서 PDF 다운로드",
                    data=st.session_state["last_generated_pdf"],
                    file_name="psycolor_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="expert_pdf_download",
                )


@_fragment_decorator
def render_expert_public_community() -> None:
    st.subheader("공개 커뮤니티")
    posts = load_posts("public", include_pending=False)
    if posts.empty:
        st.info("아직 승인된 공개 게시글이 없습니다.")
        return

    for _, row in posts.iterrows():
        render_post_detail_expander(
            row=row,
            allow_like=True,
            allow_comment=True,
            comment_form_prefix="expert_public",
            comments_title="댓글",
            anonymous_meta=False,
        )


@_fragment_decorator
def render_expert_anonymous_comments() -> None:
    st.subheader("익명 커뮤니티")
    st.caption("익명 커뮤니티는 전문가만 열람 및 댓글 작성이 가능합니다.")

    posts = load_posts("anonymous", include_pending=False)
    if posts.empty:
        st.info("등록된 익명 게시글이 없습니다.")
        return

    for _, row in posts.iterrows():
        render_post_detail_expander(
            row=row,
            allow_like=False,
            allow_comment=True,
            comment_form_prefix="anonymous",
            comments_title="전문가 댓글",
            anonymous_meta=True,
        )


@_fragment_decorator
def render_expert_send_report() -> None:
    st.subheader("보고서 발송")
    st.caption("전문가 → 일반 이용자 발송만 허용됩니다.")

    keyword = st.text_input("일반 이용자 닉네임 검색", key="receiver_search_keyword")
    if safe_text(keyword):
        results = search_general_users_by_nickname(keyword)
        if results.empty:
            st.info("검색된 일반 이용자가 없습니다.")
        else:
            st.dataframe(results, use_container_width=True, hide_index=True)

    with st.form("send_report_form"):
        receiver_nickname = st.text_input("수신자 닉네임", key="send_receiver_nickname")
        title = st.text_input("발송 제목", value="심리검사 보고서", key="send_report_title")
        message_text = st.text_area("메시지", value="보고서를 전달드립니다.", key="send_report_message")

        send_mode = st.radio(
            "발송 방식",
            options=["방금 생성한 PDF 발송", "직접 파일 업로드"],
            horizontal=True,
            key="send_mode",
        )

        uploaded = None
        if send_mode == "직접 파일 업로드":
            uploaded = st.file_uploader(
                "발송 파일 업로드",
                type=["pdf", "txt", "doc", "docx"],
                key="send_report_upload",
            )
        else:
            if st.session_state.get("last_generated_pdf") is None:
                st.info("현재 세션에서 생성된 보고서 PDF가 없습니다. 직접 파일 업로드를 선택하거나 먼저 보고서를 생성해주세요.")
            else:
                st.caption("현재 세션에서 생성한 최신 PDF가 발송됩니다.")

        submitted = st.form_submit_button("보고서 발송", type="primary", use_container_width=True)

    if submitted:
        try:
            if not safe_text(receiver_nickname):
                raise ValueError("수신자 닉네임을 입력해주세요.")

            receiver = get_user_by_nickname(receiver_nickname)
            if not receiver:
                raise ValueError("해당 닉네임의 사용자를 찾을 수 없습니다.")
            if receiver["role"] != "general":
                raise ValueError("보고서는 일반 이용자에게만 발송할 수 있습니다.")

            if send_mode == "직접 파일 업로드":
                if uploaded is None:
                    raise ValueError("업로드할 파일을 선택해주세요.")
                file_name = uploaded.name
                file_mime = uploaded.type or "application/octet-stream"
                file_bytes = uploaded.getvalue()
            else:
                file_name = "psycolor_report.pdf"
                file_mime = "application/pdf"
                file_bytes = normalize_binary_data(st.session_state.get("last_generated_pdf"))
                if not file_bytes:
                    raise ValueError("발송할 PDF가 없습니다. 먼저 보고서를 생성하거나 직접 파일 업로드를 선택해주세요.")

            send_report_message(
                sender_user_id=int(st.session_state["user_id"]),
                receiver_user_id=int(receiver["user_id"]),
                title=title,
                message_text=message_text,
                file_name=file_name,
                file_mime=file_mime,
                file_bytes=file_bytes,
            )
            st.success("보고서가 발송되었습니다.")
        except Exception as e:
            st.error(f"보고서 발송 중 오류가 발생했습니다: {e}")


def render_general_page() -> None:
    st.title("일반 이용자")
    tab1, tab2, tab3 = st.tabs(["공개 커뮤니티", "익명 커뮤니티", "보고서 수신함"])
    with tab1:
        render_general_public_community()
    with tab2:
        render_general_anonymous_write_only()
    with tab3:
        render_general_inbox()


def render_expert_page() -> None:
    st.title("전문가")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["보고서 생성", "누적 결과", "공개 커뮤니티", "익명 커뮤니티", "보고서 발송"]
    )
    with tab1:
        render_expert_report_generator()
    with tab2:
        render_cumulative_dashboard()
    with tab3:
        render_expert_public_community()
    with tab4:
        render_expert_anonymous_comments()
    with tab5:
        render_expert_send_report()


# =========================================================
# 앱 실행
# =========================================================
init_session_state()
init_db()

if not st.session_state["logged_in"]:
    render_auth_page()
else:
    render_sidebar()
    role = st.session_state["role"]

    if role == "general":
        render_general_page()
    elif role == "senior":
        render_senior_page()
    elif role == "expert":
        render_expert_page()
    elif role == "admin":
        st.error("관리자 계정은 별도 관리자 사이트에서만 접속할 수 있습니다.")
    else:
        st.error("알 수 없는 권한입니다.")
