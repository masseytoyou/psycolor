import os
from datetime import datetime
from typing import Optional

import pandas as pd
import psycopg2
import streamlit as st


st.set_page_config(page_title="Psycolor Admin", page_icon="🛠️", layout="wide")


# =========================================================
# 공통 유틸
# =========================================================
def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_text(value: Optional[str]) -> str:
    return (value or "").strip()


def get_db_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    try:
        return st.secrets["DATABASE_URL"]
    except Exception as e:
        raise ValueError("DATABASE_URL이 설정되지 않았습니다.") from e


def get_admin_access_code() -> str:
    env_code = os.getenv("ADMIN_SIGNUP_CODE") or os.getenv("ADMIN_ACCESS_CODE")
    if env_code:
        return env_code

    for key in ["ADMIN_SIGNUP_CODE", "ADMIN_ACCESS_CODE"]:
        try:
            return st.secrets[key]
        except Exception:
            continue

    return "858585"


def get_connection():
    return psycopg2.connect(get_db_url())


def init_session_state() -> None:
    defaults = {
        "admin_access_granted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout() -> None:
    st.session_state["admin_access_granted"] = False


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


def init_db() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(20260331)")

            # 관리자 앱은 조회/승인만 하므로 필요한 테이블만 최소한으로 보강
            if table_exists(conn, "community_post"):
                if not column_exists(conn, "community_post", "board_type"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN board_type TEXT")
                if not column_exists(conn, "community_post", "approval_status"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN approval_status TEXT")
                if not column_exists(conn, "community_post", "approved_by"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN approved_by BIGINT")
                if not column_exists(conn, "community_post", "approved_at"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN approved_at TEXT")
                if not column_exists(conn, "community_post", "image_name"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN image_name TEXT")
                if not column_exists(conn, "community_post", "image_mime"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN image_mime TEXT")
                if not column_exists(conn, "community_post", "image_bytes"):
                    cur.execute("ALTER TABLE community_post ADD COLUMN image_bytes BYTEA")

                cur.execute(
                    "UPDATE community_post SET approval_status = COALESCE(approval_status, 'pending')"
                )

            # 익명 게시판은 관리자 승인 없이 바로 노출되도록 상태 보정
            if table_exists(conn, "community_post"):
                cur.execute(
                    """
                    UPDATE community_post
                    SET approval_status = 'approved'
                    WHERE board_type = 'anonymous'
                      AND COALESCE(approval_status, 'pending') <> 'approved'
                    """
                )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(20260331)")
        except Exception:
            pass
        conn.close()


# =========================================================
# 데이터 로드 / 승인
# =========================================================
def load_generated_reports(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT
        tr.test_id,
        tr.test_type,
        tr.examinee_name,
        tr.date_of_birth,
        tr.sex,
        tr.examiner,
        tr.test_date,
        COALESCE(u.nickname, u.username, '-') AS created_by,
        fr.model_name,
        fr.created_at,
        LEFT(fr.final_report, 120) AS preview
    FROM test_run tr
    LEFT JOIN final_report fr
        ON tr.test_id = fr.test_id
    LEFT JOIN users u
        ON tr.expert_user_id = u.user_id
    ORDER BY fr.created_at DESC NULLS LAST
    LIMIT %s
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def load_public_pending_posts() -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT
        p.post_id,
        p.title,
        p.content,
        p.image_bytes,
        p.image_name,
        p.image_mime,
        p.created_at,
        COALESCE(u.nickname, u.username, '-') AS author_name,
        COALESCE(u.role, 'general') AS author_role
    FROM community_post p
    LEFT JOIN users u
        ON p.author_user_id = u.user_id
    WHERE p.board_type = 'public'
      AND COALESCE(p.approval_status, 'pending') = 'pending'
    ORDER BY p.created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def approve_public_post(post_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE community_post
        SET approval_status = 'approved',
            approved_at = %s
        WHERE post_id = %s
        """,
        (now_str(), post_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def reject_public_post(post_id: int) -> None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE community_post
        SET approval_status = 'rejected',
            approved_at = %s
        WHERE post_id = %s
        """,
        (now_str(), post_id),
    )
    conn.commit()
    cur.close()
    conn.close()


# =========================================================
# UI
# =========================================================
def render_gate() -> None:
    st.title("🛠️ Psycolor 관리자 사이트")
    st.caption("관리자 가입 코드를 입력하면 접속할 수 있습니다.")

    with st.form("admin_gate_form"):
        access_code = st.text_input("관리자 가입 코드", type="password")
        submitted = st.form_submit_button("접속", use_container_width=True)

    if submitted:
        if safe_text(access_code) == get_admin_access_code():
            st.session_state["admin_access_granted"] = True
            st.success("관리자 사이트에 접속했습니다.")
            st.rerun()
        else:
            st.error("관리자 가입 코드가 올바르지 않습니다.")


def role_badge(role: str) -> str:
    mapping = {
        "general": "일반",
        "senior": "시니어",
        "expert": "전문가",
        "admin": "관리자",
    }
    return mapping.get(role, role)


def render_sidebar() -> None:
    with st.sidebar:
        st.write("관리자 모드 접속 중")
        if st.button("접속 종료", use_container_width=True):
            logout()
            st.rerun()


def render_report_history() -> None:
    st.subheader("보고서 누적 생성 기록 조회")
    df = load_generated_reports(limit=200)
    if df.empty:
        st.info("아직 저장된 보고서 기록이 없습니다.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_post_approval() -> None:
    st.subheader("공개 커뮤니티 게시글 승인")
    st.caption("공개 커뮤니티만 관리자 승인을 거칩니다. 익명 게시판은 관리자 승인 없이 등록됩니다.")

    pending = load_public_pending_posts()
    if pending.empty:
        st.info("승인 대기 중인 공개 게시글이 없습니다.")
        return

    for _, row in pending.iterrows():
        with st.container(border=True):
            st.write(f"제목: {row['title']}")
            st.caption(
                f"작성자: {row['author_name']} · 권한: {role_badge(str(row['author_role']))} · 작성일: {row['created_at']}"
            )
            st.write(row["content"])

            raw_image = row.get("image_bytes")
            image_mime = safe_text(row.get("image_mime"))
            image_name = safe_text(row.get("image_name"))

            if raw_image is not None and image_mime.startswith("image/"):
                if isinstance(raw_image, memoryview):
                    raw_image = raw_image.tobytes()
                elif not isinstance(raw_image, (bytes, bytearray)):
                    raw_image = bytes(raw_image)
                st.image(raw_image, caption=image_name or "첨부 이미지", use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("승인", key=f"approve_{row['post_id']}", use_container_width=True):
                    approve_public_post(int(row["post_id"]))
                    st.success("게시글을 승인했습니다.")
                    st.rerun()
            with c2:
                if st.button("반려", key=f"reject_{row['post_id']}", use_container_width=True):
                    reject_public_post(int(row["post_id"]))
                    st.success("게시글을 반려했습니다.")
                    st.rerun()


def render_admin_page() -> None:
    st.title("🛠️ Psycolor 관리자 사이트")
    render_report_history()
    st.divider()
    render_post_approval()


# =========================================================
# 앱 실행
# =========================================================
init_session_state()
init_db()

if not st.session_state["admin_access_granted"]:
    render_gate()
else:
    render_sidebar()
    render_admin_page()
