import os
import sqlite3
import uuid
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Psycolor 보고서 생성기", page_icon="🧠", layout="wide")

# =========================
# 설정
# =========================
INDEX_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=0"
SUBTEST_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=978787284"

MODEL_NAME = "gpt-5-mini"
DB_PATH = "psycolor.db"

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


# =========================
# OpenAI / Lookup 유틸
# =========================
def get_api_key() -> Optional[str]:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


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
    lines.append("")

    lines.append("[수검자 정보]")
    for k, v in examinee_info.items():
        if str(v).strip():
            lines.append(f"- {k}: {v}")

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
    lines.append("위 정보를 바탕으로 전체 결과를 5~8문장의 자연스러운 한국어 보고서 문단 1개로 작성하라.")
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


# =========================
# PDF/TXT/CSV 유틸
# =========================
def make_txt_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def make_pdf_bytes(title: str, lines: List[str]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # 한글 표시용 기본 CID 폰트
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    c.setFont("HYSMyeongJo-Medium", 11)

    width, height = A4
    x = 50
    y = height - 50
    line_height = 18

    c.setFont("HYSMyeongJo-Medium", 14)
    c.drawString(x, y, title)
    y -= 30

    c.setFont("HYSMyeongJo-Medium", 11)
    for line in lines:
        if y < 50:
            c.showPage()
            c.setFont("HYSMyeongJo-Medium", 11)
            y = height - 50

        c.drawString(x, y, str(line)[:100])
        y -= line_height

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# SQLite DB 유틸
# =========================
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    return column_name in cols


def init_db(db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    # 처음부터 최신 구조로 생성
    cur.execute("""
    CREATE TABLE IF NOT EXISTS test_run (
        test_id TEXT PRIMARY KEY,
        test_type TEXT NOT NULL,
        examinee_name TEXT,
        date_of_birth TEXT,
        sex TEXT,
        examiner TEXT,
        test_date TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS test_result (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id TEXT NOT NULL,
        result_type TEXT NOT NULL,
        result_name TEXT NOT NULL,
        raw_score INTEGER NOT NULL,
        classification TEXT NOT NULL,
        comment TEXT NOT NULL,
        FOREIGN KEY (test_id) REFERENCES test_run(test_id) ON DELETE CASCADE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS final_report (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        test_id TEXT NOT NULL UNIQUE,
        prompt TEXT,
        final_report TEXT NOT NULL,
        model_name TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (test_id) REFERENCES test_run(test_id) ON DELETE CASCADE
    )
    """)

    # 예전 DB 파일이 이미 있을 때 컬럼 추가 보정
    required_columns = {
        "examinee_name": "TEXT",
        "date_of_birth": "TEXT",
        "sex": "TEXT",
        "examiner": "TEXT",
        "test_date": "TEXT",
    }

    for col_name, col_type in required_columns.items():
        if not column_exists(conn, "test_run", col_name):
            cur.execute(f"ALTER TABLE test_run ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def save_test_run(
    test_type: str,
    examinee_info: Dict[str, str],
    index_scores: Dict[str, int],
    subtest_scores: Dict[str, int],
    index_cla_com: Dict[str, Dict[str, str]],
    subtest_cla_com: Dict[str, Dict[str, str]],
    prompt: str,
    final_report: str,
    model_name: str = MODEL_NAME,
    db_path: str = DB_PATH,
) -> str:
    test_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_connection(db_path)
    cur = conn.cursor()

    try:
        cur.execute("""
        INSERT INTO test_run (
            test_id, test_type, examinee_name, date_of_birth, sex, examiner, test_date, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
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
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                "index",
                result_name,
                raw_score,
                classification,
                comment,
            ))

        for result_name, raw_score in subtest_scores.items():
            matched = subtest_cla_com.get(result_name)
            if not matched:
                continue

            classification, comment = next(iter(matched.items()))
            cur.execute("""
            INSERT INTO test_result (
                test_id, result_type, result_name, raw_score, classification, comment
            ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                "subtest",
                result_name,
                raw_score,
                classification,
                comment,
            ))

        cur.execute("""
        INSERT INTO final_report (
            test_id, prompt, final_report, model_name, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """, (
            test_id,
            prompt,
            final_report,
            model_name,
            now,
        ))

        conn.commit()
        return test_id

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_test_runs(limit: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    query = """
    SELECT
        tr.test_id,
        tr.test_type,
        tr.examinee_name,
        tr.date_of_birth,
        tr.sex,
        tr.examiner,
        tr.test_date,
        fr.model_name,
        tr.created_at
    FROM test_run tr
    LEFT JOIN final_report fr
        ON tr.test_id = fr.test_id
    ORDER BY tr.created_at DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df


def load_test_detail(test_id: str, db_path: str = DB_PATH):
    conn = get_connection(db_path)

    run_df = pd.read_sql_query("""
    SELECT *
    FROM test_run
    WHERE test_id = ?
    """, conn, params=(test_id,))

    result_df = pd.read_sql_query("""
    SELECT
        result_type,
        result_name,
        raw_score,
        classification,
        comment
    FROM test_result
    WHERE test_id = ?
    ORDER BY
        CASE result_type
            WHEN 'index' THEN 1
            WHEN 'subtest' THEN 2
            ELSE 3
        END,
        result_name
    """, conn, params=(test_id,))

    report_df = pd.read_sql_query("""
    SELECT
        prompt,
        final_report,
        model_name,
        created_at
    FROM final_report
    WHERE test_id = ?
    """, conn, params=(test_id,))

    conn.close()
    return run_df, result_df, report_df


def get_scores_from_result_df(result_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    index_scores: Dict[str, int] = {}
    subtest_scores: Dict[str, int] = {}

    for _, row in result_df.iterrows():
        if row["result_type"] == "index":
            index_scores[str(row["result_name"])] = int(row["raw_score"])
        elif row["result_type"] == "subtest":
            subtest_scores[str(row["result_name"])] = int(row["raw_score"])

    return index_scores, subtest_scores


def update_final_report(
    test_id: str,
    prompt: str,
    final_report: str,
    model_name: str = MODEL_NAME,
    db_path: str = DB_PATH,
) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cur.execute("""
    UPDATE final_report
    SET prompt = ?, final_report = ?, model_name = ?, created_at = ?
    WHERE test_id = ?
    """, (prompt, final_report, model_name, now, test_id))

    conn.commit()
    conn.close()


def delete_test_run(test_id: str, db_path: str = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("DELETE FROM test_run WHERE test_id = ?", (test_id,))
    conn.commit()
    conn.close()


# =========================
# DB 생성
# =========================
# 이 함수는 기존 데이터를 지우지 않음.
# DB 파일이 없으면 만들고, 있으면 유지한 채 테이블만 점검함.
init_db()


# =========================
# UI
# =========================
st.title("🧠 Psycolor 보고서 생성기")
st.caption("룩업 테이블 + OpenAI API + SQLite 누적 저장 버전")

with st.expander("사용 전 확인", expanded=True):
    st.markdown(
        """
- 이 화면은 MVP 테스트용입니다.
- 흐름: 수검자 정보 입력 → 점수 입력 → 룩업 매핑 → 보고서 생성 → DB 누적 저장
- 앱을 다시 켜도 `psycolor.db` 파일이 남아 있으면 데이터는 유지됩니다.
- 다만 Streamlit Cloud 같은 환경은 로컬 파일 영속성이 약할 수 있습니다.
        """
    )

left, right = st.columns([1, 1])

with left:
    st.subheader("수검자 정보")
    examinee_name = st.text_input("이름")
    date_of_birth = st.text_input("생년월일", placeholder="예: 2018-03-21")
    sex = st.selectbox("성별", options=["", "남", "여", "기타"])
    examiner = st.text_input("검사자")
    test_date = st.text_input("검사일", placeholder="예: 2026-03-30")

    examinee_info = {
        "이름": examinee_name,
        "생년월일": date_of_birth,
        "성별": sex,
        "검사자": examiner,
        "검사일": test_date,
    }

    st.subheader("검사 입력")
    test_type = st.selectbox("검사 유형", options=list(SELECTION.keys()))

    st.subheader("지표점수 입력")
    index_scores: Dict[str, int] = {}
    for index_code in SELECTION[test_type].keys():
        value = st.number_input(
            f"{index_code} 지표점수",
            min_value=40,
            max_value=200,
            value=None,
            step=1,
            placeholder="비워두면 입력 안 함",
            key=f"index_{test_type}_{index_code}",
        )
        if value is not None:
            index_scores[index_code] = int(value)

    st.subheader("소검사 환산점수 입력")
    subtest_scores: Dict[str, int] = {}
    for subtest_code in flatten_subtests(test_type):
        value = st.number_input(
            f"{subtest_code} 환산점수",
            min_value=1,
            max_value=19,
            value=None,
            step=1,
            placeholder="비워두면 입력 안 함",
            key=f"subtest_{test_type}_{subtest_code}",
        )
        if value is not None:
            subtest_scores[subtest_code] = int(value)

    generate_clicked = st.button("보고서 생성 및 저장", type="primary", use_container_width=True)

with right:
    st.subheader("중간 결과")
    st.write("수검자 정보", examinee_info)
    st.write("입력된 지표점수", index_scores)
    st.write("입력된 소검사점수", subtest_scores)


if generate_clicked:
    if not index_scores and not subtest_scores:
        st.error("최소 1개 이상의 점수를 입력해주세요.")
        st.stop()

    errors = validate_scores(index_scores, subtest_scores)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    index_cla_com = put_index_cla_and_com(test_type, index_scores)
    subtest_cla_com = put_subtest_cla_and_com(test_type, subtest_scores)
    prompt = build_prompt(test_type, index_cla_com, subtest_cla_com, examinee_info)

    st.divider()
    st.subheader("룩업 매핑 결과")
    st.write("지표 분류/코멘트", index_cla_com)
    st.write("소검사 분류/코멘트", subtest_cla_com)

    st.subheader("생성 프롬프트")
    st.code(prompt, language="text")

    try:
        with st.spinner("AI가 보고서를 생성하는 중입니다..."):
            final_report = generate_report(prompt)

        saved_test_id = save_test_run(
            test_type=test_type,
            examinee_info=examinee_info,
            index_scores=index_scores,
            subtest_scores=subtest_scores,
            index_cla_com=index_cla_com,
            subtest_cla_com=subtest_cla_com,
            prompt=prompt,
            final_report=final_report,
            model_name=MODEL_NAME,
        )

        st.subheader("최종 보고서")
        st.text_area("생성 결과", final_report, height=260, key=f"new_report_{saved_test_id}")
        st.success(f"DB 저장 완료: {saved_test_id}")

    except Exception as e:
        st.error(f"생성 중 오류가 발생했습니다: {e}")


# =========================
# 저장 이력 조회 / 재생성 / 삭제 / 다운로드
# =========================
st.divider()
st.subheader("저장된 검사 이력")

try:
    history_df = load_test_runs(limit=50)

    if history_df.empty:
        st.info("아직 저장된 검사 이력이 없습니다.")
    else:
        st.dataframe(history_df, use_container_width=True)

        # 전체 이력 CSV 다운로드
        st.download_button(
            label="전체 이력 CSV 다운로드",
            data=dataframe_to_csv_bytes(history_df),
            file_name="psycolor_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        test_id_options = history_df["test_id"].tolist()
        selected_test_id = st.selectbox("상세 조회할 test_id 선택", test_id_options)

        if selected_test_id:
            run_df, result_df, report_df = load_test_detail(selected_test_id)

            st.markdown("### 검사 기본 정보")
            st.dataframe(run_df, use_container_width=True)

            st.markdown("### 항목별 결과")
            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                label="이 검사 결과 CSV 다운로드",
                data=dataframe_to_csv_bytes(result_df),
                file_name=f"{selected_test_id}_results.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"csv_download_{selected_test_id}",
            )

            st.markdown("### 최종 보고서")
            if not report_df.empty:
                saved_report = str(report_df.iloc[0]["final_report"])
                saved_prompt = str(report_df.iloc[0]["prompt"])
                saved_model = str(report_df.iloc[0]["model_name"])

                st.write("모델명:", saved_model)
                st.text_area(
                    "저장된 보고서",
                    saved_report,
                    height=250,
                    key=f"saved_report_{selected_test_id}",
                )

                with st.expander("저장된 프롬프트 보기"):
                    st.code(saved_prompt, language="text")

                txt_content = "\n".join([
                    f"test_id: {selected_test_id}",
                    "",
                    "[최종 보고서]",
                    saved_report
                ])

                pdf_lines = [
                    f"test_id: {selected_test_id}",
                    ""
                ] + saved_report.splitlines()

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="보고서 TXT 다운로드",
                        data=make_txt_bytes(txt_content),
                        file_name=f"{selected_test_id}_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"txt_download_{selected_test_id}",
                    )
                with col_dl2:
                    st.download_button(
                        label="보고서 PDF 다운로드",
                        data=make_pdf_bytes("Psycolor Report", pdf_lines),
                        file_name=f"{selected_test_id}_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"pdf_download_{selected_test_id}",
                    )

                st.markdown("### 보고서 재생성 / 삭제")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("선택한 검사 보고서 재생성", key=f"regen_{selected_test_id}", use_container_width=True):
                        run_row = run_df.iloc[0].to_dict()
                        index_scores_db, subtest_scores_db = get_scores_from_result_df(result_df)

                        test_type_db = str(run_row["test_type"])
                        examinee_info_db = {
                            "이름": str(run_row.get("examinee_name", "") or ""),
                            "생년월일": str(run_row.get("date_of_birth", "") or ""),
                            "성별": str(run_row.get("sex", "") or ""),
                            "검사자": str(run_row.get("examiner", "") or ""),
                            "검사일": str(run_row.get("test_date", "") or ""),
                        }

                        index_cla_com_db = put_index_cla_and_com(test_type_db, index_scores_db)
                        subtest_cla_com_db = put_subtest_cla_and_com(test_type_db, subtest_scores_db)
                        new_prompt = build_prompt(
                            test_type_db,
                            index_cla_com_db,
                            subtest_cla_com_db,
                            examinee_info_db,
                        )

                        try:
                            with st.spinner("선택한 보고서를 다시 생성하는 중입니다..."):
                                new_report = generate_report(new_prompt)

                            update_final_report(
                                test_id=selected_test_id,
                                prompt=new_prompt,
                                final_report=new_report,
                                model_name=MODEL_NAME,
                            )

                            st.success("보고서 재생성 및 DB 업데이트 완료")
                            st.rerun()

                        except Exception as e:
                            st.error(f"재생성 중 오류가 발생했습니다: {e}")

                with col2:
                    delete_confirm = st.checkbox(
                        "삭제 확인",
                        key=f"delete_confirm_{selected_test_id}"
                    )

                    if st.button("선택한 검사 삭제", key=f"delete_{selected_test_id}", use_container_width=True):
                        if not delete_confirm:
                            st.warning("삭제 확인 체크를 먼저 해주세요.")
                        else:
                            try:
                                delete_test_run(selected_test_id)
                                st.success("검사 데이터 삭제 완료")
                                st.rerun()
                            except Exception as e:
                                st.error(f"삭제 중 오류가 발생했습니다: {e}")

except Exception as e:
    st.error(f"저장 이력 조회 중 오류가 발생했습니다: {e}")


# =========================
# 하단 디버그용
# =========================
with st.expander("룩업 테이블 미리보기"):
    index_df, subtest_df = get_test_frames(test_type)
    st.write("지표 테이블", index_df.head())
    st.write("소검사 테이블", subtest_df.head())


# =========================
# DB 파일 안내
# =========================
with st.expander("DB 파일 안내 / 열어보는 방법"):
    st.markdown(f"""
현재 앱은 `{DB_PATH}` 파일에 SQLite로 저장합니다.

로컬에서 열어보는 방법 예시

1. DB 파일 위치 확인  
- 이 코드 파일과 같은 폴더에 `{DB_PATH}`가 생성됩니다.

2. Python으로 확인
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("psycolor.db")

print(pd.read_sql_query("SELECT * FROM test_run", conn))
print(pd.read_sql_query("SELECT * FROM test_result", conn))
print(pd.read_sql_query("SELECT * FROM final_report", conn))

conn.close()""")