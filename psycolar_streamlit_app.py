import os
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Psycolar 보고서 생성기", page_icon="🧠", layout="wide")

INDEX_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=0"
SUBTEST_CSV_URL = "https://docs.google.com/spreadsheets/d/1rAgPIi_o0NsBfF89wAbUr3hwg0PX2w115twdyW9p2BQ/export?format=csv&gid=978787284"

MODEL_NAME = "gpt-5-mini"
SELECTION = {
    "K-WPPSI-IV_A": {"VCI": ["RV", "PN"], "PSI": ["BD", "OA"], "FSIQ": []},
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


def get_api_key():
    # 1) 환경변수 먼저 확인
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 2) Streamlit secrets 확인
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


@st.cache_data

def load_lookup_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_i = pd.read_csv(INDEX_CSV_URL)
    df_s = pd.read_csv(SUBTEST_CSV_URL)
    return df_i, df_s


@st.cache_data

def get_test_frames(test_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_i, df_s = load_lookup_tables()
    return df_i[df_i["test_type"] == test_type].copy(), df_s[df_s["test_type"] == test_type].copy()



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
                result[index_code] = {str(row["classification"]): str(row["comment"])}
                break
    return result



def put_subtest_cla_and_com(test_type: str, subtest_scores: Dict[str, int]) -> Dict[str, Dict[str, str]]:
    _, subtest_df = get_test_frames(test_type)
    result: Dict[str, Dict[str, str]] = {}

    for subtest_code, score in subtest_scores.items():
        matched_rows = subtest_df[subtest_df["subtest_code"] == subtest_code]
        for _, row in matched_rows.iterrows():
            if row["min_scaled_score"] <= score <= row["max_scaled_score"]:
                result[subtest_code] = {str(row["classification"]): str(row["comment"])}
                break
    return result



def build_prompt(test_type: str, index_cla_com: Dict[str, Dict[str, str]], subtest_cla_com: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("너는 심리검사 보고서 문장 정리 도우미다.")
    lines.append("반드시 제공된 정보만 사용하라.")
    lines.append("없는 사실을 추론하지 마라.")
    lines.append("진단명이나 치료 권고를 임의로 추가하지 마라.")
    lines.append("공식적이고 자연스러운 한국어 보고서 문체로 작성하라.")
    lines.append(f"검사 유형은 {test_type}이다.")
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


# ---------------- UI ----------------
st.title("🧠 Psycolar 보고서 생성기")
st.caption("룩업 테이블 + OpenAI API로 아주 간단하게 결과 문단을 생성하는 테스트용 화면")

with st.expander("사용 전 확인", expanded=True):
    st.markdown(
        """
- API 키는 코드에 직접 적지 말고 `OPENAI_API_KEY`로 넣는 걸 권장합니다.
- 이 화면은 빠르게 동작 확인해보는 MVP용입니다.
- 현재는 검사 유형 선택 → 점수 입력 → 룩업 매핑 → 보고서 문단 생성 흐름만 넣었습니다.
        """
    )

left, right = st.columns([1, 1])

with left:
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

    generate_clicked = st.button("보고서 생성", type="primary", use_container_width=True)

with right:
    st.subheader("중간 결과")
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
    prompt = build_prompt(test_type, index_cla_com, subtest_cla_com)

    st.divider()
    st.subheader("룩업 매핑 결과")
    st.write("지표 분류/코멘트", index_cla_com)
    st.write("소검사 분류/코멘트", subtest_cla_com)

    st.subheader("생성 프롬프트")
    st.code(prompt, language="text")

    try:
        with st.spinner("AI가 보고서를 생성하는 중입니다..."):
            final_report = generate_report(prompt)

        st.subheader("최종 보고서")
        st.text_area("생성 결과", final_report, height=260)
    except Exception as e:
        st.error(f"생성 중 오류가 발생했습니다: {e}")


# 하단 디버그용
with st.expander("룩업 테이블 미리보기"):
    index_df, subtest_df = get_test_frames(test_type)
    st.write("지표 테이블", index_df.head())
    st.write("소검사 테이블", subtest_df.head())
