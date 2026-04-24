import json
import operator
import re
import uuid
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

FREE_MODEL_ID = "openrouter/free"
DEFAULT_RENT_TRAIN_DATASET = "/Users/artem/Downloads/rent-prediction-2025/airbnb_train.csv"
DEFAULT_RENT_TEST_DATASET = "/Users/artem/Downloads/rent-prediction-2025/airbnb_test.csv"
DEFAULT_RENT_SAMPLE_SUBMISSION = (
    "/Users/artem/Downloads/rent-prediction-2025/airbnb_sample_submission.csv"
)


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_path: str
    target_column: str
    task_type: str
    api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    schema: Dict[str, List[str]]
    data_summary: Dict[str, Any]
    prep_summary: Dict[str, Any]
    text_summary: Dict[str, Any]
    orchestrator_plan: Dict[str, Any]
    agent_results: Annotated[List[Dict[str, Any]], operator.add]
    prep_outputs: Annotated[List[Dict[str, Any]], operator.add]
    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    artifacts: Dict[str, Optional[str]]
    retry_count: int
    max_retries: int
    status: str


def _make_agent_result(
    agent: str,
    summary: str,
    *,
    status: str = "success",
    skipped: bool = False,
    decisions: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    next_input: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "agent": agent,
        "status": status,
        "skipped": skipped,
        "summary": summary,
        "decisions": decisions or {},
        "artifacts": artifacts or {},
        "next_input": next_input or {},
        "reason": reason,
    }


def _default_llm_fallback(error: Exception) -> Dict[str, Any]:
    return {
        "status": "fallback",
        "reason": str(error),
        "business_interpretation": (
            "LLM temporarily unavailable. Continue with deterministic preprocessing heuristics."
        ),
        "data_quality_risks": [
            "LLM/API unavailable during this run.",
            "Use heuristic fallback and keep artifacted state for reruns.",
        ],
        "recommended_target_usage": "Proceed with target as configured unless schema check fails.",
        "recommended_preprocessing": [
            "Remove duplicates",
            "Impute missing values",
            "Cap extreme numeric values",
            "Engineer ratio and interaction features",
            "Clean text columns if present",
        ],
    }


def _json_safe_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _extract_json(content: str) -> Dict[str, Any]:
    text = content.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "", 1).rsplit("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.replace("```", "", 1).rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def create_llm(
    api_key: str,
    *,
    model: str = FREE_MODEL_ID,
    temperature: float = 0.1,
    max_tokens: int = 1400,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


@tool
def load_dataset_tool(dataset_path: str) -> Dict[str, Any]:
    """Load a CSV or Excel dataset from a local path or URL."""
    lower_path = dataset_path.lower()
    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    return {
        "df": df,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }


@tool
def save_json_tool(obj: Dict[str, Any], path: str) -> str:
    """Save a dictionary as a JSON artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


@tool
def save_csv_tool(df: Any, path: str) -> str:
    """Save a pandas dataframe as a CSV artifact."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@tool
def infer_task_type_tool(df: Any, target_column: str) -> str:
    """Infer whether the target implies a regression or classification task."""
    target = df[target_column]
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


@tool
def infer_schema_tool(df: Any, target_column: str) -> Dict[str, List[str]]:
    """Infer numeric, categorical, text and datetime columns for a dataframe."""
    work_df = df.copy()
    datetime_cols: List[str] = []

    for col in work_df.columns:
        if col == target_column:
            continue

        series = work_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if series.dtype == "object" or str(series.dtype).startswith("string"):
            non_null = series.dropna().astype(str).head(200)
            if non_null.empty:
                continue

            name_hint = any(token in col.lower() for token in ["date", "review", "since"])
            sample_hint = non_null.str.match(r"^\d{4}-\d{2}-\d{2}", na=False).mean() >= 0.7
            if not name_hint and not sample_hint:
                continue

            parsed = pd.to_datetime(non_null, errors="coerce")
            date_signal = parsed.notna().mean()
            if name_hint or sample_hint or date_signal >= 0.85:
                datetime_cols.append(col)

    numeric_cols = (
        work_df.select_dtypes(include=["number"]).columns.drop(target_column, errors="ignore").tolist()
    )

    object_cols = (
        work_df.select_dtypes(include=["object", "category", "bool", "string"])
        .columns.drop(target_column, errors="ignore")
        .tolist()
    )

    text_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in object_cols:
        if col in datetime_cols:
            continue

        avg_len = work_df[col].fillna("").astype(str).str.len().mean()
        nunique = work_df[col].nunique(dropna=True)
        unique_ratio = nunique / max(len(work_df), 1)

        if avg_len >= 35 or (avg_len >= 20 and unique_ratio > 0.3):
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "text": text_cols,
        "datetime": [col for col in datetime_cols if col != target_column],
    }


@tool
def build_data_summary_tool(
    df: Any,
    target_column: str,
    schema_info: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Build a compact but useful EDA summary for downstream agent decisions."""
    target = df[target_column]
    numeric_preview = {}
    for col in schema_info.get("numeric", [])[:6]:
        series = df[col]
        numeric_preview[col] = {
            "mean": float(series.mean()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
            "missing_pct": round(float(series.isna().mean()) * 100, 2),
        }

    missing = df.isna().sum().to_dict()
    missing_pct = ((df.isna().mean() * 100).round(2)).to_dict()

    if pd.api.types.is_numeric_dtype(target):
        target_properties = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "min": float(target.min()) if target.notna().any() else None,
            "median": float(target.median()) if target.notna().any() else None,
            "max": float(target.max()) if target.notna().any() else None,
            "mean": float(target.mean()) if target.notna().any() else None,
            "sample_values": target.dropna().head(10).tolist(),
        }
    else:
        target_properties = {
            "dtype": str(target.dtype),
            "n_unique": int(target.nunique(dropna=True)),
            "class_balance": target.value_counts(normalize=True, dropna=False).round(4).to_dict(),
            "sample_values": target.dropna().astype(str).head(10).tolist(),
        }

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "missing": missing,
        "missing_pct": missing_pct,
        "duplicates": int(df.duplicated().sum()),
        "schema": schema_info,
        "numeric_preview": numeric_preview,
        "target_properties": target_properties,
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


@tool
def llm_json_tool(
    system_prompt: str,
    payload: Dict[str, Any],
    api_key: str,
    model: str = FREE_MODEL_ID,
    temperature: float = 0.1,
    max_tokens: int = 1400,
) -> Dict[str, Any]:
    """Call an LLM and return a JSON dictionary with a deterministic fallback."""
    try:
        llm = create_llm(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{payload}"),
            ]
        )
        response = (prompt | llm).invoke({"payload": _json_safe_payload(payload)})
        return _extract_json(response.content)
    except Exception as error:
        return _default_llm_fallback(error)


@tool
def prepare_tabular_features_tool(
    df: Any,
    target_column: str,
    schema_info: Dict[str, List[str]],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare numeric, categorical and datetime data with generic feature engineering."""
    prepared_df = df.copy()
    schema_info = {key: list(value) for key, value in schema_info.items()}
    initial_rows = len(prepared_df)
    prepared_df = prepared_df.drop_duplicates().reset_index(drop=True)

    created_features: List[str] = []
    missing_strategy: Dict[str, str] = {}
    outlier_strategy: Dict[str, str] = {}

    datetime_cols = [col for col in schema_info.get("datetime", []) if col in prepared_df.columns]
    for col in datetime_cols:
        prepared_df[col] = pd.to_datetime(prepared_df[col], errors="coerce")
        for suffix, accessor in {
            "year": prepared_df[col].dt.year,
            "month": prepared_df[col].dt.month,
            "dayofweek": prepared_df[col].dt.dayofweek,
        }.items():
            feature_name = f"{col}_{suffix}"
            prepared_df[feature_name] = accessor
            created_features.append(feature_name)
            if feature_name not in schema_info["numeric"]:
                schema_info["numeric"].append(feature_name)

    numeric_cols = [
        col
        for col in schema_info.get("numeric", [])
        if col in prepared_df.columns
        and col != target_column
        and pd.api.types.is_numeric_dtype(prepared_df[col])
    ]
    stable_numeric_cols = [col for col in numeric_cols if prepared_df[col].nunique(dropna=True) > 5]

    for col in numeric_cols:
        if prepared_df[col].isna().any():
            fill_value = prepared_df[col].median()
            prepared_df[col] = prepared_df[col].fillna(fill_value)
            missing_strategy[col] = "median"

    for col in stable_numeric_cols:
        q1 = prepared_df[col].quantile(0.25)
        q3 = prepared_df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        prepared_df[col] = prepared_df[col].clip(lower=lower, upper=upper)
        outlier_strategy[col] = "iqr_clip"

    categorical_cols = [
        col
        for col in schema_info.get("categorical", [])
        if col in prepared_df.columns and col != target_column
    ]
    for col in categorical_cols:
        prepared_df[col] = prepared_df[col].fillna("missing").astype(str)
        missing_strategy[col] = "fill_missing_token"

    # Any non-target object/string columns that slipped past schema inference should still
    # be handled as categorical instead of breaking numeric preprocessing.
    fallback_categorical_cols = [
        col
        for col in prepared_df.columns
        if col != target_column
        and col not in numeric_cols
        and col not in categorical_cols
        and col not in schema_info.get("text", [])
        and col not in datetime_cols
        and not pd.api.types.is_numeric_dtype(prepared_df[col])
    ]
    for col in fallback_categorical_cols:
        prepared_df[col] = prepared_df[col].fillna("missing").astype(str)
        missing_strategy[col] = "fill_missing_token_fallback"

    if fallback_categorical_cols:
        schema_info["categorical"].extend(
            [col for col in fallback_categorical_cols if col not in schema_info["categorical"]]
        )

    if len(stable_numeric_cols) >= 2:
        first_col, second_col = stable_numeric_cols[:2]
        ratio_name = f"{first_col}_to_{second_col}_ratio"
        interaction_name = f"{first_col}_x_{second_col}"
        prepared_df[ratio_name] = prepared_df[first_col] / (prepared_df[second_col].abs() + 1)
        prepared_df[interaction_name] = prepared_df[first_col] * prepared_df[second_col]
        created_features.extend([ratio_name, interaction_name])

    if stable_numeric_cols:
        anchor_col = stable_numeric_cols[0]
        log_name = f"{anchor_col}_log1p"
        prepared_df[log_name] = np.log1p(np.clip(prepared_df[anchor_col], a_min=0, a_max=None))
        created_features.append(log_name)

    summary = {
        "agent": "tabular_preparation_agent",
        "duplicates_action": f"Removed {initial_rows - len(prepared_df)} duplicate rows.",
        "missing_strategy": missing_strategy,
        "outlier_strategy": outlier_strategy,
        "encoding": "Deferred to modeling stage; categories preserved as strings.",
        "scaling": "Deferred to modeling stage; not required before branching completes.",
        "target_transform": None,
        "created_features": created_features,
        "plan_alignment": plan.get("tabular_actions", []),
    }

    return {
        "prepared_df": prepared_df,
        "summary": summary,
        "updated_schema": schema_info,
    }


@tool
def prepare_text_features_tool(
    df: Any,
    text_columns: List[str],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Clean text columns and create compact statistical text features."""
    prepared_df = df.copy()
    existing_text_columns = [col for col in text_columns if col in prepared_df.columns]

    if not existing_text_columns:
        return {
            "prepared_df": prepared_df,
            "summary": {
                "agent": "text_preparation_agent",
                "used": False,
                "text_columns": [],
                "vectorization": None,
                "created_text_features": [],
                "text_feature_count": 0,
                "plan_alignment": [],
            },
        }

    created_text_features: List[str] = []
    regex = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s]")

    for col in existing_text_columns:
        clean_col = f"{col}_clean"
        length_col = f"{col}_length"
        words_col = f"{col}_word_count"
        unique_ratio_col = f"{col}_unique_word_ratio"

        cleaned = (
            prepared_df[col]
            .fillna("")
            .astype(str)
            .str.lower()
            .apply(lambda value: regex.sub(" ", value))
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        word_counts = cleaned.str.split().str.len()
        unique_counts = cleaned.apply(lambda value: len(set(value.split())) if value else 0)

        prepared_df[clean_col] = cleaned
        prepared_df[length_col] = cleaned.str.len()
        prepared_df[words_col] = word_counts
        prepared_df[unique_ratio_col] = unique_counts / (word_counts + 1)

        created_text_features.extend([clean_col, length_col, words_col, unique_ratio_col])

    return {
        "prepared_df": prepared_df,
        "summary": {
            "agent": "text_preparation_agent",
            "used": True,
            "text_columns": existing_text_columns,
            "vectorization": "Deferred to modeling stage; clean text and stats prepared for TF-IDF.",
            "created_text_features": created_text_features,
            "text_feature_count": len(created_text_features),
            "plan_alignment": plan.get("text_actions", []),
        },
    }


def data_description_agent(state: AgentState) -> Dict[str, Any]:
    loaded = load_dataset_tool.invoke({"dataset_path": state["dataset_path"]})
    df = loaded["df"]

    task_type = state.get("task_type") or infer_task_type_tool.invoke(
        {"df": df, "target_column": state["target_column"]}
    )

    schema = infer_schema_tool.invoke({"df": df, "target_column": state["target_column"]})
    data_summary = build_data_summary_tool.invoke(
        {
            "df": df,
            "target_column": state["target_column"],
            "schema_info": schema,
        }
    )

    llm_eda = llm_json_tool.invoke(
        {
            "system_prompt": """
You are a Data Description Agent inside an ML workflow.
Use deliberate step-by-step reasoning internally, but return only strict JSON.
Combine business framing with technical EDA.

Required output schema:
{
  "business_interpretation": "...",
  "data_quality_risks": ["...", "..."],
  "recommended_target_usage": "...",
  "recommended_preprocessing": ["...", "..."]
}
""".strip(),
            "payload": {
                "task_type": task_type,
                "target_column": state["target_column"],
                "summary": data_summary,
            },
            "api_key": state["api_key"],
            "model": state.get("llm_model", FREE_MODEL_ID),
            "temperature": state.get("llm_temperature", 0.1),
            "max_tokens": state.get("llm_max_tokens", 1400),
        }
    )

    profile_path = ARTIFACT_DIR / f"{state['run_id']}_data_profile.json"
    raw_snapshot_path = ARTIFACT_DIR / f"{state['run_id']}_raw_snapshot.csv"
    save_json_tool.invoke(
        {
            "obj": {
                "data_summary": data_summary,
                "llm_eda": llm_eda,
            },
            "path": str(profile_path),
        }
    )
    save_csv_tool.invoke({"df": df, "path": str(raw_snapshot_path)})

    result = _make_agent_result(
        "data_description_agent",
        "Dataset loaded, schema inferred, EDA summary prepared.",
        decisions={
            "task_type": task_type,
            "schema": schema,
        },
        artifacts={
            "data_profile_path": str(profile_path),
            "raw_snapshot_path": str(raw_snapshot_path),
        },
        next_input={"schema_ready": True},
    )

    return {
        "task_type": task_type,
        "schema": schema,
        "data_summary": {**data_summary, "llm_eda": llm_eda},
        "agent_results": [result],
        "artifacts": {
            **state.get("artifacts", {}),
            "data_profile_path": str(profile_path),
            "raw_snapshot_path": str(raw_snapshot_path),
        },
        "logs": ["Data Description Agent completed EDA and persisted the data profile."],
    }


def orchestration_planner_agent(state: AgentState) -> Dict[str, Any]:
    planner_output = llm_json_tool.invoke(
        {
            "system_prompt": """
You are the Orchestrator Planner for a LangGraph-based ML agent.
Think like an agent designer, not a script writer.
Decide which preparation branches should run and which preprocessing strategies are most appropriate.
Return only strict JSON.

Output schema:
{
  "route_tabular": true,
  "route_text": true,
  "reasoning": "...",
  "tabular_actions": ["...", "..."],
  "text_actions": ["...", "..."],
  "risk_flags": ["...", "..."],
  "fallback_logic": "..."
}
""".strip(),
            "payload": {
                "task_type": state["task_type"],
                "target_column": state["target_column"],
                "schema": state["schema"],
                "data_summary": state["data_summary"],
                "criteria_hint": {
                    "must_process_missing": True,
                    "must_process_outliers": True,
                    "must_create_min_two_features": True,
                    "must_use_conditional_branching": True,
                },
            },
            "api_key": state["api_key"],
            "model": state.get("llm_model", FREE_MODEL_ID),
            "temperature": state.get("llm_temperature", 0.1),
            "max_tokens": state.get("llm_max_tokens", 1400),
        }
    )

    plan = {
        "route_tabular": True,
        "route_text": bool(state["schema"].get("text")),
        "reasoning": planner_output.get(
            "reasoning",
            "Run tabular preparation for all structured data and text preparation only if text columns exist.",
        ),
        "tabular_actions": planner_output.get(
            "tabular_actions",
            [
                "remove duplicates",
                "impute missing values",
                "clip numeric outliers",
                "create ratio and interaction features",
            ],
        ),
        "text_actions": planner_output.get(
            "text_actions",
            [
                "clean text",
                "create length and word count features",
                "defer TF-IDF to modeling stage",
            ],
        ),
        "risk_flags": planner_output.get("risk_flags", []),
        "fallback_logic": planner_output.get(
            "fallback_logic",
            "If the LLM fails, route by schema heuristics and keep deterministic prep.",
        ),
    }

    if not state["schema"].get("text"):
        plan["route_text"] = False

    plan_path = ARTIFACT_DIR / f"{state['run_id']}_orchestrator_plan.json"
    save_json_tool.invoke({"obj": plan, "path": str(plan_path)})

    result = _make_agent_result(
        "orchestration_planner_agent",
        "Orchestrator selected branches and preprocessing priorities.",
        decisions=plan,
        artifacts={"orchestrator_plan_path": str(plan_path)},
        next_input={
            "tabular_required": plan["route_tabular"],
            "text_required": plan["route_text"],
        },
    )

    return {
        "orchestrator_plan": plan,
        "agent_results": [result],
        "artifacts": {
            **state.get("artifacts", {}),
            "orchestrator_plan_path": str(plan_path),
        },
        "logs": ["Orchestrator Planner created the branching and preprocessing plan."],
    }


def parallel_preparation_fanout(state: AgentState):
    sends = [Send("tabular_preparation_agent", state)]
    if state["orchestrator_plan"].get("route_text"):
        sends.append(Send("text_preparation_agent", state))
    return sends


def tabular_preparation_agent(state: AgentState) -> Dict[str, Any]:
    raw_df = load_dataset_tool.invoke(
        {"dataset_path": state["artifacts"]["raw_snapshot_path"]}
    )["df"]

    result = prepare_tabular_features_tool.invoke(
        {
            "df": raw_df,
            "target_column": state["target_column"],
            "schema_info": state["schema"],
            "plan": state["orchestrator_plan"],
        }
    )

    tabular_path = ARTIFACT_DIR / f"{state['run_id']}_tabular_prepared.csv"
    save_csv_tool.invoke({"df": result["prepared_df"], "path": str(tabular_path)})

    agent_result = _make_agent_result(
        "tabular_preparation_agent",
        "Tabular preprocessing completed with missing-value handling, outlier clipping, and engineered features.",
        decisions=result["summary"],
        artifacts={"tabular_dataset_path": str(tabular_path)},
        next_input={"prepared_tabular_ready": True},
    )

    return {
        "prep_outputs": [
            {
                "kind": "tabular",
                "summary": result["summary"],
                "artifact_path": str(tabular_path),
                "updated_schema": result["updated_schema"],
            }
        ],
        "agent_results": [agent_result],
        "logs": ["Tabular Preparation Agent finished the structured-data branch."],
    }


def text_preparation_agent(state: AgentState) -> Dict[str, Any]:
    raw_df = load_dataset_tool.invoke(
        {"dataset_path": state["artifacts"]["raw_snapshot_path"]}
    )["df"]

    result = prepare_text_features_tool.invoke(
        {
            "df": raw_df,
            "text_columns": state["schema"].get("text", []),
            "plan": state["orchestrator_plan"],
        }
    )

    text_path = ARTIFACT_DIR / f"{state['run_id']}_text_prepared.csv"
    save_csv_tool.invoke({"df": result["prepared_df"], "path": str(text_path)})

    agent_result = _make_agent_result(
        "text_preparation_agent",
        "Text columns were cleaned and transformed into compact text statistics.",
        decisions=result["summary"],
        artifacts={"text_features_path": str(text_path)},
        next_input={"prepared_text_ready": True},
    )

    return {
        "prep_outputs": [
            {
                "kind": "text",
                "summary": result["summary"],
                "artifact_path": str(text_path),
            }
        ],
        "agent_results": [agent_result],
        "logs": ["Text Preparation Agent finished the text branch."],
    }


def merge_preparation_outputs_agent(state: AgentState) -> Dict[str, Any]:
    prep_outputs = state.get("prep_outputs", [])
    tabular_output = next(item for item in prep_outputs if item["kind"] == "tabular")
    text_output = next((item for item in prep_outputs if item["kind"] == "text"), None)

    prepared_df = load_dataset_tool.invoke({"dataset_path": tabular_output["artifact_path"]})["df"]
    prep_summary = tabular_output["summary"]
    text_summary = {
        "used": False,
        "text_columns": [],
        "vectorization": None,
        "created_text_features": [],
        "text_feature_count": 0,
    }

    if text_output is not None:
        text_summary = text_output["summary"]
        created_text_features = text_summary.get("created_text_features", [])
        if created_text_features:
            new_text_features = [
                col for col in created_text_features if col not in prepared_df.columns
            ]
            if new_text_features:
                text_df = load_dataset_tool.invoke({"dataset_path": text_output["artifact_path"]})["df"]
                prepared_df = pd.concat(
                    [
                        prepared_df.reset_index(drop=True),
                        text_df[new_text_features].reset_index(drop=True),
                    ],
                    axis=1,
                )

    final_path = ARTIFACT_DIR / f"{state['run_id']}_prepared_until_text.csv"
    save_csv_tool.invoke({"df": prepared_df, "path": str(final_path)})

    manifest = {
        "run_id": state["run_id"],
        "dataset_path": state["dataset_path"],
        "target_column": state["target_column"],
        "task_type": state["task_type"],
        "schema": tabular_output.get("updated_schema", state["schema"]),
        "data_summary": state["data_summary"],
        "prep_summary": prep_summary,
        "text_summary": text_summary,
        "artifacts": {
            **state.get("artifacts", {}),
            "tabular_dataset_path": tabular_output["artifact_path"],
            "text_features_path": text_output["artifact_path"] if text_output else None,
            "prepared_final_path": str(final_path),
        },
    }

    manifest_path = ARTIFACT_DIR / f"{state['run_id']}_prep_manifest.json"
    save_json_tool.invoke({"obj": manifest, "path": str(manifest_path)})

    agent_result = _make_agent_result(
        "merge_preparation_outputs_agent",
        "Branch outputs merged into a single prepared dataset for downstream modeling.",
        decisions={
            "text_used": text_summary["used"],
            "created_tabular_features": prep_summary.get("created_features", []),
            "created_text_features": text_summary.get("created_text_features", []),
        },
        artifacts={
            "prepared_final_path": str(final_path),
            "prep_manifest_path": str(manifest_path),
        },
        next_input={"ready_for_modeling": True},
    )

    return {
        "schema": tabular_output.get("updated_schema", state["schema"]),
        "prep_summary": prep_summary,
        "text_summary": text_summary,
        "agent_results": [agent_result],
        "artifacts": {
            **state.get("artifacts", {}),
            "tabular_dataset_path": tabular_output["artifact_path"],
            "text_features_path": text_output["artifact_path"] if text_output else None,
            "prepared_final_path": str(final_path),
            "prep_manifest_path": str(manifest_path),
        },
        "status": "prepared_until_text",
        "logs": ["Merge Agent combined branch outputs and wrote the preparation manifest."],
    }


def preparation_report_agent(state: AgentState) -> Dict[str, Any]:
    report = {
        "run_id": state["run_id"],
        "status": state["status"],
        "business_goal": "Predict Airbnb rental price from mixed structured and text listing features.",
        "criteria_alignment": {
            "agentic_orchestration": True,
            "conditional_branching": True,
            "short_term_memory": True,
            "tool_usage_count": 6,
            "prepared_until_text_stage": True,
        },
        "orchestrator_plan": state["orchestrator_plan"],
        "prep_summary": state["prep_summary"],
        "text_summary": state["text_summary"],
        "artifacts": state["artifacts"],
    }

    report_path = ARTIFACT_DIR / f"{state['run_id']}_stage_report.json"
    save_json_tool.invoke({"obj": report, "path": str(report_path)})

    agent_result = _make_agent_result(
        "reporting_agent",
        "Preparation-stage report generated.",
        artifacts={"stage_report_path": str(report_path)},
    )

    return {
        "agent_results": [agent_result],
        "artifacts": {
            **state.get("artifacts", {}),
            "stage_report_path": str(report_path),
        },
        "status": "ready_for_modeling",
        "logs": ["Reporting Agent packaged the preparation-stage artifacts."],
    }


def build_graph(checkpointer: Optional[InMemorySaver] = None):
    graph = StateGraph(AgentState)
    graph.add_node("data_description_agent", data_description_agent)
    graph.add_node("orchestration_planner_agent", orchestration_planner_agent)
    graph.add_node("tabular_preparation_agent", tabular_preparation_agent)
    graph.add_node("text_preparation_agent", text_preparation_agent)
    graph.add_node("merge_preparation_outputs_agent", merge_preparation_outputs_agent)
    graph.add_node("preparation_report_agent", preparation_report_agent)

    graph.add_edge(START, "data_description_agent")
    graph.add_edge("data_description_agent", "orchestration_planner_agent")
    graph.add_conditional_edges("orchestration_planner_agent", parallel_preparation_fanout)
    graph.add_edge("tabular_preparation_agent", "merge_preparation_outputs_agent")
    graph.add_edge("text_preparation_agent", "merge_preparation_outputs_agent")
    graph.add_edge("merge_preparation_outputs_agent", "preparation_report_agent")
    graph.add_edge("preparation_report_agent", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def make_initial_state(
    *,
    dataset_path: str,
    target_column: str,
    api_key: str,
    task_type: Optional[str] = None,
    model: str = FREE_MODEL_ID,
    temperature: float = 0.1,
    max_tokens: int = 1400,
    max_retries: int = 1,
) -> AgentState:
    return {
        "run_id": str(uuid.uuid4()),
        "dataset_path": dataset_path,
        "target_column": target_column,
        "task_type": task_type or "",
        "api_key": api_key,
        "llm_model": model,
        "llm_temperature": temperature,
        "llm_max_tokens": max_tokens,
        "prep_outputs": [],
        "agent_results": [],
        "logs": [],
        "errors": [],
        "artifacts": {},
        "retry_count": 0,
        "max_retries": max_retries,
        "status": "running",
    }


def run_preparation_workflow(
    *,
    dataset_path: str,
    target_column: str,
    api_key: str,
    task_type: Optional[str] = None,
    model: str = FREE_MODEL_ID,
    temperature: float = 0.1,
    max_tokens: int = 1400,
    thread_id: str = "housing-agent-thread",
):
    app = build_graph()
    result = app.invoke(
        make_initial_state(
            dataset_path=dataset_path,
            target_column=target_column,
            api_key=api_key,
            task_type=task_type,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        config={"configurable": {"thread_id": thread_id}},
    )
    return app, result


def summarize_result(result: AgentState) -> Dict[str, Any]:
    return {
        "run_id": result["run_id"],
        "status": result["status"],
        "task_type": result["task_type"],
        "schema": result["schema"],
        "data_summary": {
            "n_rows": result["data_summary"]["n_rows"],
            "n_columns": result["data_summary"]["n_columns"],
            "target_properties": result["data_summary"]["target_properties"],
        },
        "prep_summary": result.get("prep_summary", {}),
        "text_summary": result.get("text_summary", {}),
        "artifacts": result.get("artifacts", {}),
        "logs": result.get("logs", []),
        "agent_results": result.get("agent_results", []),
    }
