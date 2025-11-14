import sys
import os
import pickle
from dotenv import load_dotenv
from textwrap import dedent

import pandas as pd
import streamlit as st
from openai import OpenAI

# carregar .env
load_dotenv()

# ajustar caminho do projeto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# imports do projeto
from core.pipeline import run_training_pipeline, run_prediction_pipeline, load_data
from core.chatbot.rules import answer_from_metrics

# === Helpers OpenAI ===
def get_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            key = st.secrets["openai_api_key"]
        except Exception:
            key = None
    return key

def get_client():
    k = get_api_key()
    if not k:
        st.error("❌ OPENAI_API_KEY não definida. Configure no .env ou em .streamlit/secrets.toml")
        st.stop()
    os.environ["OPENAI_API_KEY"] = k
    return OpenAI()

# === Paths e constantes ===
MODEL_DIR = "models_store"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model_sales.pkl")

# === Streamlit config ===
st.set_page_config(page_title="📊 E-commerce ML - Chat", layout="wide")
st.title("📊 E-commerce ML")

# === Estado da sessão (inicializações) ===
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "prediction_df" not in st.session_state:
    st.session_state.prediction_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "importances" not in st.session_state:
    st.session_state.importances = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Oi! Eu sou o bot do E-commerce. Envie seus dados para começar."}
    ]
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []

# util: converter df -> csv
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# === SIDEBAR: Configs e Ações ===
with st.sidebar:
    st.header("Configurações")
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
    max_ctx = st.slider("Limite do contexto (caracteres)", 500, 12000, 4000, step=500)
    show_ctx = st.checkbox("Mostrar contexto gerado", value=False)
    model_api = st.selectbox("Modelo (API)", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    sys_prompt = st.text_area(
        "System prompt (comportamento do assistente)",
        value="Você é um analista de dados especializado em e-commerce. Para cada pergunta do usuário, responda com os dados solicitados e acrescente insights interpretativos relevantes, como padrões de venda, tendências de preço, oportunidades de negócio, diferenças entre países ou comportamentos de clientes. Sempre baseie os insights nos dados fornecidos, seja claro e objetivo, e destaque informações importantes que o usuário possa não perceber imediatamente.",
        height=120
    )
    st.markdown("---")
    st.header("Ações")
    uploaded_file = st.file_uploader("📁 Faça upload do CSV (dataset)", type=["csv"])
    train_btn = st.button("🚀 Treinar modelo (com ETL / pipeline)")
    predict_btn = st.button("📦 Carregar modelo e prever")
    reset_btn = st.button("♻️ Resetar modelo, banco e sessão")
    st.markdown("---")
    st.caption("Modo de Chat: escolha na aba Chat qual fluxo usar (LLM / RAG / ML)")

# === Upload / Treinar / Prever / Reset ===
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.warning(f"Erro ao ler CSV: {e}")

if reset_btn:
    for key in ["model_trained","predictions_made","prediction_df","metrics","importances","chat_history","api_messages"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.chat_history = [{"role":"assistant","content":"Oi! Eu sou o bot do E-commerce. Envie seus dados para começar."}]
    st.success("Sessão e modelos resetados. Recarregue a página se necessário.")
    st.experimental_rerun()

if train_btn:
    if df is None:
        st.warning("Faça upload de um CSV antes de treinar.")
    else:
        with st.spinner("Treinando modelo... (rodando pipeline)"):
            metrics, model_obj = run_training_pipeline(df, test_size=test_size, model_path=MODEL_PATH)
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            try:
                if model_obj is not None and hasattr(model_obj, "feature_importances_"):
                    importances = pd.DataFrame({
                        "feature": getattr(model_obj, "feature_names_in_", None) or list(range(len(model_obj.feature_importances_))),
                        "importance": model_obj.feature_importances_
                    }).sort_values("importance", ascending=False)
                    st.session_state.importances = importances.reset_index(drop=True)
                else:
                    st.session_state.importances = None
            except Exception:
                st.session_state.importances = None
        st.success("✅ Modelo treinado e salvo!")

if predict_btn:
    if df is None:
        st.warning("Faça upload de um CSV antes de prever.")
    elif not os.path.exists(MODEL_PATH):
        st.error("Modelo não encontrado. Treine um modelo primeiro.")
    else:
        with st.spinner("Realizando predições..."):
            df_pred = run_prediction_pipeline(df, MODEL_PATH)
            st.session_state.prediction_df = df_pred
            st.session_state.predictions_made = True
        st.success("✅ Predições realizadas!")

# === Tabs ===
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Predições", "💬 Chat"])

# === Tab: Treino ===
with tab_train:
    st.header("📊 Resultados do Treino")
    if not st.session_state.get("model_trained") or st.session_state.get("metrics") is None:
        st.info("Treine um modelo para ver resultados.")
    else:
        st.subheader("📈 Métricas")
        st.json(st.session_state.metrics)
        st.subheader("🔎 Importâncias (Top 20)")
        if st.session_state.get("importances") is not None:
            st.dataframe(st.session_state.importances.head(20), use_container_width=True)
        else:
            st.info("Nenhuma importância disponível para este modelo.")

# === Tab: Predições ===
with tab_predict:
    st.header("🚀 Predições")
    if not st.session_state.get("predictions_made"):
        st.info("Faça predições usando o modelo treinado.")
    else:
        df_to_show = st.session_state.prediction_df.copy()
        st.dataframe(df_to_show, use_container_width=True)
        csv_data = convert_df_to_csv(df_to_show)
        st.download_button("⬇️ Baixar Previsões em CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")

# === Funções de contexto ===
def numeric_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0: return "(Sem colunas numéricas)"
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    return desc[["count","mean","median","std","min","max"]].head(20).to_string()

def categorical_summary(df: pd.DataFrame, top_k: int = 5) -> str:
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns
    if len(cat_cols) == 0: return "(Sem colunas categóricas)"
    lines = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(top_k)
        lines.append(f"Coluna: {c}\n{vc.to_string()}\n")
    return "\n".join(lines)

def build_context_from_df(df: pd.DataFrame, max_chars: int = 4000, target_col: str = None) -> str:
    parts = [f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas",
             "\n[Resumo numérico]\n" + numeric_summary(df),
             "\n[Resumo categórico]\n" + categorical_summary(df)]
    try: parts.append("\n[Exemplo - 5 primeiras linhas]\n" + df.head(5).to_string())
    except Exception: pass
    ctx = "\n\n".join(parts)
    return ctx[:max_chars] + ("\n... (contexto truncado)" if len(ctx) > max_chars else "")

# === Função de insights on-demand ===
def get_insight(user_prompt: str, df: pd.DataFrame) -> str:
    prompt = user_prompt.lower()
    if "produto mais caro" in prompt or "mais caro" in prompt:
        linha = df.loc[df["UnitPrice"].idxmax()]
        return f"💰 Produto mais caro: {linha['Description']} (Código: {linha['StockCode']}), preço unitário: {linha['UnitPrice']}, país: {linha['Country']}"
    elif "mais vendidos" in prompt or "top produtos" in prompt:
        import re
        n = 5
        match = re.search(r'\b\d+\b', prompt)
        if match: n = int(match.group())
        top = df.groupby(['StockCode','Description'])['Quantity'].sum().sort_values(ascending=False).head(n)
        lines = [f"{i+1}. {desc} (Código: {code}) - {qty} unidades vendidas" 
                 for i, ((code, desc), qty) in enumerate(top.items())]
        return "🏆 Top produtos mais vendidos:\n" + "\n".join(lines)
    elif "mais caro" in prompt and "por país" in prompt:
        countries = df['Country'].unique()
        lines = []
        for c in countries:
            sub = df[df['Country'] == c]
            if not sub.empty:
                linha = sub.loc[sub['UnitPrice'].idxmax()]
                lines.append(f"{c}: {linha['Description']} - {linha['UnitPrice']}")
        return "💎 Produto mais caro por país:\n" + "\n".join(lines)
    else:
        return None

# === Tab: Chat ===
with tab_chat:
    st.header("💬 Conversar (RAG)")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Digite sua pergunta")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        try:
            df_context = load_data("spec_sales")
        except Exception:
            df_context = None

        if df_context is None or df_context.empty:
            reply = "Tabela 'spec_sales' não encontrada. Execute o pipeline ou forneça os dados."
        else:
            insight_reply = get_insight(user_prompt, df_context)
            if insight_reply:
                reply = insight_reply
            else:
                context_text = build_context_from_df(df_context, max_chars=max_ctx)
                msgs = [{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Contexto da base:\n{context_text}"}]
                for m in st.session_state.chat_history[-12:]:
                    msgs.append({"role": m["role"], "content": m["content"]})
                if st.session_state.importances is not None:
                    msgs.append({"role": "user", "content": f"Top importances:\n{st.session_state.importances.head(10).to_string(index=False)}"})
                if st.session_state.metrics is not None:
                    msgs.append({"role": "user", "content": f"Métricas do modelo:\n{st.session_state.metrics}"})
                try:
                    client = get_client()
                    resp = client.chat.completions.create(
                        model=model_api,
                        messages=msgs,
                        temperature=0.2,
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Erro na API: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.api_messages = msgs if 'msgs' in locals() else []
        st.rerun()
