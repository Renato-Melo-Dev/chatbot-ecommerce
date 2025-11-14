# app/main.py
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

# ajustar caminho do projeto (se necessário)
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
st.set_page_config(page_title="📊 E-commerce ML - Cha", layout="wide")
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
    st.session_state.api_messages = []  # mensagens que serão enviadas para a API (system + contexto + histórico)

# util: converter df -> csv
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# === SIDEBAR: Configs e Ações ===
with st.sidebar:
    st.header("Configurações")

    # Treino / Test split slider
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)

    # RAG / contexto
    max_ctx = st.slider("Limite do contexto (caracteres)", 500, 12000, 4000, step=500)
    show_ctx = st.checkbox("Mostrar contexto gerado", value=False)

    # Seleção de modelo de API
    model_api = st.selectbox("Modelo (API)", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)

    # System prompt editável
    sys_prompt = st.text_area("System prompt (comportamento do assistente)",
                             value="Você é um analista de dados. Use o contexto fornecido para responder de forma precisa e objetiva.",
                             height=120)

    st.markdown("---")
    st.header("Ações")

    uploaded_file = st.file_uploader("📁 Faça upload do CSV (dataset)", type=["csv"])
    train_btn = st.button("🚀 Treinar modelo (com ETL / pipeline)")
    predict_btn = st.button("📦 Carregar modelo e prever")
    reset_btn = st.button("♻️ Resetar modelo, banco e sessão")

    st.markdown("---")
    st.caption("Modo de Chat: escolha na aba Chat qual fluxo usar (LLM / RAG / ML)")

# === Upload / treinar / prever / reset ===
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.warning(f"Erro ao ler CSV: {e}")

if reset_btn:
    # reset simples
    for key in ["model_trained","predictions_made","prediction_df","metrics","importances","chat_history","api_messages"]:
        if key in st.session_state:
            del st.session_state[key]
    # recriar mensagens iniciais
    st.session_state.chat_history = [{"role":"assistant","content":"Oi! Eu sou o bot do E-commerce. Envie seus dados para começar."}]
    st.success("Sessão e modelos resetados. Recarregue a página se necessário.")
    st.experimental_rerun()

if train_btn:
    if df is None:
        st.warning("Faça upload de um CSV antes de treinar.")
    else:
        with st.spinner("Treinando modelo... (rodando pipeline)"):
            metrics, model_obj = run_training_pipeline(df, test_size=test_size, model_path=MODEL_PATH)
            # run_training_pipeline deve retornar métricas e o objeto treinado (ou salvar em disco)
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            # tentar extrair importances — isso depende de como o pipeline salva
            try:
                # se model_obj existir e tiver attribute feature_importances_
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

# === Layout principal: tabs ===
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Predições", "💬 Chat"])

# === Tab: Treino / Métricas / Importances ===
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

# === Funções de construção de contexto (RAG) ===
def numeric_summary(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return "(Sem colunas numéricas)"
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    cols = ["count","mean","median","std","min","max"]
    return desc[cols].head(20).to_string()

def categorical_summary(df: pd.DataFrame, top_k: int = 5) -> str:
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns
    if len(cat_cols) == 0:
        return "(Sem colunas categóricas)"
    lines = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(top_k)
        lines.append(f"Coluna: {c}\n{vc.to_string()}\n")
    return "\n".join(lines)

def correlation_with_target(df: pd.DataFrame, target_col: str, top_n: int = 10) -> str:
    if target_col not in df.columns:
        return f"(Coluna alvo '{target_col}' não encontrada.)"
    try:
        t = pd.to_numeric(df[target_col], errors="coerce")
        num_cols = df.select_dtypes(include="number").columns
        corrs = []
        for c in num_cols:
            if c == target_col:
                continue
            corr = t.corr(pd.to_numeric(df[c], errors="coerce"))
            if pd.notna(corr):
                corrs.append((c, corr))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        lines = [f"{c}: {v:.3f}" for c,v in corrs[:top_n]]
        return "\n".join(lines) if lines else "(Sem correlações calculáveis)"
    except Exception as e:
        return f"(Erro ao calcular correlações: {e})"

def build_context_from_df(df: pd.DataFrame, max_chars: int = 4000, target_col: str = None) -> str:
    parts = []
    parts.append(f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas")
    parts.append("\n[Resumo numérico]\n" + numeric_summary(df))
    parts.append("\n[Resumo categórico]\n" + categorical_summary(df))
    if target_col:
        parts.append(f"\n[Correlação com '{target_col}']\n" + correlation_with_target(df, target_col))
    try:
        sample_text = df.head(5).to_string()
        parts.append("\n[Exemplo - 5 primeiras linhas]\n" + sample_text)
    except Exception:
        pass
    ctx = "\n\n".join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars] + "\n... (contexto truncado)"
    return ctx

# === Tab: Chat (múltiplos modos) ===
with tab_chat:
    st.header("💬 Converse com o Modelo")

    # escolher modo: LLM puro / RAG (base) / ML (respostas baseadas no modelo local)
    chat_mode = st.selectbox("Modo de Chat", ["LLM (API)", "RAG (contexto da base)", "ML (respostas do modelo local)"])

    # exibir histórico
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # novo prompt do usuário
    user_prompt = st.chat_input("Digite sua pergunta (ex: 'Quais as métricas?' / 'Qual a importância da feature X?')")

    if user_prompt:
        # append display user
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # carregamentos necessários
        df_spec = None
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    model_pipe = pickle.load(f)
            except Exception:
                model_pipe = None
        else:
            model_pipe = None

        # ROTEIRO por modo
        if chat_mode == "ML (respostas do modelo local)":
            # usar sua função rules.py para gerar resposta baseada em métricas / modelo
            try:
                response = answer_from_metrics(
                    question=user_prompt,
                    metrics_df_or_dict=st.session_state.metrics,
                    importances_df=st.session_state.importances,
                    model_pipe=model_pipe,
                    df_spec=load_data("spec_sales") if "spec_sales" in [f[:-3] for f in os.listdir("data") if f.endswith(".py")] else None
                )
            except Exception:
                # fallback simples: usar metrics/importances se disponível
                if st.session_state.metrics is not None:
                    response = f"Métricas: {st.session_state.metrics}\n\nImportances: {st.session_state.importances if st.session_state.importances is not None else '(não disponível)'}"
                else:
                    response = "Não há métricas/modelo disponível para responder. Treine um modelo primeiro."
            st.session_state.chat_history.append({"role":"assistant","content":response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.rerun()

        elif chat_mode == "RAG (contexto da base)":
            # carregar tabela spec (se existir)
            df_context = None
            try:
                df_context = load_data("spec_sales")
            except Exception:
                df_context = None

            if df_context is None or df_context.empty:
                reply = "Tabela 'spec_sales' não encontrada. Treine/execute o pipeline para gerar a tabela final (ou faça upload/ETL)."
                st.session_state.chat_history.append({"role":"assistant","content":reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
                st.rerun()
            else:
                context_text = build_context_from_df(df_context, max_chars=max_ctx, target_col=None)
                if show_ctx:
                    with st.expander("Ver contexto (resumo da base)"):
                        st.text(context_text)

                # montar mensagens para a API: system + contexto + histórico + pergunta
                msgs = []
                msgs.append({"role":"system","content": sys_prompt})
                msgs.append({"role":"user","content": f"Contexto da base:\n{context_text}"})

                # enviar histórico (limitado) - apenas últimos N mensagens para evitar token explosion
                history_limit = 12
                for m in st.session_state.chat_history[-history_limit:]:
                    if m["role"] in ("user","assistant"):
                        msgs.append({"role": m["role"], "content": m["content"]})

                # incluir importances/metrics para auxiliar
                if st.session_state.importances is not None:
                    # enviar top 10 importances como tabela curta
                    try:
                        top_imp = st.session_state.importances.head(10).to_string(index=False)
                        msgs.append({"role":"user","content": f"Top importances:\n{top_imp}"})
                    except Exception:
                        pass
                if st.session_state.metrics is not None:
                    msgs.append({"role":"user","content": f"Métricas do modelo:\n{st.session_state.metrics}"})

                # chamada para a API
                try:
                    client = get_client()
                    resp = client.chat.completions.create(
                        model=model_api,
                        messages=msgs,
                        temperature=0.2,
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Erro na chamada à API: {e}"

                st.session_state.chat_history.append({"role":"assistant","content":reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
                # persistir msgs usadas (opcional)
                st.session_state.api_messages = msgs
                st.rerun()

        else:  # chat_mode == "LLM (API)"
            # modo LLM puro: envia prompt + (opcional) metrics/importances se existirem
            msgs = []
            msgs.append({"role":"system","content": sys_prompt})
            if st.session_state.metrics is not None:
                msgs.append({"role":"user","content": f"Métricas do modelo:\n{st.session_state.metrics}"})
            if st.session_state.importances is not None:
                try:
                    msgs.append({"role":"user","content": f"Top importances:\n{st.session_state.importances.head(10).to_string(index=False)}"})
                except Exception:
                    pass
            msgs.append({"role":"user","content": user_prompt})

            try:
                client = get_client()
                resp = client.chat.completions.create(
                    model=model_api,
                    messages=msgs,
                    temperature=0.4,
                )
                reply = resp.choices[0].message.content
            except Exception as e:
                reply = f"Erro na chamada à API: {e}"

            st.session_state.chat_history.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.api_messages = msgs
            st.rerun()
