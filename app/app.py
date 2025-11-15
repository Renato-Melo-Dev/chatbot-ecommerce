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
from core.chatbot.insights import gerar_insights

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
        value=("Você é um analista de dados especializado em e-commerce. "
               "Responda de forma clara e objetiva. Use apenas os dados do contexto fornecido. "
               "Não inclua métricas de ML a menos que o usuário solicite especificamente."),
        height=140
    )
    st.markdown("---")
    st.header("Ações")
    uploaded_file = st.file_uploader("📁 Faça upload do CSV (dataset)", type=["csv"])
    train_btn = st.button("🚀 Treinar modelo (com ETL / pipeline)")
    predict_btn = st.button("📦 Carregar modelo e prever")
    reset_btn = st.button("♻️ Resetar modelo, banco e sessão")
    st.markdown("---")
    st.caption("Modo de Chat: RAG = usa o resumo da base (SPEC). Peça métricas explicitamente se quiser.")

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
            metrics, importances = run_training_pipeline(df, test_size=test_size, model_path=MODEL_PATH)
            st.session_state.model_trained = True
            st.session_state.metrics = metrics
            st.session_state.importances = importances
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

# === Funções de contexto (RAG helpers) ===
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


def build_context_from_df(df: pd.DataFrame, max_chars: int = 4000, include_sample: bool = True) -> str:
    parts = [f"Shape: {df.shape[0]} linhas x {df.shape[1]} colunas",
             "\n[Resumo numérico]\n" + numeric_summary(df),
             "\n[Resumo categórico]\n" + categorical_summary(df)]
    if include_sample:
        try:
            parts.append("\n[Exemplo - 5 primeiras linhas]\n" + df.head(5).to_string())
        except Exception:
            pass
    ctx = "\n\n".join(parts)
    return ctx[:max_chars] + ("\n... (contexto truncado)" if len(ctx) > max_chars else "")

# === Função de geração de resposta local (insights on-demand) ===
def get_local_insight(user_prompt: str, df: pd.DataFrame) -> str:
    """
    Tenta responder localmente (sem chamar API) para consultas comuns:
    - top produtos (por país opcional)
    - produto mais caro (por país opcional)
    - top clientes
    - gerar insights gerais via core.chatbot.insights.gerar_insights
    Retorna None se não identificar intenção específica.
    """
    q = (user_prompt or "").lower()

    # casual greetings
    if any(k in q for k in ["oi","olá","ola","tudo bem","bom dia","boa tarde","boa noite"]):
        return "Oi! 😊 Posso ajudar com análises da base (top produtos, vendas por país, clientes) ou com perguntas técnicas sobre o modelo."

    # detectar país (palavras simples em pt ou en)
    country = None
    for cand in df['Country'].unique():
        if cand.lower() in q:
            country = cand
            break

    # top produtos
    if "top" in q and ("produto" in q or "produtos" in q):
        # extrair número (top N)
        import re
        n = 5
        m = re.search(r"top\s*(\d+)", q)
        if m:
            try:
                n = int(m.group(1))
            except: pass
        sub = df[df['Country'] == country] if country else df
        top = sub.groupby(['StockCode','Description'])['Quantity'].sum().sort_values(ascending=False).head(n)
        lines = [f"{i+1}. {desc} (Código: {code}) - {qty} unidades" for i, ((code, desc), qty) in enumerate(top.items())]
        insight_text = gerar_insights(sub if country else df, top_n_products=n, country=country)
        return "🏆 Top produtos:\n" + "\n".join(lines) + "\n\n" + insight_text

    # produto mais caro
    if "mais caro" in q or ("preço" in q and "alto" in q):
        sub = df[df['Country'] == country] if country else df
        if 'UnitPrice' in sub.columns and not sub.empty:
            row = sub.loc[sub['UnitPrice'].idxmax()]
            return f"💰 Produto mais caro: {row['Description']} (Código: {row['StockCode']}), preço unitário: {row['UnitPrice']:.2f}, país: {row['Country']}"

    # top clientes
    if any(x in q for x in ["clientes que mais gastaram","top clientes","clientes mais"]) :
        sub = df[df['Country'] == country] if country else df
        if {'CustomerID','TotalPrice'}.issubset(sub.columns):
            topc = sub.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(5)
            return "🔝 Top clientes:\n" + topc.to_string()

    # fallback: not a local insight
    return None

# === Tab: Chat ===
with tab_chat:
    st.header("💬 Conversar (LLM + RAG + Local Insights)")

    # Exibir histórico
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 

    user_prompt = st.chat_input("Digite sua pergunta")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # carregar tabela spec (RAG context)
        try:
            df_context = load_data("spec_sales")
        except Exception:
            df_context = None

        if df_context is None or df_context.empty:
            reply = "A tabela `spec_sales` ainda não existe ou está vazia. Execute o pipeline de treino para gerar o contexto (treinar com o CSV)."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.rerun()

        # 1) tenta responder localmente (fast path)
        local = get_local_insight(user_prompt, df_context)
        if local:
            reply = local
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.rerun()

        # 2) construir contexto RAG e decidir quais extras enviar
        context_text = build_context_from_df(df_context, max_chars=max_ctx)
        if show_ctx:
            with st.expander("Ver contexto gerado"):
                st.text(context_text)

        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Contexto do e-commerce:\n{context_text}"},
            {"role": "user", "content": f"Pergunta do usuário: {user_prompt}"}
        ]

        # Somente anexar métricas/importances se a pergunta pedir explicitamente
        qlow = user_prompt.lower()
        if st.session_state.metrics is not None and any(k in qlow for k in ["métric","rmse","mae","r2","modelo","treino"]):
            msgs.append({"role": "user", "content": f"Métricas do modelo:\n{st.session_state.metrics}"})
        if st.session_state.importances is not None and any(k in qlow for k in ["feature","importan","variáve","variavel","coeficient"]):
            msgs.append({"role": "user", "content": f"Importances:\n{st.session_state.importances.head(10).to_string(index=False)}"})

        # incluir histórico curto (últimas 6 mensagens) para contexto de conversação
        for m in st.session_state.chat_history[-6:]:
            msgs.append({"role": m["role"], "content": m["content"]})

        # chamada ao modelo
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
        st.session_state.api_messages = msgs
        st.rerun()
