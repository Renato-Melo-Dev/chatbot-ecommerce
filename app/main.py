# app/main.py
import sys, os, pickle
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pipeline import run_training_pipeline, run_prediction_pipeline
from core.chatbot.rules import answer_from_metrics

# Diretório e arquivo do modelo
MODEL_DIR = "models_store"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "model_sales.pkl")

# Configuração da página
st.set_page_config(page_title="📊 E-commerce ML - Chat", layout="wide")
st.title("📊 Dashboard E-commerce com Treino / Predição / Chat")

# Estado da sessão
for key in ["model_trained", "predictions_made", "prediction_df", "metrics", "importances", "chat_messages"]:
    if key not in st.session_state:
        st.session_state[key] = False if key.endswith("trained") or key.endswith("made") else None

st.session_state.chat_messages = st.session_state.chat_messages or [
    {"role": "assistant", "content": "Oi! Eu sou o bot do E-commerce. Envie seus dados para começar."}
]

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Upload do CSV
uploaded_file = st.file_uploader("📁 Faça upload do CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sidebar para ações
    with st.sidebar:
        st.header("Ações")
        train_btn = st.button("🚀 Treinar modelo")
        predict_btn = st.button("📦 Carregar modelo e prever")
        reset_btn = st.button("♻️ Resetar modelo e dados")

    # Resetar sessão
    if reset_btn:
        for key in ["model_trained", "predictions_made", "prediction_df", "metrics", "importances"]:
            st.session_state[key] = False if key.endswith("trained") or key.endswith("made") else None
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Oi! Eu sou o bot do E-commerce. Envie seus dados para começar."}
        ]
        st.success("✅ Sessão reiniciada! Faça upload do CSV e treine um novo modelo.")

    # Treinar modelo
    if train_btn:
        with st.spinner("Treinando modelo..."):
            metrics, importances = run_training_pipeline(df, test_size=0.2, model_path=MODEL_PATH)
            st.session_state.model_trained = True
            st.session_state.predictions_made = False
            st.session_state.metrics = metrics
            st.session_state.importances = importances
        st.success("✅ Modelo treinado e salvo!")

    # Fazer predições
    if predict_btn:
        if uploaded_file is None:
            st.warning("📁 Faça upload de um CSV antes de prever.")
        elif not os.path.exists(MODEL_PATH):
            st.error("Modelo não encontrado. Treine um modelo primeiro.")
        else:
            with st.spinner("Realizando predições..."):
                df_pred = run_prediction_pipeline(df, MODEL_PATH)
                st.session_state.prediction_df = df_pred
                st.session_state.predictions_made = True
            st.success("✅ Predições realizadas!")


# === Layout das abas ===
tab_train, tab_predict, tab_chat = st.tabs(["📊 Resultados do Treino", "🚀 Predições", "💬 Chat"])

with tab_train:
    st.markdown("<h2 style='color:white;'>📊 Resultados do Treino</h2>", unsafe_allow_html=True)
    if not st.session_state.model_trained or st.session_state.metrics is None:
        st.info("Treine um modelo para ver resultados.")
    else:
        # Métricas
        col1, col2, col3 = st.columns(3)
        metrics = st.session_state.metrics
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.4f}" if metrics.get('rmse') else "N/A")
        with col2:
            st.metric("R²", f"{metrics.get('r2', 'N/A'):.4f}" if metrics.get('r2') else "N/A")
        with col3:
            st.metric("MAE", f"{metrics.get('mae', 'N/A'):.4f}" if metrics.get('mae') else "N/A")

        st.markdown("---")
        # Importâncias
        with st.expander("🔎 Importâncias das Features"):
            if st.session_state.importances is not None:
                df_imp = st.session_state.importances.copy()
                total_abs = df_imp["Coeficiente"].abs().sum()
                df_imp["Importância (%)"] = (df_imp["Coeficiente"].abs() / total_abs * 100).round(1)
                st.dataframe(df_imp[[df_imp.columns[0], "Importância (%)"]].head(10), use_container_width=True)
            else:
                st.info("Importâncias das features não estão disponíveis.")


with tab_predict:
    st.markdown("<h2 style='color:white;'>🚀 Predições</h2>", unsafe_allow_html=True)
    if not st.session_state.predictions_made:
        st.info("Faça predições usando o modelo treinado.")
    else:
        st.dataframe(st.session_state.prediction_df)
        csv_data = convert_df_to_csv(st.session_state.prediction_df)
        st.download_button(
            label="⬇️ Baixar Previsões em CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv",
        )

with tab_chat:
    st.markdown("<h2 style='color:white;'>💬 Converse com o Modelo</h2>", unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Pergunte sobre métricas, features ou previsões..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        response = answer_from_metrics(
            question=prompt,
            metrics_df_or_dict=st.session_state.metrics,
            importances_df=st.session_state.importances,
            model_pipe=None
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()