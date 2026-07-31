import json
import os
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Assistente Psicologico & CDS",
    page_icon="🧠",
    layout="wide"
)

st.title(" Assistente di Auto-Riflessione & Supporto Clinico")

# Sidebar - API Key Input
st.sidebar.header("⚙️ Configurazione")
groq_api_key = st.sidebar.text_input(
    "Groq API Key", 
    type="password", 
    help="Inserisci la tua API Key di Groq (ottienila gratuitamente da console.groq.com)"
)

if not groq_api_key:
    st.info("👈 Per iniziare, inserisci la tua **Groq API Key** nella barra laterale.")
    st.stop()

# Initialize OpenAI Client pointing to Groq's API
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)
MODEL_NAME = "llama-3.1-8b-instant"

# Database CSV for feedback learning loop
CSV_FILE = "storico_feedback_psicologi.csv"

if not os.path.exists(CSV_FILE):
    df_init = pd.DataFrame(columns=["caso_descritto", "ipotesi_ia", "corretto", "correzione_psicologo"])
    df_init.to_csv(CSV_FILE, index=False)

def recupera_storico_correzioni():
    """Retrieve negative feedback corrections to use as Few-Shot context."""
    try:
        df = pd.read_csv(CSV_FILE)
        correzioni = df[df["corretto"] == "No"]
        if correzioni.empty:
            return ""
        testo_storico = "\n--- ESEMPI DI VALUTAZIONI CORRETTE DA PSICOLOGI UMANI ---\n"
        for _, row in correzioni.tail(3).iterrows():
            testo_storico += f"\nCaso: {row['caso_descritto']}\n"
            testo_storico += f"Errore precedente IA: {row['ipotesi_ia']}\n"
            testo_storico += f"Correzione dello Psicologo: {row['correzione_psicologo']}\n"
        testo_storico += "-----------------------------------------------------\n"
        return testo_storico
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# 2. ROLE SELECTION ROUTING
# -----------------------------------------------------------------------------
ruolo = st.sidebar.radio(
    "Seleziona la modalità d'uso:", 
    ["👤 Paziente / Utente", "👨‍⚕️ Psicologo / Clinico", "📊 Dashboard Feedback"]
)

# -----------------------------------------------------------------------------
# MODE A: PATIENT / JOURNALING
# -----------------------------------------------------------------------------
if ruolo == "👤 Paziente / Utente":
    st.subheader("💬 Modalità Journaling & Auto-Riflessione")
    st.warning(
        "⚠️ **Disclaimer Importante**: Questo strumento è destinato esclusivamente all'auto-riflessione "
        "e al journaling personale. Non è un sostituto della terapia e non fornisce diagnosi o consigli medici. "
        "In caso di emergenza o grave sofferenza, si prega di contattare un professionista o il numero unico d'emergenza 112."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Scrivi qui come ti senti oggi..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            system_prompt = (
                "Sei un assistente empatico per il journaling e la riflessione personale. "
                "Rispondi in italiano con calore ed empatia. NON dare mai consigli medici, "
                "non fare diagnosi e fai domande aperte per aiutare l'utente a riflettere sui propri pensieri. "
                "Ricorda all'utente che non sei un terapeuta."
            )
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
                temperature=0.7
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# -----------------------------------------------------------------------------
# MODE B: PSYCHOLOGIST / CLINICAL DECISION SUPPORT
# -----------------------------------------------------------------------------
elif ruolo == "👨‍⚕️ Psicologo / Clinico":
    st.subheader("🔬 Supporto alla Valutazione Clinica (CDS)")
    st.caption("Inserisci la descrizione anonimizzata del caso clinico per ottenere un'ipotesi diagnostica strutturata.")

    caso_clinico = st.text_area(
        "Descrizione del caso clinico:", 
        height=150, 
        placeholder="Esempio: Paziente di 32 anni riferisce ansia persistente da 7 mesi, insonnia iniziale e preoccupazione costante per il futuro lavorativo..."
    )

    if st.button("🔍 Analizza Caso Clinico", type="primary"):
        if not caso_clinico.strip():
            st.error("Inserisci prima la descrizione del caso.")
        else:
            with st.spinner("Analisi in corso e consultazione delle correzioni storiche..."):
                storico = recupera_storico_correzioni()
                system_prompt = f"""Sei un assistente per la valutazione clinica dedicato a professionisti della salute mentale.
Analizza il caso descritto, fai un'ipotesi diagnostica (DSM-5 / ICD-11) e indica le motivazioni.

{storico}

Devi rispondere ESCLUSIVAMENTE con un oggetto JSON valido con questa struttura:
{{
  "ipotesi_diagnostica": "Nome della diagnosi ipotizzata",
  "motivazioni": ["Motivazione 1", "Motivazione 2"],
  "diagnosi_differenziale": ["Alternativa 1", "Alternativa 2"]
}}
"""
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": caso_clinico}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )

                risultato = json.loads(response.choices[0].message.content)
                st.session_state["ultimo_caso"] = caso_clinico
                st.session_state["ultima_analisi"] = risultato

    if "ultima_analisi" in st.session_state:
        st.divider()
        res = st.session_state["ultima_analisi"]

        st.success(f"**Ipotesi Diagnostica Ipotizzata:** {res.get('ipotesi_diagnostica', 'N/D')}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Motivazioni Cliniche:**")
            for mot in res.get("motivazioni", []):
                st.write(f"- {mot}")
        with col2:
            st.write("**Diagnosi Differenziale:**")
            for diff in res.get("diagnosi_differenziale", []):
                st.write(f"- {diff}")

        st.divider()
        st.subheader("📝 Validazione Clinica & Feedback Loop")
        corretto = st.radio("La valutazione dell'IA è corretta?", ["Sì", "No", "Parzialmente"], key="radio_val")
        correzione = st.text_area(
            "Note di correzione o integrazioni dello psicologo:", 
            placeholder="Spiega perché l'ipotesi è errata o incompleta. Queste note verranno usate per istruire l'IA nei casi futuri..."
        )

        if st.button("💾 Salva Feedback per l'Apprendimento"):
            nuovo_dato = {
                "caso_descritto": st.session_state["ultimo_caso"],
                "ipotesi_ia": res.get("ipotesi_diagnostica", ""),
                "corretto": corretto,
                "correzione_psicologo": correzione
            }
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([nuovo_dato])], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            st.success("✅ Feedback salvato con successo! L'IA ne terrà conto nelle prossime valutazioni.")

# -----------------------------------------------------------------------------
# MODE C: FEEDBACK DASHBOARD
# -----------------------------------------------------------------------------
elif ruolo == "📊 Dashboard Feedback":
    st.subheader("📊 Dataset dei Feedback e Apprendimento Continuo")
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        st.dataframe(df, use_container_width=True)
        st.metric("Totale Valutazioni Salvate", len(df))
    else:
        st.info("Nessun dato di feedback presente al momento.")


