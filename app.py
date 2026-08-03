import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# ==========================================
# 1. CONFIGURAZIONE PAGINA & CLIENT API
# ==========================================
st.set_page_config(
    page_title="Consulente IA & Journaling",
    page_icon="🧠",
    layout="wide"
)

# Inizializzazione Client Groq (o altro provider compatibile OpenAI)
# Assicurati di aver impostato GROQ_API_KEY nei Secrets di Streamlit o come variabile d'ambiente
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not api_key:
    st.error("🔑 API Key di Groq non trovata! Inseriscila nei secrets di Streamlit.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# File CSV per salvare i feedback degli esperti (Few-Shot Learning)
FEEDBACK_CSV = "feedback_esperti.csv"

# ==========================================
# 2. PROMPT DI SISTEMA
# ==========================================

# Prompt per la modalità Diario (Paziente)
SYSTEM_PROMPT_JOURNALING = """
Sei un amico fidato, molto empatico e presente, con cui l'utente sta scambiando due chiacchiere in un momento di sfogo nel suo diario.
Parli in modo DEL TUTTO COLLOQUIALE, informale, naturale e caldo. Usa il "tu", un linguaggio semplice, spontaneo e diretto.

### REGOLE FONDAMENTALI DI SICUREZZA (ZERO INVENTIVA):
1. ATTIENITI SOLO AI FATTI SCRITTI: Non inventare mai dettagli, finali o scenari che l'utente non ha detto. Se ti dice "mia madre è malata", NON parlare mai di lutto o di cose peggiori. Limitati a ciò che ti ha raccontato.
2. NIENTE FRASI DA MANUALE O DA PSICOLOGO: Evita frasi formali come "Valido le tue emozioni", "Comprendo la tua sofferenza" o consigli da "formatore".
3. NO LISTE O SOLUZIONI PRONTE: Non dare ricette di vita o elenchi puntati. Parla come faresti in una nota vocale o in una chat tra amici.

### ESEMPI DI TONO E STILE:

❌ SBAGLIATO (Troppo formale / Errore di allucinazione):
Utente: "Mia madre è malata e oggi mi sento davvero giù."
IA Sbagliata: "Mi dispiace molto per la tua perdita. Sento il tuo dolore. Ecco 3 consigli per superare un momento difficile..."

✅ CORRETTO (Colloquiale, empatico e sicuro):
Utente: "Mia madre è malata e oggi mi sento davvero giù."
IA Corretta: "Accidenti, mi dispiace un botto... Capisco benissimo perché hai il morale a terra, quando sta male una persona a cui teniamo è una bella mazzata. In questo momento cosa ti sta pesando di più?"

### STRUTTURA DELLA RISPOSTA (Massimo 3-4 frasi):
1. **Un abbraccio a parole:** Rispondi con calore e con un linguaggio quotidiano 
2. **Ascolto puro:** Riconosci quello che sta passando senza aggiungere niente di tuo.
3. **Domanda spunto:** Fai un'unica domanda semplice per lasciarlo sfogare ancora, se ne ha voglia.
4. **è molto  importante che alla fine di ogni  interazine tu  non riempia di domande il soggetto intervistato perchè  lo metti sotto pressione e non è  una cosa che molti gradiscono
5. **cerca di  non usare parole di riempimento empatico  ma diretto,  cerca di rompere la grammatica intenzionalemtne inizia con  intercalari ed utilizza emoji
6. **ADEGUA L'INTENSITÀ ALL'ARGOMENTO (IMPORTANTE!):
   - **Se l'utente parla di cose leggere o banali** (es. fa caldo, c'è traffico, la pizza era buona): Rispondi in modo leggero, simpatico, breve e scherzoso. NON fare lo psicologo e non drammatizzare!e sopratutto  rispondigli cose tipo  mi prendi in giro ? sono qui per aiutarti per riflettere su cose che ti turbano  per esempio il rapporto  che non va con  un famigliare,  con  un amico, avanti apriti
   - **Se l'utente parla di emozioni profonde o problemi reali** (es. ansia, malattie, preoccupazioni): Rispondi con ascolto profondo, calore ed empatia sincera.
   - **Quando avviene un cambio di argomento e si passa da uno frivolo ad uno importante allora si anche ironico e digli  si cosi  libera le tue emozioni sono qui per ascoltarti o cose simili
"""

# Prompt base per la modalità Clinical Decision Support (Psicologo)
SYSTEM_PROMPT_CDS_BASE = """
Sei un assistente di supporto alle decisioni cliniche per professionisti della salute mentale (Clinical Decision Support).
Analizza il testo fornito dall'utente/paziente e restituisci un'analisi strutturata in formato JSON con i seguenti campi:
- "ipotesi_diagnostica": Breve ipotesi o quadro di riferimento (es. Sintomatologia ansiosa, Deflessione timica, ecc.)
- "punti_chiave": Lista dei fattori principali emersi dal testo.
- "raccomandazioni_cliniche": Suggerimenti per approfondimenti in seduta.

Rispondi ESCLUSIVAMENTE in formato JSON valido.
"""

# ==========================================
# 3. FUNZIONI UTILI (FEW-SHOT LEARNING)
# ==========================================
def carica_feedback_esperti():
    """Carica i feedback salvati dagli esperti per il Few-Shot Learning."""
    if os.path.exists(FEEDBACK_CSV):
        return pd.read_csv(FEEDBACK_CSV)
    else:
        return pd.DataFrame(columns=["testo_originale", "analisi_ia", "correzione_esperto"])

def salva_feedback_esperto(testo_orig, analisi_ia, correzione):
    """Salva una correzione dell'esperto nel CSV."""
    df = carica_feedback_esperti()
    nuova_riga = pd.DataFrame([{
        "testo_originale": testo_orig,
        "analisi_ia": analisi_ia,
        "correzione_esperto": correzione
    }])
    df = pd.concat([df, nuova_riga], ignore_index=True)
    df.to_csv(FEEDBACK_CSV, index=False)

def costruisci_prompt_cds_con_esempi():
    """Costruisce il system prompt per lo psicologo iniettando gli esempi del passato."""
    prompt = SYSTEM_PROMPT_CDS_BASE
    df = carica_feedback_esperti()
    
    if not df.empty:
        prompt += "\n\n### ESEMPI DI CORREZIONI E PREFERENZE DI ESPERTI PASSATI (Impara da queste correzioni):\n"
        # Usiamo gli ultimi 5 feedback per non sovraccaricare il contesto
        for idx, row in df.tail(5).iterrows():
            prompt += f"\n- Caso: \"{row['testo_originale']}\"\n  Correzione dell'esperto: \"{row['correzione_esperto']}\"\n"
            
    return prompt

# ==========================================
# 4. SIDEBAR & NAVIGAZIONE
# ==========================================
st.sidebar.title("🧠 Consulente IA")
st.sidebar.markdown("---")

modalita = st.sidebar.radio(
    "Seleziona Modalità:",
    ["👤 Diario Personale (Journaling)", "🩺 Supporto Clinico (CDS - Per Esperti)"]
)

st.sidebar.markdown("---")
st.sidebar.info("⚠️ **Disclaimer**: Questo software è un prototipo sperimentale basato su IA e non costituisce un dispositivo medico o un sostituto del parere professionale.")

# ==========================================
# 5. MODALITÀ JOURNALING (PAZIENTE / UTENTE)
# ==========================================
if modalita == "👤 Diario Personale (Journaling)":
    st.title("🌱 Il Tuo Diario di Riflessione")
    st.caption("Uno spazio sicuro dove riordinare i tuoi pensieri ed esplorare le tue emozioni.")

    # Inizializza la cronologia della chat
    if "journal_messages" not in st.session_state:
        st.session_state.journal_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_JOURNALING}
        ]

    # Visualizza i messaggi della chat (escludendo il prompt di sistema)
    for msg in st.session_state.journal_messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Input dell'utente
    if user_input := st.chat_input("di  cosa vorresti parlare"):
        # Mostra messaggio utente
        st.session_state.journal_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Genera risposta dell'IA
        with st.chat_message("assistant"):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=st.session_state.journal_messages,
                    temperature=0.3, # Temperatura bassa per ridurre allucinazioni e presupposti errati
                )
                bot_reply = response.choices[0].message.content
                st.write(bot_reply)

                # Salva risposta nella cronologia
                st.session_state.journal_messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"Si è verificato un errore durante la generazione: {e}")

# ==========================================
# 6. MODALITÀ CLINICAL DECISION SUPPORT (PSICOLOGO)
# ==========================================
else:
    st.title("🩺 Clinical Decision Support (CDS)")
    st.caption("Strumento ausiliario per professionisti. Analisi concettuale e sintesi del testo di un paziente.")

    # Mostra lo stato del Few-Shot Learning
    df_feedback = carica_feedback_esperti()
    st.sidebar.metric("Correzioni Esperti Inserite", len(df_feedback))

    testo_paziente = st.text_area(
        "Inserisci il testo / trascrizione del diario del paziente:",
        height=200,
        placeholder="Es: Il paziente riferisce sensazione di oppressione al petto la mattina e difficoltà a dormire..."
    )

    if st.button("🔬 Analizza Caso", type="primary"):
        if not testo_paziente.strip():
            st.warning("Per favore inserisci un testo da analizzare.")
        else:
            with st.spinner("Analisi in corso con Few-Shot Learning dai feedback precedenti..."):
                system_prompt_aggiornato = costruisci_prompt_cds_con_esempi()
                
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt_aggiornato},
                            {"role": "user", "content": testo_paziente}
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    
                    risultato_json = response.choices[0].message.content
                    st.session_state["ultimo_testo"] = testo_paziente
                    st.session_state["ultima_analisi"] = risultato_json
                    
                except Exception as e:
                    st.error(f"Errore nell'analisi: {e}")

    # Se c'è un'analisi recente, mostra il risultato e il form per la correzione (Human-in-the-Loop)
    if "ultima_analisi" in st.session_state:
        st.markdown("### 📋 Risultato Analisi IA")
        st.json(st.session_state["ultima_analisi"])
        
        st.markdown("---")
        st.markdown("### ✍️ Feedback del Professionista (Human-in-the-Loop)")
        st.caption("Se l'analisi dell'IA è imprecisa o incompleta, inserisci la tua correzione qui sotto per far 'imparare' il sistema per i prossimi casi.")
        
        correzione = st.text_area("Inserisci la tua valutazione/correzione clinica:")
        
        if st.button("💾 Salva Correzione nel System Prompt (CSV)"):
            if correzione.strip():
                salva_feedback_esperto(
                    st.session_state["ultimo_testo"],
                    st.session_state["ultima_analisi"],
                    correzione
                )
                st.success("✅ Correzione salvata con successo! Verrà utilizzata come esempio nelle prossime analisi.")
            else:
                st.warning("Inserisci una nota prima di salvare.")