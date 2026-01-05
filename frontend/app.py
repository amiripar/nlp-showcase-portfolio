import streamlit as st
import requests
from config_frontend import BACKEND_URL
from http_client import unwrap_api_response, extract_error

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Arabic": "ar",
    "Turkish": "tr",
    "Persian (Farsi)": "fa",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
}

st.set_page_config(page_title="NLP Showcase", layout="wide")

st.title("NLP Showcase Portfolio")
st.caption("Single-page demo of multiple NLP capabilities (backend: FastAPI).")

with st.sidebar:
    st.header("Backend Status")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=60)
        if r.status_code == 200:
            st.success("Backend is running")
            st.json(r.json())
        else:
            st.error(f"Backend error: {r.status_code}")
    except Exception as e:
        st.error("Backend is not reachable")
        st.write(str(e))

    st.divider()
    st.write("Backend URL:")
    st.code(BACKEND_URL)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    ["Text Classification", "Summarization", "Translation", "QA", "NER", "IR", "NLI", "NLU", "Chat", "Speech", "RAG"]
)

# ---------- Text Classification ----------
with tab1:
    st.subheader("Text Classification")

    st.markdown(
        """
    **What this does:**
    This task assigns a **label** to a piece of text (e.g., sentiment, topic, spam vs. not spam).
    In production, this is commonly used for ticket routing, moderation, intent detection, and document tagging.

    **How to use:**
    1. Enter text in the box.
    2. Click **Classify**.
    3. The API returns a predicted label (and optional confidence/metadata depending on implementation).
    """
    )

    st.divider()


    text = st.text_area(
        "Enter text",
        value="",
        height=120,
        key="tc_text",
        placeholder="Enter text to classify...",
    )

    if not text.strip():
        st.info("Enter some text to enable classification.")

    classify_clicked = st.button(
        "Classify",
        key="btn_classify",
        disabled=not text.strip(),
    )

    if classify_clicked:
        payload = {"text": text}
        try:
            resp = requests.post(f"{BACKEND_URL}/text-classification/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                st.success("Result")

                data = resp.json()

                label = data.get("label", "")
                confidence = float(data.get("confidence", 0.0))
                top_k = data.get("top_k", [])

                st.markdown("### Prediction")

                st.metric("Predicted label", label, delta=None)
                st.metric("Confidence", f"{confidence:.2%}", delta=None)

                st.progress(min(max(confidence, 0.0), 1.0))

                if isinstance(top_k, list) and len(top_k) > 0:
                    st.markdown("### Top-k scores")

                    labels = [item.get("label", "") for item in top_k]
                    scores = [float(item.get("score", 0.0)) for item in top_k]

                    chart_data = {"label": labels, "score": scores}
                    st.bar_chart(chart_data, x="label", y="score")

                with st.expander("Show raw JSON (debug)"):
                    st.json(data)

            else:
                st.error(f"Request failed: {resp.status_code}")
                st.write(resp.text)
        except Exception as e:
            st.error("Failed to call backend")
            st.write(str(e))

# ---------- Summarization ----------
with tab2:
    st.subheader("Summarization")
    st.markdown(
        """
**What this does:**
Summarization compresses a longer text into a shorter version while keeping the key meaning.
It is commonly used for reports, meeting notes, customer tickets, and document previews.

**How to use:**
1. Paste text to summarize.
2. Choose the max summary length.
3. Click **Summarize** to get a concise summary.
"""
    )

    st.divider()

    text = st.text_area(
        "Text to summarize",
        value="",
        height=200,
        key="sum_text",
        placeholder="Paste text here to summarize...",
    )

    max_chars = st.slider("Max summary chars", min_value=30, max_value=3000, value=120, key="sum_max_chars")

    if not text.strip():
        st.info("Enter some text to enable summarization.")

    summarize_clicked = st.button(
        "Summarize",
        key="btn_summarize",
        disabled=not text.strip(),
    )

    if summarize_clicked:
        payload = {"text": text, "max_chars": max_chars}
        try:
            resp = requests.post(f"{BACKEND_URL}/summarization/summarize", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()

                summary = data.get("summary", "")
                compression_ratio = data.get("compression_ratio", None)

                st.success("Summary")
                st.write(summary)

                if compression_ratio is not None:
                    try:
                        cr = float(compression_ratio)
                        st.markdown("### Compression ratio")
                        st.progress(min(max(cr, 0.0), 1.0))
                        st.write(f"{cr:.2%}")
                    except Exception:
                        pass

                with st.expander("Show raw JSON (debug)"):
                    st.json(data)

            else:
                st.error(f"Request failed: {resp.status_code}")
                st.write(resp.text)

        except Exception as e:
            st.error("Failed to call backend")
            st.write(str(e))

# ---------- Translation ----------
with tab3:
    st.title("Machine Translation")

    st.write(
        "Machine Translation converts text from one language to another. "
        "In this demo, you choose source and target languages, then translate the input text."
    )

    st.markdown("**What you will do here:**")
    st.markdown(
        "- Enter text\n"
        "- Choose source language and target language\n"
        "- Click **Translate** to get the result"
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        source_lang = st.text_input("Source language code", value="en", help="Example: en, fr, de, fa")
    with col_b:
        target_lang = st.text_input("Target language code", value="fa", help="Example: en, fr, de, fa")

    text = st.text_area(
        "Text to translate",
        height=180,
        placeholder='Example: Translate "Hello world" from en to fr',
        key="translation_text",
    )

    if "translation_last_request_id" not in st.session_state:
        st.session_state.translation_last_request_id = None

    st.divider()

    if st.button("Translate", type="primary", key="btn_translate"):
        if not text.strip():
            st.error("Please enter some text.")
        elif not source_lang.strip() or not target_lang.strip():
            st.error("Please enter both source and target language codes.")
        else:
            try:
                payload = {
                    "text": text,
                    "source_lang": source_lang.strip(),
                    "target_lang": target_lang.strip(),
                }

                resp = requests.post(
                    f"{BACKEND_URL}/translation/translate",
                    json=payload,
                    timeout=60,
                )

                ok, data, meta = unwrap_api_response(resp)
                st.session_state.translation_last_request_id = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Backend request failed"))
                    st.json(err)
                    if st.session_state.translation_last_request_id:
                        st.caption(f"request_id: {st.session_state.translation_last_request_id}")
                else:
                    translated_text = (data or {}).get("translated_text", "")
                    note = (data or {}).get("note", "")

                    st.subheader("Translated Text")
                    st.write(translated_text)

                    if note:
                        st.caption(note)

                    if st.session_state.translation_last_request_id:
                        st.caption(f"request_id: {st.session_state.translation_last_request_id}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    if st.session_state.translation_last_request_id:
        st.caption(f"last request_id: {st.session_state.translation_last_request_id}")

# ---------- QA (Conversational) ----------
with tab4:

    st.title("Question Answering (QA) — Grounded")

    st.write(
        "In this tab, you paste a *context* (a reference text) and ask questions about it. "
        "The system must answer using only the context. "
        "If the answer is not in the context, it will say: "
        "\"I don't know based on the provided context.\""
    )

    st.markdown("**How to use:**")
    st.markdown(
        "- Step 1: Paste your context text below.\n"
        "- Step 2: Ask questions in the chat box.\n"
        "- Step 3: The answer will be grounded to the context.\n"
        "- You can ask many questions using the same context."
    )

    st.divider()

    qa_context = st.text_area(
        "Context (paste your reference text here)",
        height=220,
        placeholder="Example: Toronto is in Canada. It is a big city.",
        key="qa_context_text",
    )

    if "qa_messages_v2" not in st.session_state:
        st.session_state.qa_messages_v2 = []  

    if "qa_last_request_id" not in st.session_state:
        st.session_state.qa_last_request_id = None

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Clear QA chat", key="qa_clear_chat_btn"):
            st.session_state.qa_messages_v2 = []
            st.session_state.qa_last_request_id = None
    with col2:
        st.caption("This QA is grounded: answers must come only from the Context above.")

    st.divider()

    for m in st.session_state.qa_messages_v2:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    qa_question = st.chat_input("Ask a question about the context...")

    if qa_question:
        if not qa_context.strip():
            st.error("Please paste some context first.")
        else:
            st.session_state.qa_messages_v2.append({"role": "user", "content": qa_question})
            with st.chat_message("user"):
                st.write(qa_question)

            try:
                payload = {"context": qa_context, "question": qa_question}
                resp = requests.post(f"{BACKEND_URL}/qa/answer", json=payload, timeout=60)

                ok, data, meta = unwrap_api_response(resp)
                st.session_state.qa_last_request_id = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Backend request failed"))
                    st.json(err)
                    if st.session_state.qa_last_request_id:
                        st.caption(f"request_id: {st.session_state.qa_last_request_id}")

                    qa_answer = err.get("message", "Backend request failed")
                else:
                    evidence = (data or {}).get("evidence", "").strip()
                    if evidence:
                        st.markdown("**Evidence (from context):**")
                        st.info(evidence)

                    note = (data or {}).get("note", "").strip()
                    if note:
                        st.caption(note)

                    if st.session_state.qa_last_request_id:
                        st.caption(f"request_id: {st.session_state.qa_last_request_id}")

                    qa_answer = (data or {}).get("answer", "")

                st.session_state.qa_messages_v2.append({"role": "assistant", "content": qa_answer})
                with st.chat_message("assistant"):
                    st.write(qa_answer)

            except requests.exceptions.RequestException as e:
                qa_err = f"Request failed: {e}"
                st.session_state.qa_messages_v2.append({"role": "assistant", "content": qa_err})
                with st.chat_message("assistant"):
                    st.write(qa_err)

# ---------- NER ----------
with tab5:
    st.title("Named Entity Recognition (NER)")

    st.write(
        "NER finds important names inside text and labels them. "
        "For example: people, organizations, locations, dates, and money."
    )

    st.markdown("**What you will get:**")
    st.markdown(
        "- Entity text (the exact words)\n"
        "- Entity type (PERSON, ORG, LOCATION, DATE, TIME, MONEY)\n"
        "- Start and End position (where the entity is inside the text)\n"
        "- Optional confidence score (if available)"
    )

    st.markdown("**How to use:**")
    st.markdown(
        "- Step 1: Paste your text.\n"
        "- Step 2: Choose one or more entity types.\n"
        "- Step 3: Click **Extract Entities**.\n"
        "- Step 4: You will see a list of entities found in the text."
    )

    st.divider()

    ner_text = st.text_area(
        "Text",
        height=220,
        placeholder="Example: John Smith lives in Toronto, Canada and works at OpenAI Inc. on 2025-12-16.",
        key="ner_text_area",
    )

    entity_options = ["PERSON", "ORG", "LOCATION", "DATE", "TIME", "MONEY"]
    ner_types = st.multiselect(
        "Entity types to extract (choose one or more)",
        options=entity_options,
        default=["PERSON", "ORG", "LOCATION"],
        key="ner_types_multiselect",
    )

    if "ner_last_request_id" not in st.session_state:
        st.session_state.ner_last_request_id = None

    if st.button("Extract Entities", key="btn_ner"):
        if not ner_text.strip():
            st.error("Please enter text.")
        else:
            try:
                payload = {"text": ner_text, "labels": ner_types}

                resp = requests.post(
                    f"{BACKEND_URL}/ner/extract",
                    json=payload,
                    timeout=30,
                )

                ok, data, meta = unwrap_api_response(resp)
                st.session_state.ner_last_request_id = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Backend request failed"))
                    st.json(err)
                    if st.session_state.ner_last_request_id:
                        st.caption(f"request_id: {st.session_state.ner_last_request_id}")
                else:
                    entities = (data or {}).get("entities", [])
                    note = (data or {}).get("note", "")

                    if note:
                        st.caption(note)

                    if st.session_state.ner_last_request_id:
                        st.caption(f"request_id: {st.session_state.ner_last_request_id}")

                    if not entities:
                        st.warning("No entities found.")
                    else:
                        st.markdown("### Entities")
                        for i, e in enumerate(entities, start=1):
                            ent_text = e.get("text", "")
                            label = e.get("label", "")
                            start = e.get("start", "")
                            end = e.get("end", "")
                            score = e.get("score", None)

                            line = f"**{i}. {ent_text}** | `{label}` | [{start}, {end}]"
                            if score is not None:
                                try:
                                    line += f" | score `{float(score):.4f}`"
                                except Exception:
                                    line += f" | score `{score}`"

                            st.write(line)

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

# ---------- IR ----------
with tab6:
    st.title("Information Retrieval (IR) — Search in Your Documents")

    st.write(
        "IR means searching and ranking the most relevant parts of your documents for a user query. "
        "This tab does not generate answers. It only finds the best matching text chunks."
    )

    st.markdown("**What you will do here:**")
    st.markdown(
        "- Upload TXT and/or PDF files\n"
        "- Build an index (split documents into chunks)\n"
        "- Search using a query\n"
        "- Get a ranked list of relevant chunks with scores"
    )

    st.divider()

    st.subheader("1) Upload files and build index")

    uploaded_files = st.file_uploader(
        "Upload one or more TXT/PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        chunk_size = st.number_input(
            "Chunk size (chars)", min_value=200, max_value=3000, value=900, step=50
        )
    with col_b:
        overlap = st.number_input(
            "Overlap (chars)", min_value=0, max_value=1000, value=150, step=10
        )
    with col_c:
        st.write("")

    if "ir_index_built" not in st.session_state:
        st.session_state.ir_index_built = False
    if "ir_last_index_info" not in st.session_state:
        st.session_state.ir_last_index_info = None
    if "ir_last_request_id" not in st.session_state:
        st.session_state.ir_last_request_id = None

    if st.button("Build Index", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one TXT or PDF file.")
        else:
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
                ]

                resp = requests.post(
                    f"{BACKEND_URL}/ir/index",
                    files=files_payload,
                    params={"chunk_size": int(chunk_size), "overlap": int(overlap)},
                    timeout=180,
                )

                ok, data, meta = unwrap_api_response(resp)
                st.session_state.ir_last_request_id = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.session_state.ir_index_built = False
                    st.error(err.get("message", "Backend request failed"))
                    st.json(err)
                    if st.session_state.ir_last_request_id:
                        st.caption(f"request_id: {st.session_state.ir_last_request_id}")
                else:
                    st.session_state.ir_index_built = True
                    st.session_state.ir_last_index_info = data

                    st.success(
                        f"Index built. Chunks: {data.get('num_chunks')} | "
                        f"Vocab: {data.get('vocab_size')} | "
                        f"Semantic ready: {data.get('semantic_ready')}"
                    )

                    note = (data or {}).get("note", "")
                    if note:
                        st.caption(note)

                    if st.session_state.ir_last_request_id:
                        st.caption(f"request_id: {st.session_state.ir_last_request_id}")

            except requests.exceptions.RequestException as e:
                st.session_state.ir_index_built = False
                st.error(f"Request failed: {e}")

    if st.session_state.ir_last_index_info:
        st.caption(
            f"Last index: chunks={st.session_state.ir_last_index_info.get('num_chunks')}, "
            f"vocab={st.session_state.ir_last_index_info.get('vocab_size')}, "
            f"semantic_ready={st.session_state.ir_last_index_info.get('semantic_ready')}"
        )
        if st.session_state.ir_last_request_id:
            st.caption(f"last request_id: {st.session_state.ir_last_request_id}")

    st.divider()
    st.subheader("2) Search")

    query = st.text_input("Query", placeholder="Example: LangChain agents")
    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5)

    mode = st.selectbox("Search mode", ["auto", "keyword", "semantic"], index=0)

    if st.button("Search"):
        if not st.session_state.ir_index_built:
            st.error("Index is not built yet. Upload files and click Build Index first.")
        elif not query.strip():
            st.error("Please enter a query.")
        else:
            try:
                payload = {"query": query, "top_k": int(top_k), "mode": mode}
                resp = requests.post(f"{BACKEND_URL}/ir/search", json=payload, timeout=90)

                ok, data, meta = unwrap_api_response(resp)
                st.session_state.ir_last_request_id = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Backend request failed"))
                    st.json(err)
                    if st.session_state.ir_last_request_id:
                        st.caption(f"request_id: {st.session_state.ir_last_request_id}")
                else:
                    results = (data or {}).get("results", [])
                    note = (data or {}).get("note", "")

                    if note:
                        st.caption(note)

                    if st.session_state.ir_last_request_id:
                        st.caption(f"request_id: {st.session_state.ir_last_request_id}")

                    if not results:
                        st.warning("No results found.")
                    else:
                        st.markdown("### Results")
                        for i, r in enumerate(results, start=1):
                            doc_name = r.get("doc_name", "")
                            chunk_id = r.get("chunk_id", "")
                            score = float(r.get("score", 0.0) or 0.0)
                            text = r.get("text", "")

                            page_num = r.get("page_num", None)

                            cite = f"{doc_name}"
                            if page_num is not None:
                                cite += f" (page {page_num})"
                            else:
                                cite += " (TXT)"

                            st.markdown(
                                f"**{i}. {cite}**  |  chunk `{chunk_id}`  |  score `{score:.4f}`"
                            )
                            st.write(text)
                            st.divider()

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

# ---------- NLI ----------
with tab7:
    st.title("Natural Language Inference (NLI)")

    st.write(
        "NLI compares two texts:\n"
        "- **Premise**: a statement we assume is true\n"
        "- **Hypothesis**: a statement we want to check\n\n"
        "The result is one of these labels:\n"
        "- **entailment**: the premise guarantees the hypothesis\n"
        "- **contradiction**: the premise makes the hypothesis false\n"
        "- **neutral**: not enough information either way"
    )

    st.markdown("**How to use:**")
    st.markdown(
        "- Step 1: Write a premise.\n"
        "- Step 2: Write a hypothesis.\n"
        "- Step 3: Click **Predict NLI**.\n"
        "- Step 4: See label + confidence + short reason."
    )

    st.divider()

    premise = st.text_area(
        "Premise",
        height=140,
        value="John is a teacher.",
        placeholder="Example: John is a teacher.",
        key="nli_premise",
    )

    hypothesis = st.text_area(
        "Hypothesis",
        height=140,
        value="John is a teacher.",
        placeholder="Example: John is a teacher.",
        key="nli_hypothesis",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        run = st.button("Predict NLI", key="nli_predict_btn")

    if run:
        if not premise.strip() or not hypothesis.strip():
            st.error("Please enter both Premise and Hypothesis.")
        else:
            payload = {"premise": premise, "hypothesis": hypothesis}

            try:
                with st.spinner("Running NLI..."):
                    resp = requests.post(f"{BACKEND_URL}/nli/predict", json=payload, timeout=30)

                if resp.status_code != 200:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                else:
                    data = resp.json()

                    label = data.get("label", "")
                    confidence = float(data.get("confidence", 0.0) or 0.0)
                    rationale = data.get("rationale", "")
                    note = data.get("note", "")

                    st.success("Result")
                    st.metric("Label", label, delta=None)
                    st.metric("Confidence", f"{confidence:.2%}", delta=None)

                    if rationale:
                        st.write("**Rationale:**")
                        st.write(rationale)

                    if note:
                        st.caption(note)

                    with st.expander("Raw JSON"):
                        st.json(data)

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    
# ---------- NLU ----------
with tab8:
    st.subheader("Natural Language Understanding (NLU) — Intent + Entities + Slots")

    st.write(
        "This module parses a user message into:\n"
        "- **Intent** (what the user wants)\n"
        "- **Entities** (important spans like DATE/LOCATION)\n"
        "- **Slots** (structured arguments)\n"
        "- **Clarification** (when required info is missing)"
    )

    text = st.text_area(
        "Enter user text",
        value="Book a flight to Dallas tomorrow morning",
        height=120,
        key="nlu_text",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        run = st.button("Parse NLU", key="btn_nlu")

    if run:
        payload = {"text": text}

        try:
            with st.spinner("Calling NLU..."):
                resp = requests.post(f"{BACKEND_URL}/nlu/parse", json=payload, timeout=30)

            if resp.status_code != 200:
                st.error(f"Request failed: {resp.status_code}")
                st.write(resp.text)
            else:
                data = resp.json()

                intent = (data.get("intent") or {})
                intent_name = intent.get("name", "")
                intent_conf = intent.get("confidence", 0.0)

                st.success("NLU Result")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Intent", intent_name)
                with c2:
                    try:
                        st.metric("Intent Confidence", f"{float(intent_conf):.2%}")
                    except Exception:
                        st.metric("Intent Confidence", str(intent_conf))

                st.divider()

                entities = data.get("entities") or []
                st.subheader("Entities")

                if len(entities) == 0:
                    st.info("No entities found.")
                else:
                    rows = []
                    for e in entities:
                        rows.append(
                            {
                                "type": e.get("type", ""),
                                "text": e.get("text", ""),
                                "value": e.get("value", ""),
                                "start": e.get("start", -1),
                                "end": e.get("end", -1),
                            }
                        )
                    st.table(rows)

                st.divider()

                st.subheader("Slots")
                st.json(data.get("slots") or {})

                st.divider()

                needs = bool(data.get("needs_clarification", False))
                cq = (data.get("clarifying_question", "") or "").strip()

                if needs:
                    st.warning("Needs clarification")
                    if cq:
                        st.write(f"**Question:** {cq}")
                    else:
                        st.write("**Question:** Could you clarify your request?")
                else:
                    st.info("No clarification needed.")

                note = data.get("note", "")
                if note:
                    st.caption(note)

                with st.expander("Raw JSON"):
                    st.json(data)

        except Exception as e:
            st.error("Failed to call backend")
            st.write(str(e))

# ---------- Chat ----------
with tab9:
    st.subheader("Dialogue / Conversational AI (Session Memory)")

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = "demo1"

    session_id = st.session_state.chat_session_id

    show_debug = False
    with st.expander("Advanced", expanded=False):
        session_id_input = st.text_input(
            "Session ID (dev)",
            value=session_id,
            key="chat_sid",
            help="Same Session ID = same in-memory conversation on the backend.",
        ).strip()
        if session_id_input:
            session_id = session_id_input
            st.session_state.chat_session_id = session_id_input

        show_debug = st.checkbox("Show debug JSON", value=False, key="chat_show_debug")

    left, right = st.columns([3, 1])
    with right:
        if st.button("Clear chat", key="btn_chat_clear"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat/clear",
                    json={"session_id": session_id},
                    timeout=10,
                )
                if resp.status_code == 200:
                    st.session_state["chat_last_note"] = ""
                    st.success("Chat cleared.")
                    st.rerun()
                else:
                    st.error(f"Request failed: {resp.status_code}")
                    st.write(resp.text)
            except Exception as e:
                st.error("Failed to call backend")
                st.write(str(e))

    st.divider()

    history = []
    try:
        resp = requests.get(f"{BACKEND_URL}/chat/history/{session_id}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            history = data.get("history", []) or []

            if show_debug:
                with st.expander("History (raw JSON)"):
                    st.json(data)
        else:
            st.error(f"History request failed: {resp.status_code}")
            st.write(resp.text)
    except Exception as e:
        st.error("Failed to load history")
        st.write(str(e))

    for m in history:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()

        if role not in ("user", "assistant"):
            role = "assistant"

        with st.chat_message(role):
            st.write(content)

    user_text = st.chat_input("Type your message and press Enter...")

    if user_text:
        payload = {"session_id": session_id, "message": user_text}
        try:
            resp = requests.post(f"{BACKEND_URL}/chat/respond", json=payload, timeout=30)
            if resp.status_code == 200:
                out = resp.json()
                st.session_state["chat_last_note"] = out.get("note", "")

                if show_debug:
                    with st.expander("Last response (raw JSON)"):
                        st.json(out)

                st.rerun()
            else:
                st.error(f"Send failed: {resp.status_code}")
                st.write(resp.text)
        except Exception as e:
            st.error("Failed to call backend")
            st.write(str(e))

    last_note = (st.session_state.get("chat_last_note") or "").strip()
    if last_note:
        st.caption(last_note)

# ---------- Speech ----------
with tab10:
    st.subheader("Speech (ASR + TTS)")

    st.markdown("### Text-to-Speech (TTS)")
    tts_text = st.text_area(
        "Enter text to synthesize",
        value="Hello, this is a speech test.",
        height=100,
        key="tts_text",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        do_tts = st.button("Generate Speech", key="btn_tts")

    if do_tts:
        payload = {"text": tts_text}
        try:
            resp = requests.post(f"{BACKEND_URL}/speech/tts", json=payload, timeout=60)
            if resp.status_code == 200:
                out = resp.json()

                audio_b64 = out.get("audio_base64", "")
                mime_type = out.get("mime_type", "audio/mpeg")
                note = out.get("note", "")

                if not audio_b64:
                    st.error("No audio returned (audio_base64 is empty).")
                else:
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)

                    st.success("Audio generated.")
                    st.audio(audio_bytes, format=mime_type)

                if note:
                    st.caption(note)
            else:
                st.error(f"TTS request failed: {resp.status_code}")
                st.write(resp.text)
        except Exception as e:
            st.error("Failed to call backend for TTS")
            st.write(str(e))

    st.divider()

    st.markdown("### Speech-to-Text (ASR)")
    audio_file = st.file_uploader(
        "Upload an audio file (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "ogg", "webm"],
        key="asr_file",
    )

    if st.button("Transcribe Audio", key="btn_asr"):
        if audio_file is None:
            st.warning("Please upload an audio file first.")
        else:
            try:
                files = {"audio": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                resp = requests.post(f"{BACKEND_URL}/speech/asr", files=files, timeout=120)

                if resp.status_code == 200:
                    out = resp.json()
                    st.success("Transcription completed.")
                    st.text_area("Transcript", value=out.get("transcript", ""), height=180)

                    note = out.get("note", "")
                    if note:
                        st.caption(note)
                else:
                    st.error(f"ASR request failed: {resp.status_code}")
                    st.write(resp.text)
            except Exception as e:
                st.error("Failed to call backend for ASR")
                st.write(str(e))


# --- RAG TAB (tab11) ---
with tab11:
    st.subheader("RAG — Ask Questions About Your PDF (or Text)")

    if "rag_selected_corpus_id" not in st.session_state:
        st.session_state["rag_selected_corpus_id"] = None
    if "rag_index_ready" not in st.session_state:
        st.session_state["rag_index_ready"] = False
    if "rag_last_index_source" not in st.session_state:
        st.session_state["rag_last_index_source"] = ""
    if "rag_last_request_id" not in st.session_state:
        st.session_state["rag_last_request_id"] = None
    if "rag_corpora_cache" not in st.session_state:
        st.session_state["rag_corpora_cache"] = []

    st.markdown(
        """
**How it works (professional):**
1. Create a **Corpus** by uploading a PDF or pasting text.
2. The corpus is **saved on disk** under `storage/rag/corpora/`.
3. Select an existing corpus anytime and ask questions against it.

Notes:
- If your PDF is scanned images, text extraction may fail (OCR would be required later).
"""
    )

    st.divider()

    st.markdown("### Saved corpora (old indexes)")

    colr1, colr2 = st.columns([1, 3])
    with colr1:
        refresh = st.button("Refresh list", key="btn_rag_refresh")
    with colr2:
        st.caption("Select a previously indexed corpus and ask questions without re-indexing.")

    if refresh or not st.session_state["rag_corpora_cache"]:
        try:
            resp = requests.get(f"{BACKEND_URL}/rag/corpora", timeout=30)
            ok, data, meta = unwrap_api_response(resp)
            st.session_state["rag_last_request_id"] = (meta or {}).get("request_id")

            if not ok:
                err = extract_error(resp)
                st.error(err.get("message", "Failed to load corpora list"))
                st.json(err)
            else:
                st.session_state["rag_corpora_cache"] = data or []
        except Exception as e:
            st.error("Failed to load corpora list.")
            st.write(str(e))

    corpora = st.session_state["rag_corpora_cache"] or []
    if not corpora:
        st.info("No saved corpora yet. Create one below by indexing a PDF or text.")
    else:
        options = []
        id_map = {}
        for c in corpora:
            cid = c.get("corpus_id")
            name = c.get("name", cid)
            src = c.get("source_name", "")
            created = c.get("created_at", "")
            chunks = c.get("chunk_count", 0)
            label = f"{name} — {src} — chunks={chunks} — {created}"
            options.append(label)
            id_map[label] = cid

        default_idx = 0
        if st.session_state["rag_selected_corpus_id"]:
            for i, lbl in enumerate(options):
                if id_map.get(lbl) == st.session_state["rag_selected_corpus_id"]:
                    default_idx = i
                    break

        selected_label = st.selectbox(
            "Choose a corpus",
            options=options,
            index=default_idx,
            key="rag_corpus_selectbox",
        )
        st.session_state["rag_selected_corpus_id"] = id_map.get(selected_label)

        selected_meta = next((c for c in corpora if c.get("corpus_id") == st.session_state["rag_selected_corpus_id"]), None)
        if selected_meta:
            st.markdown("**Selected corpus details**")
            st.write(
                {
                    "corpus_id": selected_meta.get("corpus_id"),
                    "name": selected_meta.get("name"),
                    "source": f"{selected_meta.get('source_type')} / {selected_meta.get('source_name')}",
                    "created_at": selected_meta.get("created_at"),
                    "chunk_size": selected_meta.get("chunk_size"),
                    "overlap": selected_meta.get("overlap"),
                    "chunk_count": selected_meta.get("chunk_count"),
                }
            )

        st.session_state["rag_index_ready"] = True  

    st.divider()

    st.markdown("### Create a new corpus")

    pdf_file = st.file_uploader("Upload a PDF (optional)", type=["pdf"], key="rag_pdf")
    docs_text = st.text_area(
        "Or paste text to index (optional)",
        value="",
        height=220,
        key="rag_docs",
        placeholder="If a PDF is uploaded, PDF will be used first.",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        build_clicked = st.button("Build / Save Corpus", key="btn_rag_build")
    with col2:
        clear_clicked = st.button("Clear Active (pointer)", key="btn_rag_clear")
    with col3:
        if st.session_state.get("rag_last_request_id"):
            st.caption(f"last request_id: {st.session_state['rag_last_request_id']}")

    if build_clicked:
        try:
            if pdf_file is not None:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{BACKEND_URL}/rag/index_pdf", files=files, timeout=180)

                ok, data, meta = unwrap_api_response(resp)
                st.session_state["rag_last_request_id"] = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Indexing failed"))
                    st.json(err)
                else:
                    stats = (data or {}).get("stats", {})
                    corpus_id = stats.get("corpus_id")
                    st.session_state["rag_selected_corpus_id"] = corpus_id
                    st.session_state["rag_last_index_source"] = f"PDF ({pdf_file.name})"
                    st.session_state["rag_index_ready"] = True

                    st.success(f"Corpus saved. corpus_id={corpus_id}")
                    note = (data or {}).get("note", "")
                    if note:
                        st.caption(note)

                    st.session_state["rag_corpora_cache"] = []

            else:
                text = (docs_text or "").strip()
                if not text:
                    st.warning("Nothing to index. Upload a PDF or paste text first.")
                else:
                    payload = {"documents": [text]}
                    resp = requests.post(f"{BACKEND_URL}/rag/index", json=payload, timeout=180)

                    ok, data, meta = unwrap_api_response(resp)
                    st.session_state["rag_last_request_id"] = (meta or {}).get("request_id")

                    if not ok:
                        err = extract_error(resp)
                        st.error(err.get("message", "Indexing failed"))
                        st.json(err)
                    else:
                        stats = (data or {}).get("stats", {})
                        corpus_id = stats.get("corpus_id")
                        st.session_state["rag_selected_corpus_id"] = corpus_id
                        st.session_state["rag_last_index_source"] = "Text box"
                        st.session_state["rag_index_ready"] = True

                        st.success(f"Corpus saved. corpus_id={corpus_id}")
                        note = (data or {}).get("note", "")
                        if note:
                            st.caption(note)

                        st.session_state["rag_corpora_cache"] = []

        except Exception as e:
            st.error("Failed to call backend during indexing.")
            st.write(str(e))

    if clear_clicked:
        try:
            resp = requests.post(f"{BACKEND_URL}/rag/clear", timeout=30)
            ok, data, meta = unwrap_api_response(resp)
            st.session_state["rag_last_request_id"] = (meta or {}).get("request_id")

            if not ok:
                err = extract_error(resp)
                st.error(err.get("message", "Clear failed"))
                st.json(err)
            else:
                st.session_state["rag_selected_corpus_id"] = None
                st.session_state["rag_index_ready"] = False
                st.success("Active corpus pointer cleared (saved corpora remain).")
                note = (data or {}).get("note", "")
                if note:
                    st.caption(note)

        except Exception as e:
            st.error("Failed to call backend for clear.")
            st.write(str(e))

    st.divider()

    st.markdown("### Ask a question")
    question = st.text_input(
        "Question",
        value="",
        key="rag_question",
        placeholder="Type your question here...",
    )

    can_ask = bool(st.session_state.get("rag_selected_corpus_id"))
    if not can_ask:
        st.info("Select a saved corpus (above) or create one first.")

    ask_clicked = st.button("Ask", key="btn_rag_ask", disabled=not can_ask)

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                payload = {
                    "question": question,
                    "corpus_id": st.session_state.get("rag_selected_corpus_id"),
                }
                resp = requests.post(f"{BACKEND_URL}/rag/ask", json=payload, timeout=180)

                ok, data, meta = unwrap_api_response(resp)
                st.session_state["rag_last_request_id"] = (meta or {}).get("request_id")

                if not ok:
                    err = extract_error(resp)
                    st.error(err.get("message", "Request failed"))
                    st.json(err)
                else:
                    st.markdown("### Answer")
                    st.write((data or {}).get("answer", ""))

                    if st.session_state.get("rag_last_request_id"):
                        st.caption(f"request_id: {st.session_state['rag_last_request_id']}")

                    with st.expander("Show retrieved chunks (debug / transparency)"):
                        for i, item in enumerate((data or {}).get("retrieved", []), start=1):
                            score = float(item.get("score", 0.0) or 0.0)
                            st.markdown(f"**Chunk {i} — score: {score:.4f}**")
                            st.code(item.get("text", ""), language="text")

                    note = (data or {}).get("note", "")
                    if note:
                        st.caption(note)

            except Exception as e:
                st.error("Failed to call backend for RAG ask.")
                st.write(str(e))
