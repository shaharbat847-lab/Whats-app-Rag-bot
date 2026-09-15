import os
import time
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()

# --- Config (set these as environment variables on your host) ---
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]          # Meta permanent access token
WHATSAPP_PHONE_ID = os.environ["WHATSAPP_PHONE_ID"]    # from Meta developer dashboard
VERIFY_TOKEN = os.environ["VERIFY_TOKEN"]              # any string you choose, used in Meta setup
INDEX_PATH = os.environ.get("INDEX_PATH", "./my_rag_index")

# --- Load RAG chain once at startup ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

prompt_template = """You are a helpful assistant that answers questions using only the context below, which is personal information about the user. If the answer isn't in the context, say you don't know — don't make things up.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)


def get_rag_reply(text: str, max_retries: int = 3) -> str:
    """Calls the RAG chain, retrying with backoff if Gemini's quota is hit."""
    for attempt in range(max_retries):
        try:
            result = qa_chain.invoke({"query": text})
            return result["result"]
        except Exception as e:
            is_quota_error = "429" in str(e) or "quota" in str(e).lower()
            if is_quota_error and attempt < max_retries - 1:
                wait_seconds = 20 * (attempt + 1)  # 20s, then 40s
                time.sleep(wait_seconds)
                continue
            if is_quota_error:
                return "I've hit my reply limit for now — please try again in a few minutes."
            raise
    return "Something went wrong generating a reply. Please try again."


def send_whatsapp_message(to: str, body: str):
    url = f"https://graph.facebook.com/v20.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta calls this once when you set up the webhook in the dashboard."""
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_message(request: Request):
    """Meta calls this every time a WhatsApp message arrives."""
    data = await request.json()

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]["value"]
        messages = changes.get("messages")

        if messages:
            msg = messages[0]
            from_number = msg["from"]
            text = msg["text"]["body"]

            reply = get_rag_reply(text)
            send_whatsapp_message(from_number, reply)

    except (KeyError, IndexError):
        # Non-message events (delivery receipts, etc.) — ignore
        pass

    return {"status": "ok"}


@app.get("/")
async def health():
    return {"status": "running"}
