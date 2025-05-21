import sys
import re
import fitz  # PyMuPDF
import gradio as gr
import requests
import xml.etree.ElementTree as ET
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pymysql
from pymysql.err import OperationalError
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from gradio.utils import NamedString

# 1. Database connection (PyMySQL)
try:
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="ZX5568zx",
        database="papers_db",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    print("✅ Connected to MySQL database via PyMySQL!")
    cursor = db.cursor()
except OperationalError as e:
    print("❌ Database connection error (PyMySQL):", e)
    sys.exit(1)

# 2. Embeddings & FAISS
emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = None

# 3. LLM: Flan-T5
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Extract title and abstract from the PDF
def extract_title_abstract(path):
    doc = fitz.open(path)
    lines = doc[0].get_text().split("\n")
    title = lines[0] if lines else "Unknown"
    abstract = ""
    for i, l in enumerate(lines):
        if "abstract" in l.lower():
            abstract = " ".join(lines[i+1:i+6])
            break
    return title, abstract

# 5. Extract key points from the paper
def extract_paper_highlights(abstract):
    prompt = f"Please extract the main contributions, methods, conclusions, etc., from the following paper abstract:\n\n{abstract}"
    return llm(prompt)

# 6. Tool definitions
def upload_tool(path):
    global vectorstore
    title, abs_ = extract_title_abstract(path)
    highlights = extract_paper_highlights(abs_)
    cursor.execute(
        "INSERT INTO papers (title, abstract, source) VALUES (%s, %s, %s)",
        (title, abs_, "upload")
    )
    db.commit()
    if vectorstore is None:
        vectorstore = FAISS.from_texts([abs_], emb, metadatas=[{"title": title}])
    else:
        vectorstore.add_texts([abs_], metadatas=[{"title": title}])
    return f"✅ Uploaded: {title}\n\nPaper Highlights:\n{highlights}"

def search_tool(q):
    if vectorstore is None:
        return "⚠️ No data available. Please upload papers first."
    res = vectorstore.similarity_search(q, k=3)
    return "\n---\n".join(
        [f"【{r.metadata['title']}】\n{r.page_content}" for r in res]
    )

def compare_tool(_):
    cursor.execute("SELECT title, abstract FROM papers ORDER BY created_at DESC LIMIT 2")
    rows = cursor.fetchall()
    if len(rows) < 2:
        return "Not enough papers to compare."
    (t1, a1), (t2, a2) = rows
    prompt = (
        f"Compare the two papers:\n"
        f"Paper1: {t1}\nAbstract: {a1}\n\n"
        f"Paper2: {t2}\nAbstract: {a2}\n\n"
        "Provide goals, methods, contributions, strengths, weaknesses, similarities."
    )
    return llm(prompt)

def search_arxiv(query, max_results=3):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            papers.append((title, abstract))
        return papers
    except Exception as e:
        return f"❌ arXiv API request failed: {e}"

def web_search_tool(query):
    global vectorstore
    results = search_arxiv(query)
    if not results or isinstance(results, str):
        return results if isinstance(results, str) else "❌ 沒有找到相關論文"
    texts = []
    for title, abstract in results:
        cursor.execute(
            "INSERT INTO papers (title, abstract, source) VALUES (%s, %s, %s)",
            (title, abstract, "web_search")
        )
        texts.append(abstract)
    db.commit()
    if vectorstore is None:
        vectorstore = FAISS.from_texts(texts, emb, metadatas=[{"title": t} for t, _ in results])
    else:
        vectorstore.add_texts(texts, metadatas=[{"title": t} for t, _ in results])
    return "\n\n".join([f"【{t}】\n{a}" for t, a in results])

# 7. Initialize Agent
tools = [
    Tool("UploadPDFTool", upload_tool, "Upload and parse PDF"),
    Tool("InternalSearchTool", search_tool, "Semantic search"),
    Tool("ComparePapersTool", compare_tool, "Compare papers"),
    Tool("WebSearchTool", web_search_tool, "Search arXiv for recent papers")
]
agent = initialize_agent(tools, llm, AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 8. Smarter natural language command detection
def respond(input, history):
    print("DEBUG upload input:", input, type(input))
    ui = str(input).lower().strip()

    if isinstance(input, (dict, tuple, NamedString)) or any(k in ui for k in [".pdf", "/tmp/"]):
        if isinstance(input, dict):
            path = input.get("tmp_path") or input.get("name") or input.get("file")
        elif isinstance(input, tuple):
            file_obj = input[1]
            path = getattr(file_obj, "name", None)
        elif isinstance(input, NamedString):
            path = str(input)
        else:
            path = input
        out = upload_tool(path) if path else "⚠️ Cannot read uploaded file path."

    elif any(k in ui for k in ["compare", "比較"]):
        out = compare_tool(None)
    elif any(k in ui for k in ["recent", "arxiv", "最新"]):
        query = re.sub(r"(search|recent|arxiv|find|最新)", "", ui).strip()
        out = web_search_tool(query or "machine learning")
    elif any(k in ui for k in ["find", "search", "找"]):
        query = re.sub(r"(find|search|related papers|找)", "", ui).strip()
        out = search_tool(query or "transformer")
    else:
        out = "❓ 請使用自然語言指令，例如 'Upload PDF', 'Search contrastive learning', 'Compare papers' 或 'Search arXiv diffusion models'。"

    history.append((input, out))
    return history, ""

# 9. Start Gradio
try:
    print("🚀 Starting Gradio app...")
    with gr.Blocks() as demo:
        gr.Markdown("## 📚 LLM Research Assistant Agent")
        bot = gr.Chatbot(type="tuples")
        with gr.Row():
            txt = gr.Textbox(placeholder="For example: Find diffusion model papers")
            up = gr.File(file_types=[".pdf"])
            btn = gr.Button("Send")
        btn.click(respond, [txt, bot], [bot, txt])
        up.upload(respond, [up, bot], [bot, txt])
    demo.launch(server_name="0.0.0.0", server_port=7861)
except Exception as e:
    print("❌ Error occurred while starting the Gradio app:", e)