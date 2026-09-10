import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from google import genai

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(
    title="Edge-Cloud AI Orchestrator Research Assistant",
    version="1.0.0",
)

MODEL_NAME = "gemini-3.1-flash-lite"
@app.get("/robot.png", include_in_schema=False)
async def robot_image():
    return FileResponse("robot.png")


SYSTEM_PROMPT = """
You are the Research Assistant for the MSc Artificial Intelligence project
"AI-Driven Infrastructure Orchestration for Edge-Cloud Environments".

Your purpose is to explain the implemented academic project accurately,
clearly and concisely.

PROJECT CONTEXT

Architecture:
- React dashboard/frontend.
- Python FastAPI orchestration backend.
- Edge node: Ubuntu Linux virtual machine running through Oracle VirtualBox.
- AWS IoT Greengrass is used to deploy and manage the Edge inference component on the Ubuntu Edge node. It does not imply that every AI workload executes at the Edge.
- Amazon S3 was used to store the Greengrass component artefact used for Edge deployment.
- AWS Systems Manager Session Manager was used to access and administer the Amazon EC2 cloud node without exposing direct remote access credentials.
- Cloud node: Amazon EC2. Amazon EC2 hosts the Cloud node and provides the environment for cloud-side AI workload execution. Do not state that the central orchestration backend is hosted on EC2; the main orchestration backend runs in the local development/control environment.
- Execution routes: Edge, Cloud and Hybrid.
- Final orchestration policies: Rule-Based and Q-Learning.
- AI workloads: YOLOv8 and Florence-2. YOLOv8 represents the object-detection workload. Florence-2 represents the semantic vision-language interpretation workload.
- In the Cooperative Hybrid execution path, YOLOv8 executes at the Edge and Florence-2 executes in the Cloud. Do not state or imply that Florence-2 executes on the Edge in this Hybrid pipeline.
- Gemini is used only as the Research Assistant interface and is not a routing policy.

Rule-Based policy:
- If connectivity == 0, force Edge.
- If cost_budget_usd <= 0.001, force Edge.
- Otherwise calculate a cloud suitability score.

Cloud-score contributions:
- network_latency_ms < 80: +0.25
- cpu_available < 40: +0.30
- memory_available < 40: +0.20
- batch_size >= 8: +0.15
- 20 <= model_size_mb < 300: +0.15
- model_size_mb >= 300: +0.65
- priority >= 4: +0.20

Cooperative Hybrid conditions:
- model_size_mb >= 300
- connectivity == 1
- cpu_available >= 50
- memory_available >= 40
- network_latency_ms <= 150
- cost_budget_usd > 0.01

When Cooperative Hybrid is selected:
- Edge performs YOLO object detection.
- Cloud performs Florence-2 semantic interpretation.

Ordered Rule-Based route selection:
1. Cooperative Hybrid conditions -> Hybrid.
2. cloud_score between 0.35 and 0.60, with sufficient cost budget -> Hybrid.
3. cloud_score above 0.5 -> Cloud.
4. Otherwise -> Edge.

Because the Hybrid branch is evaluated before the Cloud branch,
a cloud_score from 0.35 through 0.60 with sufficient budget selects Hybrid.
In practice, the later Cloud branch therefore applies when the score is above 0.60.

Q-Learning policy:
- Q-Learning is the adaptive routing policy.
- It learns from real execution outcomes.
- Its action space is Edge, Cloud and Hybrid.
- Its state representation uses factors including network latency,
  CPU availability, cost budget, priority and connectivity.
- Q-values are algorithm-derived values, not physical measurements.
- The route with the preferred learned value for the current state
  can be selected according to the learned policy.
- Exploration may also occur during learning.

Experimental interpretation:
- Infrastructure telemetry and execution measurements in the original
  experimental system are real measurements.
- This public demonstration does NOT have live access to the user's
  local Edge VM, SQLite decision database or current telemetry.
- Never describe values in this public demonstration as current live telemetry
  unless such values are explicitly supplied in the conversation.
- If asked for a current live measurement, state that live telemetry is
  available in the original experimental environment but not exposed
  through this public demonstration.

Academic accuracy rules:
- Describe the Rule-Based policy as deterministic threshold-based routing logic. Do not call it a Decision Tree, decision-tree model, classifier, or learned machine-learning policy.
- Do not invent, reconstruct or infer exact research questions, findings, conclusions, limitations, future-work items, experimental results, numerical values or comparative performance claims unless they are explicitly included in this public project context or supplied by the user.
- If exact research questions, findings, conclusions or future work are not present in this public context, state that the exact information is not available in the public demonstration context rather than fabricating it.
- Do not claim that Q-Learning outperforms, is superior to, or represents an improvement over Rule-Based unless explicit measured evidence is provided.
- Do not invent power-consumption metrics, energy evaluation, multi-node Edge clusters, scalability experiments or other future-work items not explicitly provided.
- Distinguish implemented architecture and system design from measured experimental findings.
STRICT RULES
1. Do not invent experimental numbers, measurements or results.
2. Do not claim that Decision Tree or Random Forest are final orchestration policies.
3. Clearly distinguish measured infrastructure values, workload inputs and
   algorithm-derived values.
4. Do not expose or speculate about API keys, credentials, private IP addresses,
   database contents or other sensitive configuration.
5. Answer questions about this academic project only.
6. Default to English.
7. If the user writes in another language, you may respond in that language.
8. Use concise MSc-level academic language.
"""


class AssistantRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-assistant"}


@app.post("/api/ask")
async def ask_assistant(req: AssistantRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
{SYSTEM_PROMPT}

USER QUESTION:
{question}

Answer using only the project context above.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as e:
        print(f"Gemini error: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="The AI service is temporarily unavailable.",
        )

    return {
        "session_id": req.session_id or str(uuid.uuid4()),
        "question": question,
        "answer": response.text or "No answer was returned.",
        "model": MODEL_NAME,
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Edge-Cloud AI Research Assistant</title>
<style>
    * { box-sizing: border-box; }
    body {
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: #0b1020;
        color: #eef3ff;
    }
    .page {
        max-width: 850px;
        margin: 0 auto;
        padding: 28px 18px 60px;
    }
    .header {
        text-align: center;
        margin-bottom: 26px;
    }
    .header h1 {
        font-size: 28px;
        margin: 0 0 8px;
    }
    .header p {
        color: #aeb9d4;
        line-height: 1.5;
        margin: 0;
    }
    .card {
        background: #141b31;
        border: 1px solid #293451;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 12px 35px rgba(0,0,0,.25);
    }
    #messages {
        min-height: 0;
        max-height: 58vh;
        overflow-y: auto;
        margin-bottom: 16px;
    }
    .message {
        padding: 12px 14px;
        border-radius: 12px;
        margin: 10px 0;
        line-height: 1.55;
        white-space: pre-wrap;
    }
    .user {
        background: #223158;
        margin-left: 12%;
    }
    .assistant {
        background: #19243d;
        border: 1px solid #2d3d62;
        margin-right: 7%;
    }
    .label {
        display: block;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: .08em;
        color: #9fb5ed;
        margin-bottom: 5px;
    }
    .input-row {
        display: flex;
        gap: 10px;
    }
    textarea {
        flex: 1;
        resize: none;
        min-height: 58px;
        border-radius: 10px;
        border: 1px solid #3a496d;
        background: #0e1528;
        color: white;
        padding: 12px;
        font-size: 15px;
        outline: none;
    }
    button {
        width: 110px;
        border: 0;
        border-radius: 10px;
        background: #e6edf9;
        color: #111827;
        font-weight: bold;
        cursor: pointer;
    }
    button:disabled {
        opacity: .55;
        cursor: wait;
    }
    .note {
        font-size: 12px;
        color: #8896b6;
        text-align: center;
        margin-top: 14px;
    }
    .brandline {
        color: #4fd1ff;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .creator {
        margin-top: 10px;
        color: #c7d2ea;
        font-size: 13px;
    }
    .creator strong {
        color: #ffffff;
    }
    .topics-intro {
        text-align: center;
        color: #c7d2ea;
        margin: 26px 0 15px;
        font-size: 14px;
    }
    .topics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 22px;
    }
    .topic-card {
        min-height: 108px;
        padding: 13px 10px;
        border-radius: 12px;
        border: 1px solid #2e4167;
        background: linear-gradient(180deg, #17223c 0%, #111a30 100%);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
    }
    .topic-icon {
        font-size: 21px;
        margin-bottom: 7px;
    }
    .topic-title {
        color: #67d8ff;
        font-size: 12px;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 5px;
    }
    .topic-sub {
        color: #9daac5;
        font-size: 10px;
        line-height: 1.3;
    }
    @media (max-width: 600px) {
        .header h1 { font-size: 23px; }
        .input-row { flex-direction: column; }
        button { width: 100%; height: 46px; }
        .user, .assistant { margin-left: 0; margin-right: 0; }
    }
@media (max-width: 600px) {
    .page { max-width: 100%; padding: 22px 14px 42px; }
    .header { margin-bottom: 20px; padding: 0 8px; }
    .brandline { font-size: 11px; letter-spacing: .10em; margin-bottom: 12px; }
    .header h1 { font-size: 27px; line-height: 1.15; margin-bottom: 12px; }
    .header p { font-size: 15px; line-height: 1.45; margin: 0 auto; max-width: 92%; }
    .creator { margin-top: 14px; font-size: 13px; line-height: 1.45; }
    .topics-intro { margin: 24px 0 16px; font-size: 14px; }
    .card { padding: 12px; border-radius: 14px; }
    #messages { min-height: 0; max-height: none; overflow-y: visible; margin-bottom: 14px; }
    .message { width: 100%; box-sizing: border-box; padding: 14px; line-height: 1.55; }
    .user, .assistant { margin-left: 0; margin-right: 0; }
    .input-row { flex-direction: column; gap: 10px; }
    textarea { width: 100%; box-sizing: border-box; }
    button { width: 100%; height: 46px; }
}
@media (max-width: 600px) {
    .topics-grid {
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 5px;
        width: 100%;
    }
    .topic-card {
        min-width: 0;
        min-height: 112px;
        padding: 8px 3px;
        border-radius: 10px;
    }
    .topic-icon {
        font-size: 17px;
        margin-bottom: 6px;
    }
    .topic-title {
        font-size: 10px;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }
    .topic-sub {
        font-size: 8px;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
}
    .assistant-panel {
        padding: 20px;
    }
    .assistant-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 2px 2px 16px;
        margin-bottom: 16px;
        border-bottom: 1px solid #293451;
    }
    .assistant-avatar {
        width: 58px;
        height: 58px;
        object-fit: contain;
        flex: 0 0 auto;
    }
    .assistant-identity .label {
        margin-bottom: 4px;
        font-size: 11px;
        letter-spacing: .12em;
    }
    .assistant-subtitle {
        color: #aeb9d4;
        font-size: 13px;
        letter-spacing: .02em;
    }
    .assistant-panel #messages {
        min-height: 0;
        max-height: none;
        overflow: visible;
        margin: 14px 0 0;
    }
    .assistant-panel .note {
        margin-top: 14px;
    }
</style>
</head>
<body>
<div class="page">
    <div class="header">
        <div class="brandline">University of Bedfordshire · UK</div>
        <h1>Edge-Cloud AI Research Assistant</h1>
        <p>AI-Driven Infrastructure Orchestration for Edge-Cloud Environments</p>
        <div class="creator">
            <strong>Sophia Souza Marcal</strong> · MSc Artificial Intelligence · 2026<br>
            School of Computer Science and Technology
        </div>
    </div>

    <div class="topics-intro">Explore the topics below or type your question.</div>

    <div class="topics-grid">
        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">◆</div>
            <div class="topic-title">System Architecture</div>
            <div class="topic-sub">End-to-End Design</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">☁</div>
            <div class="topic-title">Edge, Cloud & Hybrid</div>
            <div class="topic-sub">Execution Routes</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">◉</div>
            <div class="topic-title">Routing Policies</div>
            <div class="topic-sub">Rule-Based · Q-Learning</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">◎</div>
            <div class="topic-title">AI Models</div>
            <div class="topic-sub">YOLOv8 · Florence-2</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">▣</div>
            <div class="topic-title">Infrastructure & Machines</div>
            <div class="topic-sub">Edge VM · AWS EC2</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">⚙</div>
            <div class="topic-title">AWS Services</div>
            <div class="topic-sub">Greengrass · S3 · Session Manager</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">&lt;/&gt;</div>
            <div class="topic-title">Implementation</div>
            <div class="topic-sub">React · FastAPI · Python</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">▥</div>
            <div class="topic-title">Evaluation & Results</div>
            <div class="topic-sub">Latency · CPU · Memory · RTT</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">◇</div>
            <div class="topic-title">Project Methodology</div>
            <div class="topic-sub">Design · Development · Testing</div>
        </div>

        <div class="topic-card" onclick="selectTopic(this)">
            <div class="topic-icon">▤</div>
            <div class="topic-title">Research & Conclusions</div>
            <div class="topic-sub">Questions · Findings · Future Work</div>
        </div>
    </div>

    <div class="card assistant-panel">
        <div class="assistant-brand">
            <img src="/robot.png" alt="AI Research Assistant Robot" class="assistant-avatar">
            <div class="assistant-identity">
                <span class="label">AI ASSISTANT</span>
                <div class="assistant-subtitle">Project Knowledge Interface</div>
            </div>
        </div>

        <div class="input-row">
            <textarea id="question" placeholder="Ask a question about the project..."></textarea>
            <button id="send">Ask</button>
        </div>

        <div id="messages"></div>

        <div class="note">
            Public academic demonstration · Live infrastructure telemetry is not exposed.
        </div>
    </div>
</div>

<script>
const messages = document.getElementById("messages");
const question = document.getElementById("question");
const send = document.getElementById("send");
let sessionId = null;

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderMarkdown(text) {
    let html = escapeHtml(text);

    html = html
        .replaceAll(String.fromCharCode(92) + "*", "*").replaceAll(String.fromCharCode(92) + "_", "_").replaceAll(String.fromCharCode(92) + "`", "`")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/^\s*\*\s+(.*)$/gm, "• $1")
        .replace(/^\s*-\s+(.*)$/gm, "• $1")
        .replace(/\\n/g, "<br>");

    return html;
}

function addMessage(label, text, cssClass) {
    const div = document.createElement("div");
    div.className = "message " + cssClass;

    const span = document.createElement("span");
    span.className = "label";
    span.textContent = label;

    div.appendChild(span);

    const content = document.createElement("div");

    if (cssClass === "assistant") {
        content.innerHTML = renderMarkdown(text);
    } else {
        content.textContent = text;
    }

    div.appendChild(content);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function selectTopic(card) {
    const title = card.querySelector(".topic-title")?.textContent.trim() || "";
    const prompts = {
        "System Architecture": "Explain the system architecture of this project.",
        "Edge, Cloud & Hybrid": "Explain how Edge, Cloud and Hybrid execution work in this project.",
        "Routing Policies": "Explain the Rule-Based and Q-Learning routing policies used in this project.",
        "AI Models": "Explain the role of YOLOv8 and Florence-2 in this project.",
        "Infrastructure & Machines": "Explain the Edge VM and AWS EC2 infrastructure used in this project.",
        "AWS Services": "Explain the AWS services used in this project, including Greengrass, S3 and Session Manager.",
        "Implementation": "Explain how the system was implemented using React, FastAPI and Python.",
        "Evaluation & Results": "Explain how the project was evaluated and which performance metrics were used.",
        "Project Methodology": "Explain the project methodology, including design, development and testing.",
        "Research & Conclusions": "Summarise the research questions, findings, conclusions and future work of this project."
    };
    if (prompts[title]) {
        question.value = prompts[title];
        question.focus();
    }
}
async function ask() {
    const text = question.value.trim();
    if (!text) return;

    addMessage("YOU", text, "user");
    question.value = "";
    send.disabled = true;
    send.textContent = "Thinking...";

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                question: text,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Request failed.");
        }

        sessionId = data.session_id;
        addMessage("AI ASSISTANT", data.answer, "assistant");
    } catch (error) {
        addMessage("AI ASSISTANT", "Unable to answer: " + error.message, "assistant");
    } finally {
        send.disabled = false;
        send.textContent = "Ask";
        question.focus();
    }
}

send.addEventListener("click", ask);

question.addEventListener("keydown", function(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask();
    }
});
</script>
</body>
</html>
"""





















