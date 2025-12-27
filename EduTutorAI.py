# EdututorAI - Gradio frontend only (no Pinecone)
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, re, time, json

# --- In-memory "databases" (runtime only) ---
students_db = {}    # key: email -> {"name":..., "email":..., "registered_at":...}
educators_db = {}   # key: email -> {"name":..., "email":..., "registered_at":...}
dashboard_db = {}   # key: "email::topic" -> {"student_name", "email", "topic", "highest_score", "attempts_count", "last_attempt_score", "created_at", "updated_at"}

# --- Gradio session states ---
user_role = gr.State()
user_name = gr.State()
user_email = gr.State()

# --- Embedding dimension placeholder (for compatibility with old code) ---
DIMENSION = 128
DUMMY_VECTOR = [1.0] + [0.0] * (DIMENSION - 1)


# -------------------------
# Simple helper functions
# -------------------------
def register_student(name: str, email: str):
    key = email.strip().lower()
    students_db[key] = {
                                   "name": name,
        "email": key,
        "registered_at": time.time()
    }
    return students_db[key]

def register_educator(name: str, email: str):
    key = email.strip().lower()
    educators_db[key] = {
        "name": name,
        "email": key,
        "registered_at": time.time()
    }
    return educators_db[key]

def record_quiz_result(email: str, student_name: str, topic: str, score: int):
    """
    Update dashboard_db for (email, topic). Maintain highest_score & attempts_count.
    """
    email_k = email.strip().lower()
    topic_k = (topic or "general").strip().lower()
    rec_key = f"{email_k}::${topic_k}"

    now = time.time()
    existing = dashboard_db.get(rec_key)
    if existing:
        existing_attempts = existing.get("attempts_count", 0) + 1
        new_best = max(existing.get("highest_score", 0), int(score))
        existing.update({
            "student_name": student_name,
            "email": email_k,
            "topic": topic,
            "last_attempt_score": int(score),
            "highest_score": new_best,
            "attempts_count": existing_attempts,
            "updated_at": now
        })
        dashboard_db[rec_key] = existing
    else:
        dashboard_db[rec_key] = {
            "student_name": student_name,
            "email": email_k,
            "topic": topic,
            "last_attempt_score": int(score),
            "highest_score": int(score),
            "attempts_count": 1,
            "created_at": now,
            "updated_at": now
        }
    return dashboard_db[rec_key]

def get_student_dashboard(email: str):
    email_k = (email or "").strip().lower()
    results = []
    for k, v in dashboard_db.items():
        if v.get("email") == email_k:
            results.append(v)
    # sort by updated_at desc
    results.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
    return results


# -------------------------
# Model & generation
# -------------------------
model_name = "ibm-granite/granite-3.2-2b-instruct"  # keep same as your original; change if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_new_tokens=400, do_sample=True, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def concept_explanation(concept):
    prompt = f"Explain the concept of {concept} in detail with examples:"
    return generate_response(prompt, max_new_tokens=800, do_sample=True, temperature=0.7)

def generate_interactive_quiz(topic):
    prompt = f"Write 5 multiple choice quiz questions about {topic}. For each question, give exactly 4 options (A, B, C, D). After all questions, add an ANSWERS section with the correct options."
    full_quiz = generate_response(prompt, max_new_tokens=400, do_sample=False)

    # Debug: print raw output to logs (helpful in Colab)
    print("RAW QUIZ OUTPUT:\n", full_quiz)

    parts = re.split(r'\bANSWERS?:', full_quiz, flags=re.IGNORECASE)
    if len(parts) < 2:
        q_defaults = [f"{i+1}. (Question missing)" for i in range(5)]
        radio_updates = []
        for _ in range(5):
            choices = ["A) (missing)", "B) (missing)", "C) (missing)", "D) (missing)"]
            radio_updates.append(gr.update(choices=choices, value=None))
        answers = ["A"] * 5
        return tuple(q_defaults) + tuple(radio_updates) + (answers,)

    quiz_part, answer_part = parts[0].strip(), parts[1].strip()
    raw_questions = re.split(r'(?=\d+\.\s)', quiz_part)
    raw_questions = [q.strip() for q in raw_questions if q.strip()][:5]

    ans_lines = [ln.strip() for ln in answer_part.splitlines() if ln.strip()]
    answers = []
    for ln in ans_lines:
        m = re.search(r'([A-D])', ln, flags=re.I)
        if m:
            answers.append(m.group(1).upper())
    answers = answers[:5]

    q_texts = []
    radio_updates = []
    for q in raw_questions:
        first_line = q.splitlines()[0] if q.splitlines() else q
        q_text = re.sub(r'^\d+\.\s*', '', first_line).strip()
        opt_matches = re.findall(r'([A-D])[\.\)]\s*(.*?)(?=(?:\s+[A-D][\.\)]\s)|$)', q, flags=re.S)
        opts_dict = {}
        for letter, text in opt_matches:
            opts_dict[letter.upper()] = text.strip()
        choices = []
        for letter in ['A', 'B', 'C', 'D']:
            text = opts_dict.get(letter, "(missing option)")
            text = " ".join(text.split())
            choices.append(f"{letter}) {text}")
        q_texts.append(q_text)
        radio_updates.append(gr.update(choices=choices, value=None))

    while len(q_texts) < 5:
        n = len(q_texts) + 1
        q_texts.append(f"{n}. (Question missing)")
        choices = ["A) (missing)", "B) (missing)", "C) (missing)", "D) (missing)"]
        radio_updates.append(gr.update(choices=choices, value=None))
    while len(answers) < 5:
        answers.append("A")

    return (q_texts[0], q_texts[1], q_texts[2], q_texts[3], q_texts[4],
            radio_updates[0], radio_updates[1], radio_updates[2], radio_updates[3], radio_updates[4],
            answers)


# -------------------------
# UI: Login screen
# -------------------------
def login_screen():
    with gr.Blocks() as login_app:
        gr.Markdown("## Welcome to Edututor AI")
        gr.Markdown("Please select your role and enter your details:")

        # role + name + email inputs
        role = gr.Radio(["Student", "Educator"], label="Role")
        username = gr.Textbox(label="Enter your name")
        email = gr.Textbox(label="Enter your Google email")
        proceed_btn = gr.Button("Proceed")

        # Small status area for validation messages
        status_msg = gr.Markdown("")

        # The two UI columns (keeps previous layout)
        student_ui = gr.Column(visible=False)
        educator_ui = gr.Column(visible=False)

        with student_ui:
            student_interface()

        with educator_ui:
            educator_dashboard()

        # route: store registration and show correct UI
        def handle_proceed(role, username, email):
            # basic validation
            if not role or not username or not email:
                return ("⚠️ Please enter role, name and email.",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None, None)

            try:
                if role == "Student":
                    register_student(username, email)
                    return (f"✅ Registered as Student: {username} ({email})",
                            gr.update(visible=True),
                            gr.update(visible=False),
                            username, email)
                else:
                    register_educator(username, email)
                    return (f"✅ Registered as Educator: {username} ({email})",
                            gr.update(visible=False),
                            gr.update(visible=True),
                            username, email)
            except Exception as e:
                return (f"❌ Registration failed: {e}",
                        gr.update(visible=False),
                        gr.update(visible=False),
                        None, None)

        # When Proceed is clicked:
        # outputs order must match the tuple returned by handle_proceed:
        # (status_msg_text, student_ui_visibility, educator_ui_visibility, user_name_state, user_email_state)
        proceed_btn.click(
            fn=handle_proceed,
            inputs=[role, username, email],
            outputs=[status_msg, student_ui, educator_ui, user_name, user_email]
        )

    return login_app


# -------------------------
# Screen 3A: Student quiz interface
# -------------------------
def student_interface():
    # Create Gradio interface
    with gr.Blocks() as student_app:
        gr.Markdown("# Educational AI Assistant")

        with gr.Tabs():
            with gr.TabItem("Concept Explanation"):
                concept_input = gr.Textbox(label="Enter a concept", placeholder="e.g., machine learning")
                explain_btn = gr.Button("Explain")
                explanation_output = gr.Textbox(label="Explanation", lines=10)

                explain_btn.click(concept_explanation, inputs=concept_input, outputs=explanation_output)

            with gr.TabItem("Quiz Generator"):
                quiz_input = gr.Textbox(label="Enter a topic", placeholder="e.g., physics")
                quiz_btn = gr.Button("Generate Quiz")

                status_msg = gr.Markdown("", elem_id="status")

                # Predefine 5 Markdown placeholders (question text) and 5 Radios (will be updated)
                q1 = gr.Markdown("")
                r1 = gr.Radio(choices=["A) (loading)", "B) (loading)", "C) (loading)", "D) (loading)"], label="Your answer for Q1")
                q2 = gr.Markdown("")
                r2 = gr.Radio(choices=["A) (loading)", "B) (loading)", "C) (loading)", "D) (loading)"], label="Your answer for Q2")
                q3 = gr.Markdown("")
                r3 = gr.Radio(choices=["A) (loading)", "B) (loading)", "C) (loading)", "D) (loading)"], label="Your answer for Q3")
                q4 = gr.Markdown("")
                r4 = gr.Radio(choices=["A) (loading)", "B) (loading)", "C) (loading)", "D) (loading)"], label="Your answer for Q4")
                q5 = gr.Markdown("")
                r5 = gr.Radio(choices=["A) (loading)", "B) (loading)", "C) (loading)", "D) (loading)"], label="Your answer for Q5")

                # State to store correct answers (list)
                correct_answers = gr.State([])

                # Loading indicator and chained generation (status -> generation -> clear)
                def show_loading():
                    return "⏳ Generating quiz… please wait."

                quiz_btn.click(
                    fn=show_loading,
                    inputs=[],
                    outputs=status_msg
                ).then(
                    fn=generate_interactive_quiz,
                    inputs=quiz_input,
                    outputs=[q1, q2, q3, q4, q5, r1, r2, r3, r4, r5, correct_answers]
                ).then(
                    fn=lambda: "",  # clear status after done
                    inputs=[],
                    outputs=status_msg
                )

                submit_btn = gr.Button("Submit Answers")
                result_output = gr.Textbox(label="Result", lines=5)

                def score_quiz(a1, a2, a3, a4, a5, correct):
                    # correct is the list we stored in the state
                    correct_list = correct or []
                    user_answers = [a1, a2, a3, a4, a5]
                    score = 0
                    feedback = []

                    for i, (ua, ca) in enumerate(zip(user_answers, correct_list)):
                        ua_letter = ""
                        if isinstance(ua, str) and ua.strip():
                            # Extract the first letter before ")"
                            ua_letter = ua.strip()[0].upper()

                        if ua_letter == ca:
                            score += 1
                            feedback.append(f"Q{i+1}: ✅ Correct")
                        else:
                            feedback.append(f"Q{i+1}: ❌ Incorrect (Correct: {ca})")

                    return f"Your Score: {score}/5\n\n" + "\n".join(feedback)


                submit_btn.click(
                    fn=score_quiz,
                    inputs=[r1, r2, r3, r4, r5, correct_answers],
                    outputs=[result_output]
                )



    return student_app


# -------------------------
# Screen 3B: Educator dashboard
# -------------------------
def educator_dashboard():
    with gr.Blocks() as educator_app:
        gr.Markdown("# Educator Dashboard")

        with gr.Tabs():
            with gr.TabItem("Student Progress"):
                gr.Markdown("View student performance and quiz results.")
                student_email = gr.Textbox(label="Enter Student Email")
                fetch_btn = gr.Button("Fetch Progress")
                progress_output = gr.Textbox(label="Progress Report", lines=10)

                def fetch_progress(email):
                    return fetch_progress_by_email(email)

                fetch_btn.click(fetch_progress, inputs=student_email, outputs=progress_output)

    return educator_app


# -------------------------
# Educator helper (reads from in-memory dashboard)
# -------------------------
def fetch_progress_by_email(email):
    try:
        rows = get_student_dashboard(email)  # this reads dashboard_db
        if not rows:
            return "No quiz records found for that student."
        report_lines = []
        for r in rows:
            report_lines.append(
                f"Topic: {r.get('topic')}\n"
                f"  Highest: {r.get('highest_score')}/5  "
                f"Attempts: {r.get('attempts_count')}  "
                f"Last: {r.get('last_attempt_score')}/5\n"
            )
        return "\n".join(report_lines)
    except Exception as e:
        return f"Error fetching dashboard data: {e}"


# -------------------------
# Launch
# -------------------------
if __name__ == "__main__":
    login_screen().launch()
