from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from proctor import generate_frames, get_warning_count, increment_warning
from questions import questions
import mysql.connector

app = Flask(__name__)
app.secret_key = "supersecretkey"


# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="exam_system"
    )


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("login.html")


# ---------------- REGISTER ----------------
@app.route('/register', methods=["POST"])
def register():
    role = request.form["role"]
    fullname = request.form["fullname"]
    email = request.form["email"]
    userid = request.form["userid"]
    password = request.form["password"]

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password, role, userid) VALUES (%s, %s, %s, %s, %s)",
            (fullname, email, password, role, userid)
        )
        conn.commit()
    except Exception as e:
        return str(e)

    cursor.close()
    conn.close()

    return redirect(url_for("home"))


# ---------------- LOGIN ----------------
@app.route('/login', methods=["POST"])
def login():
    role = request.form["role"]
    userid = request.form["userid"]
    password = request.form["password"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # ✅ fetch only by userid
    cursor.execute(
        "SELECT * FROM users WHERE userid=%s",
        (userid,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:
        if user["password"] == password and user["role"] == role:
            session["user"] = user["id"]
            session["role"] = role
            return redirect(url_for("dashboard"))
        else:
            return "Invalid Credentials"
    else:
        return "User not found"


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("home"))


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if "user" not in session:
        return redirect(url_for("home"))

    user_id = session["user"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
    user = cursor.fetchone()

    cursor.execute("""
        SELECT * FROM results 
        WHERE user_id=%s 
        ORDER BY id DESC LIMIT 1
    """, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return render_template("dashboard.html", fullname=user["name"], result=result)


# ---------------- ADMIN PANEL ----------------
@app.route('/admin')
def admin():
    if "role" not in session or session["role"] != "admin":
        return "Access Denied"

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Users
    cursor.execute("SELECT id, name, email, role FROM users")
    users = cursor.fetchall()

    # Results
    cursor.execute("""
        SELECT users.name, results.score, results.warnings, results.created_at
        FROM results
        JOIN users ON users.id = results.user_id
    """)
    results = cursor.fetchall()

    # Logs
    cursor.execute("""
        SELECT users.name, logs.event_type, logs.timestamp
        FROM logs
        JOIN users ON users.id = logs.user_id
    """)
    logs = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin.html", users=users, results=results, logs=logs)


# ---------------- LOG EVENT (🔥 NEW) ----------------
@app.route('/log_event', methods=["POST"])
def log_event():
    if "user" not in session:
        return jsonify({"status": "error"})

    data = request.get_json()
    event = data.get("event")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO logs (user_id, event_type) VALUES (%s, %s)",
        (session["user"], event)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"status": "ok"})


# ---------------- EXAM ----------------
@app.route('/exam')
def exam():
    if "user" not in session:
        return redirect(url_for("home"))

    return render_template("exam.html", questions=questions)


# ---------------- SUBMIT EXAM ----------------
@app.route('/submit', methods=["POST"])
def submit():
    if "user" not in session:
        return redirect(url_for("home"))

    score = 0
    total_questions = 0

    for subject, subject_questions in questions.items():
        for i, q in enumerate(subject_questions):
            selected = request.form.get(f"{subject}_{i}")
            total_questions += 1
            if selected:
                if selected == q["answer"]:
                    score += 4
                else:
                    score -= 1

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO results (user_id, score, warnings) VALUES (%s, %s, %s)",
        (session["user"], score, get_warning_count())
    )

    conn.commit()
    cursor.close()
    conn.close()

    return render_template("result.html", score=score, total=total_questions * 4)


# ---------------- TAB SWITCH ----------------
@app.route('/tab_switch', methods=["POST"])
def tab_switch():
    increment_warning("Tab_Switch")

    # also log to DB
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO logs (user_id, event_type) VALUES (%s, %s)",
        (session["user"], "Tab Switch")
    )

    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"status": "ok"})


# ---------------- WARNING STATUS ----------------
@app.route('/warning_status')
def warning_status():
    return jsonify({"warning_count": get_warning_count()})


# ---------------- CAMERA ----------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)