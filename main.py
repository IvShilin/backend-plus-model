import os

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from cv_scan import process_image_and_find_solution

app = Flask(__name__)

users = {
    "login": "test",
    "admin": "test"
}
tests = {}

if "USERS" in os.environ:
    users.update({k: v for k, v in [pair.split(":") for pair in os.environ["USERS"].split(",")]})


@app.route('/upload', methods=['POST'])
def upload_photo():
    test_number = int(request.form['test_number'])
    photo = request.files['photo']

    if test_number not in tests:
        return jsonify({"error": "Номер теста не найден"})

    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)

    filename = secure_filename(photo.filename)
    photo.save(os.path.join(photos_dir, filename))

    correct_answers = [{"question": str(i), "correct_answer": answer} for i, answer in tests[test_number].items()]

    try:
        answer = process_image_and_find_solution(os.path.join(photos_dir, filename), correct_answers, test_number)
    except ValueError as e:
        return jsonify({"error": f"Ошибка процессинга изображения: {e}", "answer": None})

    if answer is None:
        return jsonify({"error": "Invalid photo format"})

    if "answer" not in answer or "total-correct-answers" not in answer or "total-incorrect-answers" not in answer:
        return jsonify({"error": "Invalid photo format"})

    result = {
        "qr_info": answer["qr_info"],
        "answers": answer["answer"],
        "total-correct-answers": answer["total-correct-answers"],
        "total-incorrect-answers": answer["total-incorrect-answers"],
        "test_number": answer["total-incorrect-answers"]
    }

    return jsonify(result)


@app.route('/auth', methods=['POST'])
def auth():
    data = request.get_json()
    login = data.get("login")
    password = data.get("password")
    test_number = int(data.get("number"))
    answers = data.get("test")

    if login not in users or users[login] != password:
        return jsonify({"error": "Invalid login or password"}), 401

    if test_number not in tests:
        tests[test_number] = {}

    for answer in answers:
        question = answer.get("question")
        correct_answer = answer.get("correct_answer")
        tests[test_number][question] = correct_answer

    return jsonify({"result": "ok", "test_data": tests[test_number]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
