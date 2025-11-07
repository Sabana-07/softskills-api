from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model
clf = joblib.load('softskills_clf.joblib')

# Function to map answers to categories
def map_to_categories(answers):
    c = {}
    c['communication'] = (answers[0] + answers[1]) / 10 * 100
    c['leadership'] = (answers[2] / 5) * 100
    c['adaptability'] = (answers[3] / 5) * 100
    c['teamwork'] = (answers[4] / 5) * 100
    c['time_management'] = (answers[5] / 5) * 100
    c['problem_solving'] = (answers[6] + answers[9]) / 10 * 100
    c['emotional_intelligence'] = (answers[7] / 5) * 100
    c['negotiation'] = (answers[8] / 5) * 100
    return c

def recommend_for_profile(profile):
    recs = {
        'Communicator': [
            "Join a public speaking club like Toastmasters.",
            "Take an 'Effective Communication' course on Coursera.",
            "Practice summarizing ideas in 60 seconds."
        ],
        'Leader': [
            "Take a short leadership & people management course.",
            "Lead a small team project or volunteer group.",
            "Practice giving constructive feedback sessions."
        ],
        'Problem Solver': [
            "Practice case studies and puzzles to build structured thinking.",
            "Attend design-thinking workshops.",
            "Participate in hackathons or problem-solving groups."
        ],
        'Reliable Contributor': [
            "Use Trello/Asana for productivity.",
            "Plan weekly goals and deadlines.",
            "Follow a daily time-blocking routine."
        ],
        'Needs Development': [
            "Take beginner communication & teamwork courses.",
            "Join group projects for practical learning.",
            "Find a mentor and ask for feedback."
        ]
    }
    return recs.get(profile, [])

app = Flask(__name__)

@app.route('/')
def home():
    return {"message": "Soft Skills API is running ✅"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "answers" not in data:
        return jsonify({"error": "Please send 'answers' as a list of 10 numbers"}), 400

    answers = data['answers']
    if len(answers) != 10:
        return jsonify({"error": "Exactly 10 answers are required"}), 400

    # map and predict
    cats = map_to_categories(answers)
    X_user = pd.DataFrame([cats])
    profile = clf.predict(X_user)[0]
    recs = recommend_for_profile(profile)
    return jsonify({
        "profile": profile,
        "recommendations": recs,
        "categories": cats
    })

if __name__ == '__main__':
    app.run(debug=True)
