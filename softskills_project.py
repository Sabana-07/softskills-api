# Step 6: synthetic data generation + utility functions
import random
import pandas as pd
import numpy as np

questions = [
    "I find it easy to explain my ideas clearly to others.",
    "I feel comfortable speaking in front of a group.",
    "I often take the lead when a group needs direction.",
    "I adapt quickly when project requirements change.",
    "I actively listen and consider others’ opinions.",
    "I manage my tasks to meet deadlines consistently.",
    "I stay calm and find solutions during stressful situations.",
    "I give and receive constructive feedback without taking it personally.",
    "I am comfortable negotiating to reach mutual agreements.",
    "I use creative approaches to solve work problems."
]

def map_to_categories(answers):
    """answers: list of 10 ints (1-5). returns dict of category scores (0-100)."""
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

def assign_profile_from_categories(c):
    """Simple rule-based profile assignment from category scores."""
    profiles = []
    if c['communication'] >= 70 and c['teamwork'] >= 60:
        profiles.append('Communicator')
    if c['leadership'] >= 60 and c['communication'] >= 50:
        profiles.append('Leader')
    if c['problem_solving'] >= 70:
        profiles.append('Problem Solver')
    if c['time_management'] >= 70 and c['teamwork'] >= 60:
        profiles.append('Reliable Contributor')
    if not profiles:
        profiles.append('Needs Development')
    return profiles[0]  # primary profile

def gen_random_response(bias=None):
    """Generate 10 answers (1-5). Optional bias to increase certain skills."""
    r = [random.randint(1,5) for _ in range(10)]
    if bias == 'comm':
        r[0] = min(5, r[0] + random.randint(1,2)); r[1] = min(5, r[1] + random.randint(1,2))
    if bias == 'lead':
        r[2] = min(5, r[2] + random.randint(1,2))
    if bias == 'ps':
        r[6] = min(5, r[6] + random.randint(1,2)); r[9] = min(5, r[9] + random.randint(1,2))
    if bias == 'time':
        r[5] = min(5, r[5] + random.randint(1,2))
    return r

def make_dataset(n=800):
    rows = []
    biases = [None, 'comm', 'lead', 'ps', 'time']
    for _ in range(n):
        b = random.choice(biases)
        answers = gen_random_response(b)
        cats = map_to_categories(answers)
        profile = assign_profile_from_categories(cats)
        row = cats.copy()
        row['profile'] = profile
        rows.append(row)
    return pd.DataFrame(rows)

# create dataset
df = make_dataset(800)
print("Dataset created. Shape:", df.shape)
print("Class distribution:\n", df['profile'].value_counts())

# Step 7: train Decision Tree and inspect
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# features and label
X = df.drop(columns=['profile'])
y = df['profile']

# split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Show decision rules (text)
rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree rules:\n")
print(rules)

# Step 8: save model to file
import joblib
joblib.dump(clf, 'softskills_clf.joblib')
print("Saved model to softskills_clf.joblib")

# Step 9: interactive demo (console)
def recommend_for_profile(profile):
    recs = {
        'Communicator': [
            "Join a public speaking club like Toastmasters.",
            "Take an 'Effective Communication' course on Coursera/LinkedIn.",
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
            "Participate in problem-solving groups/hackathons."
        ],
        'Reliable Contributor': [
            "Adopt time-blocking and use a task manager (Trello/Asana).",
            "Do small deadline-driven mini-projects.",
            "Follow a daily planning routine."
        ],
        'Needs Development': [
            "Start with beginner soft-skill courses: communication & time management.",
            "Seek mentorship and join group projects for practice.",
            "Work on self-reflection and short weekly goals."
        ]
    }
    return recs.get(profile, [])

def console_demo():
    print("\n--- Soft Skills Assessment (Console Demo) ---")
    print("Answer on a scale 1 (Strongly Disagree) to 5 (Strongly Agree).\n")
    answers = []
    for i, q in enumerate(questions):
        while True:
            try:
                v = int(input(f"Q{i+1}. {q} (1-5): ").strip())
                if v < 1 or v > 5:
                    raise ValueError
                answers.append(v)
                break
            except ValueError:
                print("Please enter an integer between 1 and 5.")
    cats = map_to_categories(answers)
    print("\nCategory scores (0-100):")
    for k, v in cats.items():
        print(f" - {k}: {v:.1f}")
    X_user = pd.DataFrame([cats])
    pred = clf.predict(X_user)[0]
    print(f"\nPredicted Profile: {pred}\nRecommendations:")
    for r in recommend_for_profile(pred):
        print(" -", r)

# if you want to run the interactive demo directly:
if __name__ == '__main__':
    console_demo()

# Step 7: train Decision Tree and inspect
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# features and label
X = df.drop(columns=['profile'])
y = df['profile']

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Show decision rules (text)
rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree rules:\n")
print(rules)

import joblib
joblib.dump(clf, 'softskills_clf.joblib')
print("✅ Model saved as softskills_clf.joblib")

