

import numpy as np
import pandas as pd

np.random.seed(42)

num_students = 200

# Generate somewhat realistic feature values
study_hours = np.random.uniform(0, 10, num_students)          
attendance = np.random.uniform(40, 100, num_students)        
previous_marks = np.random.uniform(30, 100, num_students)     
assignment_scores = np.random.uniform(30, 100, num_students) 

# A simple "score" that decides Pass/Fail, plus some randomness/noise
# so the data is not perfectly separable (more realistic).
score = (
    0.35 * study_hours * 10 +
    0.25 * attendance +
    0.25 * previous_marks +
    0.15 * assignment_scores
)

noise = np.random.normal(0, 5, num_students)
final_score = score + noise

# Pass if final_score crosses a threshold

result = np.where(final_score >= 65, 1, 0)  

df = pd.DataFrame({
    "Study_Hours": np.round(study_hours, 1),
    "Attendance": np.round(attendance, 1),
    "Previous_Marks": np.round(previous_marks, 1),
    "Assignment_Scores": np.round(assignment_scores, 1),
    "Result": result
})

df.to_csv("student_data.csv", index=False)
print("student_data.csv created with", num_students, "rows")
print(df["Result"].value_counts())
