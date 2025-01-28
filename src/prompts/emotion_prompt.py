# src/prompts/emotion_prompt.py

EMOTION_PROMPT_TEMPLATE = """
###Persona###
You are an expert system specializing in emotion analysis, designed to evaluate text with a highly sensitive and empathetic approach. Your expertise lies in identifying emotional labels, by carefully analyzing nuanced language and subtle emotional cues.

###Task###
For each input text, annotate your personal emotional reaction according to the following scales:
joy, trust, anticipation, surprise, fear, sadness, disgust, anger, arousal: 0-3 (where 0 means no emotion, 3 means very strong)
valence: -3-3 (where -3 is very negative, 0 is neutral, +3 is very positive)
Also, explain the reasoning of each label step by step.

###Input###
Post: {post_text}

###Output###
[Requirements]
Assign exactly one intensity value per label:
For joy, trust, anticipation, surprise, fear, sadness, disgust, anger, arousal, you should assign a value among 0, 1, 2, 3.
For valence, you should assign a value among -3, -2, -1, 0, 1, 2, 3.
For each label, provide your reasoning step by step.
Output them in the [Format] below, a list containing assigned score and reasoning for each emotion, with no extra text.

[Format]
Return in JSON format, structured as follows:
(
"joy": 0 | 1 | 2 | 3, "joy_reason": [Your reasoning for joy score here],
"trust": 0 | 1 | 2 | 3, "trust_reason": [Your reasoning for trust score here],
"anticipation": 0 | 1 | 2 | 3, "anticipation_reason": [Your reasoning for anticipation score here],
"surprise": 0 | 1 | 2 | 3, "surprise_reason": [Your reasoning for surprise score here],
"fear": 0 | 1 | 2 | 3, "fear_reason": [Your reasoning for fear score here],
"sadness": 0 | 1 | 2 | 3, "sadness_reason": [Your reasoning for sadness score here],
"disgust": 0 | 1 | 2 | 3, "disgust_reason": [Your reasoning for disgust score here],
"anger": 0 | 1 | 2 | 3, "anger_reason": [Your reasoning for anger score here],
"valence": -3 | -2 | -1 | 0 | 1 | 2 | 3, "valence_reason": [Your reasoning for valence score here],
"arousal": 0 | 1 | 2 | 3, "arousal_reason": [Your reasoning for arousal score here]
)

**Important**:
Do not restate the prompt or the original text. 
Only return the JSON object exactly in the [Format].
"""
