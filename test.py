import matplotlib.pyplot as plt

# Emotion Summary Report data
emotion_summary = {
    "Sad": [
        "2023-06-13 12:17:03", "2023-06-13 12:17:04", "2023-06-13 12:17:07",
        "2023-06-13 12:22:54", "2023-06-13 12:22:55", "2023-06-13 12:22:56",
        "2023-06-13 12:48:24", "2023-06-13 12:48:25", "2023-06-13 12:48:25",
        "2023-06-13 12:48:25", "2023-06-13 12:48:41", "2023-06-13 12:48:44"
    ],
    "Happy": [
        "2023-06-13 12:17:04", "2023-06-13 12:21:05", "2023-06-13 12:22:56",
        "2023-06-13 12:22:57", "2023-06-13 12:22:57", "2023-06-13 12:48:24",
        "2023-06-13 12:48:27", "2023-06-13 12:48:28", "2023-06-13 12:48:28",
        "2023-06-13 12:48:29", "2023-06-13 12:48:30", "2023-06-13 12:48:31",
        "2023-06-13 12:48:32", "2023-06-13 12:48:32", "2023-06-13 12:48:34",
        "2023-06-13 12:48:34", "2023-06-13 12:48:34", "2023-06-13 12:48:35",
        "2023-06-13 12:48:35", "2023-06-13 12:48:35", "2023-06-13 12:48:35",
        "2023-06-13 12:48:36", "2023-06-13 12:48:36", "2023-06-13 12:48:36",
        "2023-06-13 12:48:36", "2023-06-13 12:48:37", "2023-06-13 12:48:37",
        "2023-06-13 12:48:37", "2023-06-13 12:48:37", "2023-06-13 12:48:37",
        "2023-06-13 12:48:38", "2023-06-13 12:48:38", "2023-06-13 12:48:38",
        "2023-06-13 12:48:38", "2023-06-13 12:48:39", "2023-06-13 12:48:39",
        "2023-06-13 12:48:40", "2023-06-13 12:48:40", "2023-06-13 12:48:40",
        "2023-06-13 12:48:41", "2023-06-13 12:48:41", "2023-06-13 12:48:42",
        "2023-06-13 12:48:42", "2023-06-13 12:48:42", "2023-06-13 12:48:43",
        "2023-06-13 12:48:43", "2023-06-13 12:48:43", "2023-06-13 12:48:44",
        "2023-06-13 12:48:45", "2023-06-13 12:48:45", "2023-06-13 12:48:45",
        "2023-06-13 12:48:45", "2023-06-13 12:48:46", "2023-06-13 12:48:47"
    ],
    "Surprise": [
        "2023-06-13 12:22:55", "2023-06-13 12:22:56", "2023-06-13 12:22:56"
    ],
    "Fear": [
        "2023-06-13 12:22:55", "2023-06-13 12:35:44", "2023-06-13 12:48:46"
    ],
    "Neutral": [
        "2023-06-13 12:48:25", "2023-06-13 12:48:26", "2023-06-13 12:48:26",
        "2023-06-13 12:48:26", "2023-06-13 12:48:26", "2023-06-13 12:48:27",
        "2023-06-13 12:48:27", "2023-06-13 12:48:27", "2023-06-13 12:48:27",
        "2023-06-13 12:48:28", "2023-06-13 12:48:28", "2023-06-13 12:48:28",
        "2023-06-13 12:48:28", "2023-06-13 12:48:29", "2023-06-13 12:48:29",
        "2023-06-13 12:48:29", "2023-06-13 12:48:29", "2023-06-13 12:48:30",
        "2023-06-13 12:48:30", "2023-06-13 12:48:30", "2023-06-13 12:48:30",
        "2023-06-13 12:48:31", "2023-06-13 12:48:31", "2023-06-13 12:48:31",
        "2023-06-13 12:48:31", "2023-06-13 12:48:31", "2023-06-13 12:48:32",
        "2023-06-13 12:48:32", "2023-06-13 12:48:32", "2023-06-13 12:48:32",
        "2023-06-13 12:48:33", "2023-06-13 12:48:33", "2023-06-13 12:48:33",
        "2023-06-13 12:48:33", "2023-06-13 12:48:34", "2023-06-13 12:48:34",
        "2023-06-13 12:48:34", "2023-06-13 12:48:34", "2023-06-13 12:48:35",
        "2023-06-13 12:48:35", "2023-06-13 12:48:35", "2023-06-13 12:48:35",
        "2023-06-13 12:48:36", "2023-06-13 12:48:36", "2023-06-13 12:48:36",
        "2023-06-13 12:48:36", "2023-06-13 12:48:37", "2023-06-13 12:48:37",
        "2023-06-13 12:48:37", "2023-06-13 12:48:37", "2023-06-13 12:48:37",
        "2023-06-13 12:48:38", "2023-06-13 12:48:38", "2023-06-13 12:48:38",
        "2023-06-13 12:48:38", "2023-06-13 12:48:39", "2023-06-13 12:48:39",
        "2023-06-13 12:48:40", "2023-06-13 12:48:40", "2023-06-13 12:48:40",
        "2023-06-13 12:48:40", "2023-06-13 12:48:41", "2023-06-13 12:48:41",
        "2023-06-13 12:48:41", "2023-06-13 12:48:41", "2023-06-13 12:48:42",
        "2023-06-13 12:48:42", "2023-06-13 12:48:42", "2023-06-13 12:48:42",
        "2023-06-13 12:48:43", "2023-06-13 12:48:43", "2023-06-13 12:48:43",
        "2023-06-13 12:48:43", "2023-06-13 12:48:44", "2023-06-13 12:48:44",
        "2023-06-13 12:48:44", "2023-06-13 12:48:44", "2023-06-13 12:48:45",
        "2023-06-13 12:48:45", "2023-06-13 12:48:45", "2023-06-13 12:48:45",
        "2023-06-13 12:48:46", "2023-06-13 12:48:46", "2023-06-13 12:48:46",
        "2023-06-13 12:48:46", "2023-06-13 12:48:46", "2023-06-13 12:48:47"
    ]
}

# Count the occurrences of each emotion
emotion_counts = {emotion: len(dates) for emotion, dates in emotion_summary.items()}

# Plot the emotion summary
emotions = list(emotion_counts.keys())
counts = list(emotion_counts.values())

plt.bar(emotions, counts)
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.title("Emotion Summary Report")
plt.show()