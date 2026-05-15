import cv2
import numpy as np
import webbrowser
from tensorflow.keras.models import load_model

IMG_SIZE = (96, 96)

# Load trained model
model = load_model("emotion_model.h5")

# Class labels
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion -> Movie Playlist Mapping (YouTube playlists)
emotion_movies = {
    'happy': {
        'movies': ['Vadivelu Comedy Collection', 'Oh My Kadavule', 'Kannum Kannum Kollaiyadithaal', '3 Idiots', 'Santhanam Comedy'],
        'url': 'https://www.youtube.com/results?search_query=best+comedy+full+movies+playlist'
    },
    'sad': {
        'movies': ['96 Tamil Movie', 'Mersal', 'Vinnaithaandi Varuvaayaa', 'Ae Dil Hai Mushkil', 'The Pursuit of Happyness'],
        'url': 'https://www.youtube.com/results?search_query=feel+good+emotional+full+movies+playlist'
    },
    'angry': {
        'movies': ['Vikram', 'Beast', 'Varisu', 'KGF Chapter 2', 'Baahubali'],
        'url': 'https://www.youtube.com/results?search_query=mass+action+full+movies+playlist'
    },
    'fear': {
        'movies': ['Demonte Colony', 'Iruttu', 'Ratchasan', 'Conjuring', 'Get Out'],
        'url': 'https://www.youtube.com/results?search_query=thriller+horror+full+movies+playlist'
    },
    'surprise': {
        'movies': ['Vikram Vedha', 'Ratchasan', 'Master', 'Andhadhun', 'Shutter Island'],
        'url': 'https://www.youtube.com/results?search_query=suspense+twist+full+movies+playlist'
    },
    'disgust': {
        'movies': ['Sarpatta Parambarai', 'Super Deluxe', 'Jai Bhim', 'Article 15', 'Taare Zameen Par'],
        'url': 'https://www.youtube.com/results?search_query=inspirational+full+movies+playlist'
    },
    'neutral': {
        'movies': ['Pariyerum Perumal', 'Kaithi', 'Peranbu', 'Drishyam', 'Forrest Gump'],
        'url': 'https://www.youtube.com/results?search_query=best+drama+full+movies+playlist'
    }
}

# Colors for each emotion (BGR)
emotion_colors = {
    'happy':    (0, 255, 255),
    'sad':      (255, 100, 100),
    'angry':    (0, 0, 255),
    'fear':     (128, 0, 128),
    'surprise': (0, 165, 255),
    'disgust':  (0, 128, 0),
    'neutral':  (200, 200, 200)
}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

print("=" * 50)
print("  EMOTION BASED MOVIE RECOMMENDER")
print("  By Varsha")
print("=" * 50)
print("Press 'S' on camera window to get movie playlist")
print("Press 'Q' on camera window to quit")
print("=" * 50)

current_emotion = 'neutral'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMG_SIZE)
        face_array = np.expand_dims(face_resized, axis=0) / 255.0

        prediction = model.predict(face_array, verbose=0)
        emotion_index = np.argmax(prediction)
        current_emotion = classes[emotion_index]
        confidence = round(float(np.max(prediction)) * 100, 1)

        color = emotion_colors[current_emotion]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label = f"{current_emotion.upper()}  {confidence}%"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        movies = emotion_movies[current_emotion]['movies']
        cv2.putText(frame, "Recommended Movies:", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, movie in enumerate(movies[:3]):
            cv2.putText(frame, f"  {i+1}. {movie}", (x, y + h + 40 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
    cv2.putText(frame, "EMOTION MOVIE RECOMMENDER | Press S=Playlist  Q=Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow('Emotion Based Movie Recommender - Varsha', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') or key == ord('S'):
        url = emotion_movies[current_emotion]['url']
        print(f"\n Detected Emotion  : {current_emotion.upper()}")
        print(f" Opening YouTube Playlist for: {current_emotion} movies...")
        print(f" Recommended Movies: {', '.join(emotion_movies[current_emotion]['movies'])}")
        webbrowser.open(url)

    elif key == ord('q') or key == ord('Q'):
        print("\nThank you for using Emotion Based Movie Recommender!")
        break

cap.release()
cv2.destroyAllWindows()