import numpy as np
import cv2
import pickle
import os
import mediapipe as mp
from tensorflow import keras
from flask import Flask, Response, render_template
import json

browserExe = "chrome"

app = Flask(__name__)

if os.environ.get('WERKZEUG_RUN_MAIN') or Flask.debug is False:
    cap = cv2.VideoCapture(0)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

symptom_model = keras.models.load_model('/home/hackerpro/PycharmProjects/DumbDiseasePredictor/symptom.h5')
pickle_in = open("disease_predict_model.pkl", "rb")
disease_model = pickle.load(pickle_in)

sequence = []
sentence = []
predictions = []

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

actions = os.listdir(DATA_PATH)

actions = np.array(actions)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def generate_frames():
    # 1. New detection variables
    global sentence, sequence, predictions, actions, disease_model
    threshold = 0.5
    index = 0
    find_test_symp = True
    disp_disease = False
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            detector = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if len(sequence) == 30 and find_test_symp:
                res = symptom_model.predict(np.expand_dims(sequence, axis=0))[0]
                print(res[np.argmax(res)])
                acc = res[np.argmax(res)]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if len(sentence) <= 4:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    cv2.putText(frame, sentence[index], (x + w - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 0, 0), 2
                                                )
                                    cv2.putText(frame, str(int(acc * 100)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (255, 0, 0), 2)
                                    index += 1
                            else:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    cv2.putText(frame, sentence[index], (x + w - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 0, 0), 2)
                                    cv2.putText(frame, str(int(acc * 100)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                1, (255, 0, 0), 2)
                                    index += 1
                                pass

                        else:
                            sentence.append(actions[np.argmax(res)])
                            cv2.putText(frame, sentence[index], (x + w - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 0), 2)
                            cv2.putText(frame, str(int(acc * 100)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 0), 2)

                            index += 1

                cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            if len(sentence) >= 5:

                find_test_symp = False

                disease_test = np.zeros(21, dtype=int)

                with open('prognosis.json', 'r') as openfile:
                    # Reading from json file
                    dis_list = json.load(openfile)

                with open('column_names.json', 'r') as openfile:
                    # Reading from json file
                    col_list = json.load(openfile)

                saved_index = []
                for sen in sentence:
                    for key, value in col_list.items():
                        if value == sen:
                            saved_index.append(int(key))

                for i in saved_index:
                    disease_test[i] = 1

                p = disease_model.predict([disease_test])

                for key, value in dis_list.items():
                    if int(key) == p[0]:
                        cv2.putText(frame, value, (x + w - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2)
            if 0xFF == ord('q'):
                disp_disease = True

            if disp_disease:
                break
            print(sentence)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        os.system("pkill " + browserExe)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
