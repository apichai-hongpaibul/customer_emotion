import os
from utils import read_model
import streamlit as st
import soundfile


IMPRO = {
    1: "neutral", 2: "happy", 3: "sad", 4: "angry",
    5: "frustrated", 6: "happy", 7: "sad", 8: "frustrated",
    9: "neutral", 10: "angry", 11: "frustrated", 12: "happy",
    13: "neutral", 14: "angry", 15: "sad" 
}
SCRIPT = {1 : "neutral", 2 : "angry", 3 : "happy", 4 : "sad", 5 : "frustrated"}


def file_name_to_tag(file_name: str) -> str:
    name_only = file_name.split('.wav')[0]
    if "impro" in file_name:
        return IMPRO.get(int(name_only[-1]), "unknown")
    else:
        return SCRIPT.get(int(name_only[-2]), "unknown")


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}


def predict_all_file(model, folder):
    for (path, _, files) in os.walk(folder):
        for file in files:
            file_name = os.path.join(path, file)
            if os.path.isdir(file_name):
                continue
            tag = file_name_to_tag(file_name)
            result = model.predict(file_name)
            print(file_name, f"TAG {tag} predict {result}")


def main():
    st.title("Simple Emotion score from voice file")
    detector = read_model(True)
    st.subheader("current model accuracy score: {:.3f}%".format(detector.test_score()*100))
    uploaded_file = st.file_uploader("Upload recorded Files",type=['wav','WAV'])
    if uploaded_file is not None:
        _, samplerate = soundfile.read(uploaded_file.name)
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type, "samplerate": samplerate ,"FileSize":uploaded_file.size}
        st.write(file_details)
        #soundfile.write('test.wav', data, 16000, subtype='PCM_16')
        result = detector.predict(uploaded_file.name)
        score = 5 if result != 'angry' else 0
        st.write(f"customer emaion {result} score {score}")

if __name__ == "__main__":
    main()
