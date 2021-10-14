import os
from utils import read_model
import argparse


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="simple predict for test.wav")
    args = parser.parse_args()
    detector = read_model()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_path, "data")
    if args.test:
        result = detector.predict("test.wav")
        print("test.wav", "unknown", result)
    else:
        predict_all_file(detector, data_folder)
