from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.utils import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import os
import numpy as np

class CustomIntentSVM(IntentClassifier):
        name = "svm"
        provides = ["intent"]
        requires = ["text"]
        defaults = {}
        language_list = ["en"]

        def __init__(self, component_config=None):
                super().__init__(component_config)

        def _define_model(self):
                clf = SVC(
                        kernel=self.component_config['kernel'],
                        coef0=self.component_config['coef0'], 
                        degree=self.component_config['degree'], 
                        class_weight=self.component_config['class_weight']
                )
                pipeline = Pipeline([("vect", TfidfVectorizer()), ("clf", clf)])
                return pipeline

        def _transform_data(self, data):
                documents = []
                labels = []
                for message in data.training_examples:
                        if "text" in message.data:
                                documents.append(message.data["text"])
                                labels.append(message.data["intent"])
                return documents, labels

        def train(self, train_data, cfg, **kwargs):
                self.model = self._define_model()
                documents, labels = self._transform_data(train_data)
                self.model.fit(documents, labels)

        def _predict(self, text):
                prediction = self.model.predict([text])[0]
                confidences = self.model.decision_function([text])
                confidence = max(np.round(confidences / 100, 4)[0])
                return prediction, confidence

        def _convert_to_rasa(self, prediction, confidence):
                intent = {"name": prediction, "confidence": confidence}
                return intent

        def process(self, message, **kwargs):
                text = message.data["text"]
                prediction, confidence = self._predict(text)
                intent = self._convert_to_rasa(prediction, confidence)
                message.set("intent", intent, add_to_output=True)

        def persist(self, file_name, model_dir):
                path = os.path.join(model_dir, file_name)
                io.json_pickle(path, self)
                return {"file": file_name}

        @classmethod
        def load(cls, meta, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
                file_name = meta.get("file")
                path = os.path.join(model_dir, file_name)
                return io.json_unpickle(path)
