import os

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from rasa.nlu.classifiers.classifier import IntentClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    """
    Dataset for training the model.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    Helper function to compute aggregated metrics from predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

class TransformerClassifier(IntentClassifier):
    name = "transformer_classifier"
    provides = ["intent"]
    requires = ["text"]
    defaults = {}
    language_list = ["en"]
    model_name = "roberta-base"

    def __init__(self, component_config=None):
        self.model_name = component_config.get("model_name", "albert-base-v2")
        super().__init__(component_config)

    def _define_model(self):
        """
        Loads the pretrained model and the configuration after the data has been preprocessed.
        """

        self.config = AutoConfig.from_pretrained(self.model_name)
        self.config.id2label = self.id2label
        self.config.label2id = self.label2id
        self.config.num_labels = len(self.id2label)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.config
        )

    def _compute_label_mapping(self, labels):
        """
        Maps the labels to integers and stores them in the class attributes.
        """

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels)
        self.label2id = {}
        self.id2label = {}
        for label in np.unique(labels):
            self.label2id[label] = int(label_encoder.transform([label])[0])
        for i in integer_encoded:
            self.id2label[int(i)] = label_encoder.inverse_transform([i])[0]

    def _preprocess_data(self, data, params):
        """
        Preprocesses the data to be used for training.
        """

        documents = []
        labels = []
        for message in data.training_examples:
            if "text" in message.data:
                documents.append(message.data["text"])
                labels.append(message.data["intent"])
        self._compute_label_mapping(labels)
        targets = [self.label2id[label] for label in labels]
        encodings = self.tokenizer(
            documents,
            padding="max_length",
            max_length=params.get("max_length", 64),
            truncation=True,
        )
        dataset = CustomDataset(encodings, targets)

        return dataset

    def train(self, train_data, cfg, **kwargs):
        """
        Preprocesses the data, loads the model, configures the training and trains the model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset = self._preprocess_data(train_data, self.component_config)
        self._define_model()

        training_args = TrainingArguments(
            output_dir="./custom_model",
            num_train_epochs=self.component_config.get("epochs", 15),
            evaluation_strategy="no",
            per_device_train_batch_size=self.component_config.get("batch_size", 24),
            warmup_steps=self.component_config.get("warmup_steps", 500),
            weight_decay=self.component_config.get("weight_decay", 0.01),
            learning_rate=self.component_config.get("learning_rate", 2e-5),
            lr_scheduler_type=self.component_config.get("scheduler_type", "constant"),
            save_strategy="no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def _process_intent_ranking(self, outputs):
        """
        Processes the intent ranking, sort in descending order based on confidence. Get only top 10

        Args:
            outputs: model outputs

        Returns:
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """

        confidences = [float(x) for x in outputs["logits"][0]]
        intent_names = list(self.label2id.keys())
        intent_ranking_all = zip(confidences, intent_names)
        intent_ranking_all_sorted = sorted(
            intent_ranking_all, key=lambda x: x[0], reverse=True
        )
        intent_ranking = [
            {"confidence": x[0], "intent": x[1]} for x in intent_ranking_all_sorted[:10]
        ]
        return intent_ranking

    def _predict(self, text):
        """
        Predicts the intent of the input text.

        Args:
            text (str): input text

        Returns:
            prediction (string) - intent name
            confidence (float) - confidence of the intent
            intent_ranking (list) - list of dicts with intent name and confidence (top 10 only)
        """

        inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.component_config.get("max_length", 64),
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = self.model(**inputs)

        confidence = float(outputs["logits"][0].max())
        prediction = self.id2label[int(outputs["logits"][0].argmax())]
        intent_ranking = self._process_intent_ranking(outputs)

        return prediction, confidence, intent_ranking

    def process(self, message, **kwargs):
        """
        Processes the input given from Rasa. Attaches the output to the message object.

        Args:
            message (Message): input message
        """

        text = message.data["text"]
        prediction, confidence, intent_ranking = self._predict(text)

        message.set(
            "intent", {"name": prediction, "confidence": confidence}, add_to_output=True
        )
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name, model_dir):
        """
        Persists the model, the tokenizer and it's configuration to the given path. Will be archived in a .tar.gz by Rasa.

        Args:
            file_name (str): name of the component given by Rasa based on order in the interpreter pipeline
            model_dir (str): path to the interpreter model directory

        Returns:
            model_metadata (dict): dictionary with the model, the tokenizer and the configuration names that will help load it
        """

        tokenizer_filename = "tokenizer_{}".format(file_name)
        model_filename = "model_{}".format(file_name)
        config_filename = "config_{}".format(file_name)
        tokenizer_path = os.path.join(model_dir, tokenizer_filename)
        model_path = os.path.join(model_dir, model_filename)
        config_path = os.path.join(model_dir, config_filename)
        self.tokenizer.save_pretrained(tokenizer_path)
        self.model.save_pretrained(model_path)
        self.config.save_pretrained(config_path)

        return {
            "config": config_filename,
            "tokenizer": tokenizer_filename,
            "model": model_filename,
        }

    @classmethod
    def load(
        cls, meta, model_dir=None, model_metadata=None, cached_component=None, **kwargs
    ):
        """
        Loads the model, tokenizer and configuration from the given path.

        Returns:
            component (Component): loaded component
        """

        tokenizer_filename = meta.get("tokenizer")
        model_filename = meta.get("model")
        config_filename = meta.get("config")
        tokenizer_path = os.path.join(model_dir, tokenizer_filename)
        model_path = os.path.join(model_dir, model_filename)
        config_path = os.path.join(model_dir, config_filename)

        x = cls(meta)
        x.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        x.config = AutoConfig.from_pretrained(config_path)
        x.id2label = x.config.id2label
        x.label2id = x.config.label2id
        x.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=x.config
        ).to(DEVICE)

        return x
