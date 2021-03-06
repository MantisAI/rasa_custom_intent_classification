# Example of implementing custom classifiers for Rasa

This repository is an example of a simple Rasa project that shows how you can implement your own classifiers.
The data was automatically generated by running:
```
rasa init
```

## The SkLearn Classifier
`sklearn_classifier.py` exemplifies implementing a simple SVM classifier. 

## Hugging Face Transformers Classifier
`transformer_classifier.py` can take any Hugging Face transformer that has an AutoModelForSequenceClassification implementation and train it. We're using `albert-base-v2` as an example because it is a small model trainable without the need for a GPU.

## config.yml

`config.yml` contains examples of how you would insert the two classifiers into the Rasa pipeline, and how you would pass parameters to them.

# Running the project

To train the custom rasa intent classification model:
```rasa train nlu```

To test the model:
```rasa test nlu --nlu <path_to_test_data>```

To interact with the model:
```rasa shell```

