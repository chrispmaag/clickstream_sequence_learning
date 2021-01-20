# Clickstream Sequence Learning

In this project, we will predict the outcomes of online browsing sessions through session-level features or sequence of page visits to predict if the next sequence will be a purchase or not.

![rnn_image_wiki](https://github.com/chrispmaag/clickstream_sequence_learning/blob/main/images/rnn_image_wiki.png)

![gru_gates_wiki](https://github.com/chrispmaag/clickstream_sequence_learning/blob/main/images/gru_gates_wiki.png)

![lstm_gates_wiki](https://github.com/chrispmaag/clickstream_sequence_learning/blob/main/images/lstm_gates_wiki.png)

We'll compare the results we get from using simple recurrent neural networks (RNNs), gated recurrent units (GRUs), and long short-term memory (LSTMs). 

## Library Requirements
- TensorFlow, pandas, and numpy

## Data
Two pickle files contain the different types of data. One pickle file, `shopping.pkl` has the timestamped session data. The other, `Session_features.pkl`, contains feature-level session data. 

## Get the Code
Download the `clickstream_sequence_learning.ipynb` file and run it. In order to run through the notebook, you'll have to provide your own pickle files with clickstream data, as this data set isn't publicly available.

## Results

After transforming the clickstream data in a sequence up to ten events long, 

![sequence_10](https://github.com/chrispmaag/clickstream_sequence_learning/blob/main/images/sequence_10.jpg)

we can fit an RNN, GRU, and LSTM to predict sessions that have a purchase in them.

The table below summarizes the results of the three models on the short sequences data.


| Data Type     | Model         | F1 Score|
|:------------- |:-------------:| -----:|
| Sequence <= 10| RNN           | 0.39 |
| Sequence <= 10| GRU           | 0.39 |
| Sequence <= 10| LSTM          | 0.37 |

For this smaller sequence version of the data we would prefer to use the simpler RNN model  because it achieved the same F1 score has the GRU model while having approximately three times fewer trainable parameters (12K vs 36K).


Next, we repeated the process, but expanded the sequence length to include up to 300 events.

| Data Type     | Model         | F1 Score|
|:------------- |:-------------:| -----:|
| Sequence <= 300| RNN           | 0.22 |
| Sequence <= 300| GRU           | 0.22 |
| Sequence <= 300| LSTM          | 0.23 |

It looks like the significantly longer sequences did not add much signal or the model hyperparameters need to be tweaked because the models struggled to achieve the same F1 scores as the models trained on the shorter sequences. This would be a good area to experiment with some of the hyperparameters of the models to see if we could get better results with the longer sequences.

Finally, we look at using the feature-based dataset.

| Data Type     | Model         | F1 Score|
|:------------- |:-------------:| -----:|
| Feature-based | RNN           | 0.82 |
| Feature-based | GRU           | 0.85 |
| Feature-based | LSTM          | 0.98 |


In this case, feature-level data offers a massive boost in our performance metrics over the sequence-level data. Our best performing LSTM using feature-level data had an F1 score of 0.98 and recall of 0.96. This significantly outperforms the best model trained on sequence data, a simple RNN which achieved an F1 score of 0.39.