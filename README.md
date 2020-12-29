# Naive Bayes Classifier
This is an implementation of Naive Bayes Classifer. The current implementation classifies instances into two different classes. This classifier works only on discrete variables. This classifier has been implemented using python, but it can be extended to classify on multiple class labels.

* [How to use](#how-to-use)
* [Input Format](#Input-format)
* [Output](#Output-Explained)
* [How does this work?](#how-does-this-work)
* [Time Complexity](#Time-Complexity)
* [Data Structures Used](#Data-Structures-Used)
* [Libraries Used](#Libraries-Used)
* [Further Improvements](#Further-Improvements)


## How to use
To use this implementation simply download the '.zip' file and run the 'NaiveBayes' python file. To run the python file we require a training file and a test file. The model is going to be trained and tested using the given files. The training and the test cases should be formatted as specified below. There should be atleast one instance of each class in the training file.

## Input Format
The format of the files should be as follows:
* Training File
Each line should start with the actual classLabel followed by the attricutes and values respectively. Using this file we train the model.

**ClassLabel _Attr1:Value_ _Attr2:Value_ ...**

* Test File
Similar to the training file, the test file requires the same format.

**ClassLabel _Attr1:Value_ _Attr2:Value_ ...**

## Output Explained
The file outputs two lines. The first line corresponds to the accuracy measures of the training file and the second line corresponds to the accuracy measures of the test file. Each line has four numbers in it, they are given below.

**true positive, false negative, false positive, true negative**

## How does this work?
This model first gets the count of each attribute in the training file. It then gets each instance from the test file and computes the probability of the test instance agaist each class label. The model predicts the instance to be the class label with the highest calculated probability. To prevent underflow we calculate the probability using logarithms.

[!NaiveBayes](/pics/NaiveBayes.png)

## Time Complexity
Since training the model is just parsing the file, the time taken is linear. The testing on the other hand requires the computation of probability which scales linearly with respect to the number of attributes. So the time complexity for Naive Bayes is O(N * d), where N is the number of instances and d is the number of attributes.

## Data Structures Used
* **Dictionaries**
* **Arrays**
* **Sets**

## Libraries Used
* os
* math
* sys

## Further Improvements
We can further improve this using smoothing. We can further extend this to accomodate continuous variables.
