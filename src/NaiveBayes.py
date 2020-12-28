# Name: Venkata Vadrevu
# FSU id: vv18d
# CIS 4930

import sys
import os
import math 

class NaiveBayes:
    def __init__(self):
        ''' Default constructor'''


        # A dictionary that stores the names of different class_labels along with their count
        self.classCount = dict()

        # A dictionary that stores the label along with count for each label value
        self.labelDict = dict()

        # A dictionary that stores the names of different class_labels and maps them to its labelDict
        self.classDict = dict()

        # labels is a set that holds all the labels
        self.labels = set()

        # A variable that stores the size of the model
        self.size = None

        # A variable to know whether teh model was trained or not
        self.trained = False

        return


    def train(self, filename):
        '''We train the model using a file '''
        
        # Parsing format
        # classLabel index1:value1 index2:value2  . . . 

        # Set size to zero
        self.size = 0

        # We use a three pass algorithm to colelct all the data
        # We read the file three times

        # In the first pass we get all the different Labels and count of each classLabel ('+1', '-1')
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip('\n').split(' ')
                if len(items) > 0:
                    self.size += 1

                # Class label is the first item
                classLabel = items[0]                

                # We get the count of each classLabel
                if classLabel not in self.classCount:
                    self.classCount[classLabel] = 1
                else:
                    self.classCount[classLabel] += 1


                # We add the labels to the set called "self.labels" to identify all the different labels
                items = items[1:]
                for item in items:
                    label, value = item.split(':')
                    if label not in self.labels:
                        self.labels.add(label)


        # Second pass
        # We populate the labelDict        
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip('\n').split(' ')

                # class Label is the first item
                classLabel = items[0]

                items = items[1:]

                # we make a dictionary where each key is a label  and its value is the number in the file
                # "label":"value" 
                instance = dict()
                for item in items:
                    label, value = item.split(':')
                    instance[label] = value


                # For we iterate throught each label in the set self.labels 
                for label in self.labels:

                    # if the label is not in instance then its value is set to 0
                    if label not in instance:
                        value = '0'
                    else:
                        value = instance[label]

                    # if the label is not in self.labelDict we insitialize it to a dict
                    if label not in self.labelDict:
                        self.labelDict[label] = dict()

                    # Each label can have differnet values
                    # We get the count of each value ie.. count the number of times each value happened
                    # for a particular label
                    if value not in self.labelDict[label]:
                        self.labelDict[label][value] = 1
                    else:
                        (self.labelDict[label])[value] += 1


        # Initialize each class in the self.classDict to an empty dictionary
        for Class in self.classCount:
            self.classDict[Class] = dict()

        # Last pass        
        with open(filename, 'r') as f:
            for line in f:
                items = line.strip('\n').split(' ')

                classLabel = items[0]

                items = items[1:]

                instance = dict()
                
                for item in items:
                    label, value = item.split(":")
                    instance[label] = value 


                # We do the same as in the label dict but we do the same for each classLabel
                for label in self.labels:
                    if label not in instance:
                        value = '0'
                    else:
                        value = instance[label]

                    if label not in self.classDict[classLabel]:
                        self.classDict[classLabel][label] = dict()

                    if value not in self.classDict[classLabel][label]:
                        self.classDict[classLabel][label][value] = 1
                    else:
                        self.classDict[classLabel][label][value] += 1

        # We set trained to True
        self.trained = True
        return

    
    def testFile(self, filename):
        '''Predicts that truPostive, falseNegative, trueNegative and falsePostive for a testfile'''
        
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0


        with open(filename, 'r') as f:
            for line in f:
                items = line.strip('\n').split(' ')
                # The true label is the class Label
                # We store it in a variable to test out prediction
                trueLabel = items[0]
                items = items[1:]

                # Create a dictionary that stores the instance 
                Instance = dict()
                for item in items:
                    label, value = item.split(':')
                    Instance[label] = value 

                # if a label is not in the instance its value should map to '0'
                for label in self.labels:
                    if label not in Instance:
                        Instance[label] = '0'
            

                # We predict the instance using the self.predict funciton
                predictedLabel = self.predict(Instance)
                
                # We increment the counter based on the validity of the prediction

    
                if predictedLabel == "+1" and trueLabel == "+1":
                   truePositive += 1
                elif predictedLabel == "-1" and trueLabel == "-1":
                    trueNegative += 1
                elif predictedLabel == "-1" and trueLabel == "+1":
                    falseNegative += 1
                else:
                    falsePositive += 13
                
        
        # We print the values to the screen
        print(truePositive, falseNegative, falsePositive, trueNegative)
        return


    def predict(self, instance):
        '''Predicts the class label based on the trained model'''

        # THE MATH
        #P(C_k| x_1, x_2, x_3 ....) = P(C_k)*P(x_1, x_2, x_3 ... | C_k)/P(x_1, x_2, ....)
        #                      proportional   P(C_k)*P(x_1|C_k)*P(x_2|C_k)........
        #                                     P(C_k)*product ( P(x_i | C_k) )
        #                                   log(P(C_k)) + sum( log(P(x_i | C_k)) )
        # P(C_k) = #(C_k)/ #Total
        # P(x_i & C_k) = # x_i & C_k / #Total
        # P(x_i | C_k) = #(x_i & C_k)/ #C_k


        # If we have not trained the model yet we will not compute
        if not self.trained:
            return None

        # dictionary to store the negative log of the probaility of each class label
        # classLabel: value
        Neg_Log_probabilities = dict()

        # We calculate the probaility for each class label
        for classLabel in self.classDict:
            Log_probability = 0

            # Probability for the class label
            # P(C_k) = #(C_k)/#TotalSize
            P_ClassLabel = (self.classCount[classLabel]*1.0)/self.size 

            Log_probability += math.log(P_ClassLabel)

            # We caluclate the probablilty of the label having a value given that it belong to a class
            # ie ... P(label_i=value_i | class = C_k)

            for label in instance:
                value = instance[label]
                
                # If we find the value in the training instance 
                # We caluculate the probability
                if value in self.classDict[classLabel][label]:

                    # P(label_i=value_i | class = C_k) = P(label_i = value_i & class = C_k)/ P(class = C_k)
                    #
                    #                                  = #(label_i = value_i & class = C_k)/ #(class = C_k)
                    #                                  = classDict[classLabel]][label][value]/classCount[classLabel]
                    
                    P_Labelval_class = (self.classDict[classLabel][label][value]*1.0)/(self.classCount[classLabel])
                    
                    # We take the log of that value to avoid underflow erros
                    Log_probability += math.log(P_Labelval_class)
                
                else:
                    # If we do not find any value the porbaility is 0
                    # therefore the log of the value is -infinity
                    Log_probability = float('-inf')
                    break
            
            # We take the negative value of the log of the prbability
            # and choose the minimum value
            Neg_Log_probabilities[classLabel] = -1*Log_probability

        return min(Neg_Log_probabilities.keys(), key = (lambda k: Neg_Log_probabilities[k]))





def main(Trainfile, Testfile):
    # We first create an object from the NaiveBayes class
    Model = NaiveBayes()

    # We train it on the trinfile
    Model.train(Trainfile)

    # We test it on the trainfile
    Model.testFile(Trainfile)

    # We test it on the testfile
    Model.testFile(Testfile)



if __name__ == "__main__":
    if len(sys.argv) == 3:
        TrainFile = sys.argv[1]
        TestFile = sys.argv[2]

        # Check for validity of files
        try:
            open(TrainFile, "r")
        except (FileNotFoundError, IOError):
            # If we have an error then we print error statement and exit
            print("{} not Found. Please enter a valid file.".format(TrainFile))
            exit(0)

    
        try:
            open(TestFile, "r")
        except (FileNotFoundError, IOError):
            print("{} not Found. Please enter a valid file.".format(TestFile))
            exit(0)

        # call the main function
        main(TrainFile, TestFile)

    else:
        print("Error!!!\nformat: python3 NaiveBayes.py Trainfile Testfile")
