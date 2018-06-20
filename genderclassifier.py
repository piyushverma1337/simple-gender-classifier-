#imports
from sklearn.tree import DecisionTreeClassifier

# [height, weight, shoe_size]
Data = [[159, 55, 37], [171, 75, 42], [181, 85, 43], [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40]]

Gender = ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female']

#creating the classifier variable
#updating variable by training using fit 
gender_dtree = DecisionTreeClassifier().fit(Data,Gender)

#user input
height = input("Enter Height(cm): ")
weight = input("Enter Weight(kg): ")
shoe_size = input("Enter Shoe_size(Eur): ")


#predicting from the model
prediction = gender_dtree.predict([[height,weight,shoe_size]])

#output
print("Predicted Gender: ", prediction[0])