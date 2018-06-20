from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO	
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

# [height, weight, shoe_size]
Data = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Gender = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#creating the classifier variable based on data
gender_dtree = DecisionTreeClassifier()
gender_dtree = gender_dtree.fit(Data,Gender)

#read/write string as file
dot_data = StringIO()

#generate graphviz representation
export_graphviz(gender_dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True)

#creating graph with pydot
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())	

#If using IPython notebook
#Image(graph.create_png())

#For file output
graph.write_png("tree.png")