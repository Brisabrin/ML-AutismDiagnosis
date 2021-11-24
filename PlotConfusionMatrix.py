import seaborn as sns 
import matplotlib.pyplot as plt 

#algorithm - #subject #n - fold #n
file_name = "DecisionTreeconfusionmatrixplotsubject_Adult_fold_1.png"


#tp | fn 
#fp | tn  
tn, fp, fn, tp = (12, 6, 8, 25)

confusion_matrix_data = [[tp, fn ], [ fp , tn ]]

sns.heatmap(confusion_matrix_data, annot=True, cmap='Blues', fmt='d').get_figure().savefig(file_name) 

