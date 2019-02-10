Note: in each of the ZIP files, car_code_dataset and phishing_code_dataset, there is the csv file that will need to be in the same folder as any of the code files that is running. 
The car evaluation dataset is in car_evaluation.csv
The phishing dataset is in dataset.csv

Running Code
---------------------------------------------------------------
python phishing_dtree.py /  python car_dtree.py  (Decision Tree)
python phishing_boost.py /  python car_boost.py  (Boosted Decision Tree)
python phishing_svm.py   /  python car_svm.py    (SVM)
python phishing_nn.py    /  python car_nn.py     (Neural Net)
python phishing_knn.py   /  python car_knn.py    (KNN)

Graphs plotted every file and saved, not shown.
Console output: confusion matrix and classification report along with artibrary accuracy/error metrics used for tuning. 