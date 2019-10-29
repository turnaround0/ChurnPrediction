# ChurnPrediction
## Environment
This program was tested on Anaconda3 / python 3.7 environment.<br>
Please install required python libraries.
``` console
$ pip install -r requirements.txt
```

## How to execute
### Example of test for all experiments with full dataset.
``` console
$ python churn.py -s full
```
-s option gives type of dataset. (tiny, small, full)<br>
path: dataset/tiny/, dataset/small/, dataset/full/ <br>
In case of full dataset,<br>
you can use users_reduce.pkl and posts_reduce.pkl.<br>
If you want to use them, please put them on dataset/full path.


After reading xml files, this program will convert them to pkl files.<br>
Result: Users.pkl, Posts.pkl<br>
If those pkl files exist, this program will skip reading xml files.

### Store features
After extracting features, they will be stored on output/features folder.<br>
If using -r option, you can skip the process to extract features.
``` console
$ python churn.py -s full -r
```

### Store plots and tables
All plots and tables will be saved on output/ folder.<br>
If you want to see those plots immediately, you can use -d option.
``` console
$ python churn.py -s full -d
```
