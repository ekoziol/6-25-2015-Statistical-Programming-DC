**Machine Learning in R**

This is the code for a talk that I initially gave at the Statisical Programming DC meetup on 6/25/2015.It is a tutorial on how to use the caret and h2o packages.  The data used is the Pima Indians Diabetes data set from the UCI Machine Learning Repository which can be found here:

https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

The code is you will want to view is ml-compare.R.  I have added a variable so that you can try out different random seeds.  To see caret models just type the name after training.  AUC curves are automatically plotted after running the roc() command.  If you save the roc() function to a variable, just type the variable name.

Towards the end of the code I have added a quick comparison plot of different AUC curves. 

I hope this gets you started on your machine learning journey!