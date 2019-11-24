* Final SAS project for STAT202A;
* Import car data;
proc import out= work.data
datafile= "/folders/myfolders/study1/regression_auto.csv"
dbms=csv replace; getnames=yes; datarow=2;
run;

* Compute the correlation between car length and mpg;
proc corr  data=data;
var length mpg;
run;
* Make a scatterplot of price (x-axis) and mpg (y-axis);
proc sgplot data=data;
scatter x = price y = mpg;
run;
proc print data=data;
run;
* Make a box plot of mpg for foreign vs domestic cars;
proc boxplot data = data;
plot mpg*foreign;
run;
* Perform simple linear regression, y = mpg, x = price1; 
* Do NOT include the intercept term;
proc reg data=data;
model mpg = price1 / noint;
run;

* Perform linear regression, y = mpg, x1 = length, x2 = length^2; 
* Include the intercept term;
DATA data1;
SET data;
length_sq = length*length;
run;

proc reg data=data1;
model mpg = length length_sq;
run;
