# Statistics-for-Data-Science-with-Python
import piplite
await piplite.install(['numpy'],['pandas'])
await piplite.install(['seaborn'])

import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
resp = await fetch(URL)
boston_url = io.BytesIO((await resp.arrayBuffer()).to_py())

boston_df=pd.read_csv(boston_url)

print(boston_df)

#Boxplot of "Median value of owner-occupied homes"
x=sns.boxplot(data=boston_df,y='MEDV')
x.set_title("Median value of owner-occupied homes")

#Bar plot for the Charles river variable
x=sns.barplot(data=boston_df,x='CHAS',y='MEDV')
x.set_xticklabels(['Far_from_river', 'Near_river'])
x.set_title("Median value of owner-occupied homes (in $1000's)- by distance from river")
x.bar_label(x.containers[0],fmt='%.2f',padding=10)

#Boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)
#first i created the 3 required groups
boston_df.loc[(boston_df['AGE']<=35,'age_group')]='35 and less'
boston_df.loc[(boston_df['AGE']>35)&(boston_df['AGE']<70),'age_group']='36 to 70'
boston_df.loc[(boston_df['AGE']>=70,'age_group')]='70 and above'

#order variable will allow me to reorder the barplot by age
order=['35 and less','36 to 70','70 and above']

#create the bar plot
x=sns.barplot(data=boston_df,x='age_group',y='MEDV',order=order)
x.set_title("Median value of owner-occupied homes (in $1000's)- by AGE")
x.bar_label(x.containers[0],fmt='%.2f',padding=10)

#Scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?

sns.scatterplot(data=boston_df,x='NOX',y='INDUS')

#There is quite strong positive relationship between these 2 values (high NOX correlate with high INDUS, the same for low values)

#Histogram for the pupil to teacher ratio variable
sns.histplot(data=boston_df,x='PTRATIO')

#Task 3: Statistical Tests
#Use α = 0.05

#Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

print(f"Statistical info on the two groups:\n{boston_df.groupby('CHAS')['MEDV'].describe().round(2)}")
print("CHAS=0- houese far from the river,\nCHAS=1- houses near the river")

#Levene test to check if the 2 group's variance are equal
print("""\nLevene test's hypothesis:
Null hypothesis: The median values of houses' variance of houses near the Charles river and houses far from the river are equal
Alternative hypothesis: The median values of houses' variance of the two groups are different\n""")

lev_stat,lev_p=scipy.stats.levene(boston_df[boston_df['CHAS']==0]['MEDV'],boston_df[boston_df['CHAS']==1]['MEDV'])
#pvalue=0.032- we reject the null hypothesis, the variances are different
print(f"Levene-test result: p-value is {round(lev_p,4)}, we reject the null hypothesis\n")

#t-test to check if the difference between the two groups in MEDV is significant
print("""t-test hypothesis:
Null hypothesis: The difference between houses near the Charles river and houses far from the river in MEDV isn't significant
Alternative hypothesis: There is significant difference between houses near the Charles river and houses far from the river in MEDV""")

stat,p_val=scipy.stats.ttest_ind(boston_df[boston_df['CHAS']==0]['MEDV'],boston_df[boston_df['CHAS']==1]['MEDV'],equal_var=False)
#pvalue=0.0036- we reject the null hypothesis, the difference in MEDV is significant
print(f"\nt-test result: p-value is {round(p_val,4)}, we reject the null hypothesis")

#Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
#statistical info on the 3 AGE groups
print(f"Statistical info on the 3 AGE groups:\n{boston_df.groupby('age_group')['MEDV'].describe()}\n")

#create 3 variables that contains the MEDV's values of the age groups
low_group=boston_df.loc[(boston_df['age_group']=='35 and less')]['MEDV']
middle_group=boston_df.loc[(boston_df['age_group']=='36 to 70')]['MEDV']
top_group=boston_df.loc[(boston_df['age_group']=='70 and above')]['MEDV']

#ANOVA test to check if the difference between the AGE groups in MEDV is significant
print("""ANOVA test hypothesis:
Null hypothesis: Samples in all AGE groups are from populations with the same mean values 
Alternative hypothesis: The means of the populations are not the same""")

f_stat,f_pval=scipy.stats.f_oneway(low_group,middle_group,top_group)
#pvalue=1.7105011022702984e-15 - we reject the null hypothesis, the means of the groups are not the same
print(f"\nANOVA test result: p-value is {f_pval}, we reject the null hypothesis")

#Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

#Pearson correlation test to check if their is a relationship between NOX and INDUS and its attributes
print("""Pearson correlation hypothesis:
Null hypothesis: There is no correlation between Nitric oxide concentrations and proportion of non-retail business acres per town
Alternative hypothesis: There is a correlation between Nitric oxide concentrations and proportion of non-retail business acres per town""")

pear_stat,pear_pval=scipy.stats.pearsonr(boston_df['NOX'],boston_df['INDUS'])
print(f"\nPearson correlation result: pvalue={pear_pval}, statistic= {round(pear_stat,3)}. \nWe reject the null hypothesis- there is strong positive correlation between the 2 variables")

#What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

#Regression analysis to check the impact (connection between these two variables)
x=boston_df['DIS'] #Independent variable
y=boston_df['MEDV'] #Dependent variable

x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
predictions=model.predict(x)

print(model.summary())

print("\nWe reject the null hypothesis, p-value equal to 0")

