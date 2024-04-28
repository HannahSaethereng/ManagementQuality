# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:29:06 2023

@author: hs0420
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import wooldridge as woo
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.api as sm 


df1 = pd.read_excel("C:/Users/hs0420/OneDrive - Westminster University/ECON 499/Project/management_quality.xlsx")
df2 = df1.loc[df1["country"]== "Sweden",:]
df=df2.loc[:,["firmid", "wave", "cty", "country", "sic", "management",
              "operations", "monitor","target", "people","emp_firm", 
              "firmage","ownership", "mne_yn","mne_cty", "competition",
              "export", "degree_m", "degree_nm", "degree_t",
              "i_seniority", "reliability"]]

df["ownership"].value_counts()

list_ownership=["Dispersed Shareholders","Private Individuals",
"Family owned, family CEO","Other","Founder owned, founder CEO",
"Private Equity/Venture Capital","Family owned, external CEO",
"Founder owned, external CEO","Government"]

df[list_ownership]=pd.get_dummies(df["ownership"])

df=df.rename({"Family owned, family CEO": "family", "Family owned, external CEO": "family_ext_ceo"},axis=1)

df["family"].value_counts()  
df["family_ext_ceo"].value_counts()
 

df["competition"].value_counts()

#high competition dummy
df["compet_dummy"]=np.where(df["competition"]=="10+ competitors",1,0)
df["compet_dummy"].value_counts()



reg=smf.ols(formula="management~C(compet_dummy)+C(mne_yn)+"
             "emp_firm+firmage+C(family)+C(family_ext_ceo)",data=df).fit()

sns.histplot(data=df,x="emp_firm",bins=30)

#making employment log
df["lemp"]=np.log(df["emp_firm"])


reg=smf.ols(formula="management~C(compet_dummy)+C(mne_yn)+"
            "lemp+firmage+C(family)+C(family_ext_ceo)",data=df).fit()

#[T.1.0] means T stands for treatment. treatment is 1 and multinational, 
#the controll group are the once without multinationality
#if t says zero it result is relative to the controll group
reg.summary()

#Everything before thjis is from The python for management_quality_data





df.shape
df.info()
reg1=smf.ols(formula="management~degree_m",data=df).fit()
reg2=smf.ols(formula="management~degree_m+I(degree_m**2)",data=df).fit()
reg1.summary()
reg2.summary()


sns.kdeplot(data=df, x="management")
ax = sns.catplot(data=df, x="competition", y="management", kind="box")
ax.set_ylim(0, 150) #error
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha="right")
sns.kdeplot(data=df, x="management")
ax = sns.catplot(data=df, x="competition", y="management", kind="box")
ax.set(ylim=(1, 5))  # Set y-axis limits directly on FacetGrid
ax.set_xticklabels(rotation=45) 
plt.show()
df["management"].mean()



sns.histplot(data=df,x="emp_firm",bins=30)
#making employment log
df["lemp"]=np.log(df["emp_firm"])
sns.histplot(data=df,x="lemp",bins=30)


df.isna().sum()

df3 = df.loc[:,['management','emp_firm','firmage','compet_dummy','lemp']]
sns.pairplot(df3)
sns.lmplot(x="lemp", y="management", data=df3, ci=False)



df=df.rename({"Family owned, family CEO": "family", 
                "Family owned, external CEO": "family_ext_ceo",
                "Dispersed Shareholders": "dis_sh",
                "Private Individuals": "pvt_ind",
                "Other":"other",
                "Founder owned, founder CEO": "founder_owned_ceo",
                "Private Equity/Venture Capital": "pvt_eq_vent_cap",
                "Founder owned, CEO unknown": "founder_ceo_unkn",
                "Family owned, CEO unknown": "family_ceo_unkn",
                "Founder owned, external CEO": "founder_ext_ceo",
                "Government": "gov"},axis=1)

df.to_csv("management_US_data_python_edited.csv")

fig,ax=plt.subplots()
ax = sns.boxplot(x=df["ownership"], y=df["management"])
ax.set_xticklabels(df['ownership'],rotation=45)

fig, ax = plt.subplots()
sns.boxplot(x=df["ownership"], y=df["management"], ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Adjust rotation and alignment

plt.show()

fig,ax=plt.subplots()
ax = sns.boxplot(x=df["compet_dummy"], y=df["management"])

fig,ax=plt.subplots()
ax = sns.boxplot(x=df["dis_sh"], y=df["management"])


sns.lmplot(data=df, x="lemp", y="management", hue="mne_yn")

sns.lmplot(data=df, x="lemp", y="management", hue="compet_dummy")
sns.lmplot(data=df, x="lemp", y="management", hue="ownership")

sns.lmplot(data=df, x="lemp", y="management", row="mne_yn")
sns.lmplot(data=df, x="lemp", y="management", row="dis_sh") #does not work

fig, ax = plt.subplots(figsize=(5, 2.7),layout='constrained')
ax.scatter(df['lemp'],df['management'])
ax.set_xlabel('Employment size (logs)')  # Add an x-label to the axes.
ax.set_ylabel('Management quality')  # Add a y-label to the axes.
ax.set_title("Plot of Management quality against employment size\n US data")  # Add a title to the axes.

df['lfirmage']=np.log(df['firmage']+0.001)

df2=df.loc[:,['management', 'lemp', 'lfirmage', 'mne_yn']]
df3=df.loc[:,['management', 'degree_m','i_seniority', 'reliability', 'compet_dummy']]
sns.pairplot(df2)
sns.pairplot(df3)

#this is the main regression
reg=smf.ols(formula='management~ np.log(emp_firm)+firmage+ownership+mne_yn+'
            'export+degree_m+degree_nm+degree_t+i_seniority+reliability+'
            'compet_dummy',data=df).fit()

reg.summary() 

table_anova=sm.stats.anova_lm(reg,type=2)
table_anova
#This is the results from the main regression
reg.summary(xname=["Intercept",
                   "Family owned, external CEO",
                   "Family owned, family CEO",
                   "Founder owned, external CEO",
                   "Founder owned, founder CEO",
                   "Government",	"Other", 
                   "Private Equity/Venture Capital",
                   "Private Individuals",
                   "Log (employment)",	"Firm age in years",
                   "Multinational", "Exports",
                   "Managers with college degree(%)",
                   "Non-Managers with college degree(%)",
                   "All workers with college degree(%)",
                   "Manager's seniority in company",
                   "Reliability measure","Competition"])
fig, ax = plt.subplots()
ax.scatter(reg.fittedvalues, reg.resid)
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.axhline(y = 0, color = 'r')
fig.show() 

name = ["Lagrange multiplier statistic", "LM p-value", "F-statistic", "F p-value"]
test = sms.het_breuschpagan(reg.resid, reg.model.exog)
lzip(name, test)
#we fail to reject the null of hetroskedasticity
studres=reg.get_influence().resid_studentized_external
fig,ax=plt.subplots()
sns.displot(x=studres,kde=True)

leverage = reg.get_influence().hat_matrix_diag
fig, ax = plt.subplots()
sns.displot(x=leverage,kde=True)

sm.graphics.influence_plot(reg)

df4=df.loc[:,[ "management", "emp_firm","firmage","i_seniority"]]
df4=df.loc[:,[ "export", "degree_m", "degree_nm", "degree_t"]]

df4.describe()
description
