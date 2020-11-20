import math
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sm_mod
import statsmodels.graphics.gofplots as plots
import matplotlib.pyplot as plt
import statsmodels.tools.tools as smtools
import statsmodels.genmod.families as sm_fam
from pygam import LogisticGAM, GAM, s, te
import sklearn.metrics as skm
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

# Problem #1
data = pd.read_csv("https://donatello-telesca.squarespace.com/s/diabetes.csv")
data = data.dropna()

data['diagnosis'] = [1 if x > 7 else 0 for x in data.glyhb]
data['BMI'] = (data.weight/data.height**2)*703
data['wh_ratio'] = data.waist/data.hip
data['gender'] = [1 if x == 'female' else 0 for x in data.gender]
data['genderxBMI'] = data.gender*data.BMI
data['genderxwh_ratio'] = data.gender*data.wh_ratio
data = smtools.add_constant(data)

# Fit separate logistic regression models per variable
y = data.diagnosis
mod_params = []

mod1 = sm_mod.Logit(y,data[['const', 'age']])
mod1_results = mod1.fit()
print(mod1_results.summary())
mod_params.extend([mod1.exog_names,mod1_results.params,mod1_results.pvalues])

mod2 = sm_mod.Logit(y,data[['const','gender']])
mod2_results = mod2.fit()
print(mod2_results.summary())
mod_params.extend([mod2.exog_names,mod2_results.params,mod2_results.pvalues])

mod3 = sm_mod.Logit(y,data[['const','BMI']])
mod3_results = mod3.fit()
print(mod3_results.summary())
mod_params.extend([mod3.exog_names,mod3_results.params,mod3_results.pvalues])

mod4 = sm_mod.Logit(y,data[['const','wh_ratio']])
mod4_results = mod4.fit()
print(mod4_results.summary())
mod_params.extend([mod4.exog_names,mod4_results.params,mod4_results.pvalues])

# Fit logistic regression model with all covariates
log_mod = sm_mod.Logit(y,data[['const','age','gender','BMI','wh_ratio']])
logmod_results = log_mod.fit()
print(logmod_results.summary())

log_mod2 = sm_mod.Logit(y,data[['const','age','gender','BMI','wh_ratio','genderxBMI','genderxwh_ratio']])
# log_mod2 = sm_mod.Logit(y,data[['const','age','gender','BMI','wh_ratio','genderxBMI']])
# log_mod2 = sm_mod.Logit(y,data[['const','age','gender','BMI','wh_ratio','genderxwh_ratio']])
logmod2_results = log_mod2.fit()
print(logmod2_results.summary())

# Fit the predicted values of log_mod
predicted = log_mod.predict(logmod_results.params.values)
data['prob_predicted'] = predicted
for i in range(len(predicted)):
	predicted[i] = math.log(predicted[i]/(1-predicted[i]))
data['lin_predicted'] = predicted

# fig, axs = plt.subplots(2,2)
# axs[0,0].scatter(data.age,data.lin_predicted,label='age')
# axs[0,0].set_title('Age vs. Predicted')
# axs[0,1].scatter(data.BMI,data.lin_predicted,label='BMI')
# axs[0,1].set_title('BMI vs. Predicted')
# axs[1,0].scatter(data.wh_ratio,data.lin_predicted,label='wh_ratio')
# axs[1,0].set_title('Waist-Hip Ratio vs. Predicted')
# plt.show()

# Use GAM to test linearity assumption
# gam = LogisticGAM(terms='auto').fit(data[['age','gender','BMI','wh_ratio']],data.diagnosis)
# # plt.rcParams['figure.figsize'] = (28,8)
# fig, axs = plt.subplots(1,4)
# titles = ['age','gender','BMI','wh_ratio']
# for i, ax in enumerate(axs):
# 	XX = gam.generate_X_grid(term=i)
# 	ax.plot(XX[:,i], gam.partial_dependence(term=i,X=XX))
# 	ax.plot(XX[:,i], gam.partial_dependence(term=i, X=XX, width=0.95)[1],c='r',ls='--')
# 	ax.set_title(titles[i])
# plt.show()

# Creating test and training data for single predictors
X_train1, X_test1, y_train1, y_test1 = train_test_split(data[['const','BMI']],y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data[['const','wh_ratio']],y)
# Creating test and training data for full models
X_train3, X_test3, y_train3, y_test3 = train_test_split(data[['const','age','gender','BMI','wh_ratio']],y)
X_train4, X_test4, y_train4, y_test4 = train_test_split(data[['const','age','gender','BMI','wh_ratio', 'genderxBMI','genderxwh_ratio']],y)

# Fitting logit models to training data on individual predictors
rocmod1 = sm_mod.Logit(y_train1, X_train1)
rocmod1_results = rocmod1.fit()
ypreds1 = rocmod1_results.predict()
rocmod2 = sm_mod.Logit(y_train2, X_train2)
rocmod2_results = rocmod2.fit()
# Fitting logit models to training data on full data
rocmod3 = sm_mod.Logit(y_train3, X_train3)
rocmod3_results = rocmod3.fit()
rocmod4 = sm_mod.Logit(y_train4, X_train4)
rocmod4_results = rocmod4.fit()

# Generating the predicted probabilities for single predictors
preds1 = rocmod1_results.predict(X_test1)
preds2 = rocmod2_results.predict(X_test2)
preds3 = rocmod3_results.predict(X_test3)
preds4 = rocmod4_results.predict(X_test4)

# Calculate each model's FPR and TPR
fpr1, tpr1, threshold1 = skm.roc_curve(y_test1,preds1)
fpr2, tpr2, threshold2 = skm.roc_curve(y_test2,preds2)
fpr3, tpr3, threshold3 = skm.roc_curve(y_test3,preds3)
fpr4, tpr4, threshold4 = skm.roc_curve(y_test4,preds4)

no_skill_probs = [0 for _ in range(len(y_test1))]
no_skill_auc = roc_auc_score(y_test1, no_skill_probs)

auc1 = roc_auc_score(y_test1, preds1)
ns_fpr1, ns_tpr1, _ = roc_curve(y_test1, no_skill_probs)
fpr1, tpr1, _ = roc_curve(y_test1, preds1)
# plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# plt.plot(fpr1, tpr1, marker='.', label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

auc2 = roc_auc_score(y_test2, preds2)
fpr2, tpr2, _ = roc_curve(y_test2, preds2)
# plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# plt.plot(fpr2, tpr2, marker='.', label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

auc3 = roc_auc_score(y_test3, preds3)
fpr3, tpr3, _ = roc_curve(y_test3, preds3)
# plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# plt.plot(fpr3, tpr3, marker='.', label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

auc4 = roc_auc_score(y_test4, preds4)
fpr4, tpr4, _ = roc_curve(y_test4, preds4)
# plt.plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# plt.plot(fpr4, tpr4, marker='.', label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

# fig, axs = plt.subplots(2,2)
# axs[0,0].plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# axs[0,0].plot(fpr1, tpr1, marker='.', label='Logistic Regression')
# axs[0,0].set_title('Y ~ BMI')
# axs[0,1].plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# axs[0,1].plot(fpr2, tpr2, marker='.', label='Logistic Regression')
# axs[0,1].set_title('Y ~ wh_ratio')
# axs[1,0].plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# axs[1,0].plot(fpr3, tpr3, marker='.', label='Logistic Regression')
# axs[1,0].set_title('Y ~ age + gender + BMI + wh_ratio')
# axs[1,1].plot(ns_fpr1, ns_tpr1, linestyle='--', label='No Skill')
# axs[1,1].plot(fpr4, tpr4, marker='.', label='Logistic Regression')
# axs[1,1].set_title('Y ~ age + gender + BMI + wh_ratio + genderxBMI + genderxwh_ratio')
# plt.show()

# Problem #2
data2 = pd.read_csv('https://donatello-telesca.squarespace.com/s/medpar.csv')
data2 = smtools.add_constant(data2)
data2['type2xage80'] = data2.type2*data2.age80
data2['type3xage80'] = data2.type3*data2.age80
pmod = sm_mod.Poisson(data2.los,data2[['const','type2','type3']])
pmod_results = pmod.fit()
print(pmod_results.summary())

pmod2 = sm_mod.Poisson(data2.los,data2[['const','type2','type3','age80']])
pmod2_results = pmod2.fit()
print(pmod2_results.summary())

pmod3 = sm_mod.Poisson(data2.los,data2[['const','type2','type3','age80','type2xage80','type3xage80']])
pmod3_results = pmod3.fit()
print(pmod3_results.summary())

# Generate diagnostic plots for poisson
# Fitted vs. pearson residual
X_train2, X_test2, y_train2, y_test2 = train_test_split(data2[['const', 'type2','type3','age80']],data2.los)
pmod_cv = sm_mod.Poisson(y_train2,X_train2)
pmodcv_results = pmod_cv.fit()
fitted = pmodcv_results.predict(X_test2,linear=False)

train = pd.concat([X_train2,y_train2],axis=1)
train_data = pd.DataFrame(train)
glm = statsmodels.formula.api.gee
model = glm("los ~ type2 + type3 + age80", groups=data2.los, data=data2, family=sm_fam.Poisson())
model_results = model.fit()
model_fitted = model_results.fittedvalues
model_residuals = pd.Series(model_results.resid_pearson)

# Fitted vs residuals plot
plt.scatter(model_fitted,model_residuals)
plt.xlabel('Fitted')
plt.ylabel('Pearson Residuals')
plt.plot(model_fitted,np.zeros([1495,]))
plt.show()

plt.hist(model_residuals)
plt.show()

# QQ plot
sm.qqplot(model_results.resid_deviance,stats.poisson(),line='45')
plt.show()
