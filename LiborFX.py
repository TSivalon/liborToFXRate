import pandas_datareader as wb
import pandas as pd
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

trainStartDate = datetime(2010,1,1)
trainEndDate = datetime(2019,11,1)

# importing US, EUR LIBORS and translation rates
usLibor = wb.DataReader("USDONTD156N", "fred", trainStartDate, trainEndDate)
eurLibor = wb.DataReader("EURONTD156N", "fred", trainStartDate, trainEndDate)
XRate = wb.DataReader("DEXUSEU", "fred", trainStartDate, trainEndDate)

# renaming columns so that they make more sense
usLibor.columns=["US Libor"]
eurLibor.columns=["EUR Libor"]
XRate.columns=["Exchange Rate"]

# joining it all together
join = usLibor.join(eurLibor, how="inner").join(XRate, how="inner")

# new column for the absolute rate differential, then converged into a percentage, exchange rate percentage
join["Rate Diff"] = abs(join["US Libor"] - join["EUR Libor"])
join["Rate Diff Pct"] = join["Rate Diff"].pct_change()
join["FX Rate Pct"] = join["Exchange Rate"].pct_change()

# remove all the negative infinities, positive infinities, get rid of everything that doesn't have a value
join = join.replace([np.inf,-np.inf], np.nan)
join = join.dropna()

# inputs and results, used for the regression
joinInputs = join[['Rate Diff Pct']]
joinResults = join[['FX Rate Pct']]

# linear regression model
linRegr = linear_model.LinearRegression()

# naming all variables with nifty train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(joinInputs, joinResults, test_size=0.3,random_state=1)

linRegr.fit(Xtrain,Ytrain)
Ypredsk = linRegr.predict(Xtest)
r2easy = r2_score(Ytest,Ypredsk)

# time to display everything
print("Coefficients m:", linRegr.coef_)
print("Intercept b:", linRegr.intercept_)

print("r2 score for our model is: ", r2easy)

print("Based on the above assumptions, I will infer that there is no relationship between the LIBOR rates in conjunction with \n exchange rates. The r2 shows zero correlation after performing a linear regression.")
join.plot()
plt.show()


# print(join.to_excel("DataFrame.xlsx"))