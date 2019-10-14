---
layout: post
title: Predicting Tomorrows Gold Price
subtitle: Do we have a market advantage?
image: "/img/GoldPrices/GoldBars.png"
tags: [Regression, Time Series, Forecasting, Naive Solution, Stock Market, Commodities Market, Stock Forecasting]
comments: true
---

### Introduction
Central banks across the world maintain gold reserves to guarantee the money of their depositors, foreign-debt creditors, and currency holders.  Central banks also use gold reserves as a means to control inflation and strengthen their country’s financial standing. In addition, gold is used by finance companies, global institutions, money managers and individual investors to hedge against inflation and to diversify their portfolios.  As a precious metal, gold is popular for jewelry and ornamentation.

Given the standing of gold’s popularity in the contemporary world, forecasting its price is a widely explored topic and interesting to multiple global institutions and small & large scale investors. It is also a complex problem given gold’s price fluctuations are not entirely based on supply and demand, but also depends on a multitude of geo-political and financial factors.

<hr>

### Methodology
The approach used for this analysis has three steps: data collection/assembly step, model building step, and an evaluation step.

![Methodology](/img/GoldPrices/Methodology.png "Methodology")

#### Data Collection and Assembly
The independent variables considered for this analysis include <i>economic and market factors</i> like S&P 500, Dow Jones Industrial average, FTSE 100 Index, treasury bond rates, bank interest rates, US Dollar Index; and other <i>commodity indicators</i> like silver, platinum, palladium and crude oil prices. In addition, <i>Gold Futures</i> were also considered to reflect market expectation of future gold prices. A number of studies reviewed as part of this work have considered similar variables, with the exception of Gold Futures. In the interest of containg the scope of this project, the international economic and market indicators from gold supplying and consuming countries, and variables indicating the seasonality in gold buying and selling, espcecially as it relates to some asian consumer markets, were not considered for this analysis.  The data colleted for this project was the daily closing market price of these variables for the years 2000 to 2018. Other daily <i>market indications</i> like opening price, maximum, minumum, and trading volume were not used in this analysis, but might be worthy for future work.

The dependent variable, price of gold, was collected from kitco.com website and is based on the <i>London Gold Fix</i> benchmark.  This benchmark price of gold is set by a group of participating international banks twice every business day in London as the name indicates, and is used for pricing majority of gold derivatives and products throughout the world markets. The highest of the two rates set each day was used for this analysis. The gold prices used for this project are shown below:

![Gold Price](/img/GoldPrices/GoldPriceActual2.png "Gold Prices: 2000 to 2019")

 The database was assembled from multiple soruces as lised below, and stitched together using dates as the merging attribute.   Once the data were "stitched" together, US holidays were removed from the analysis. In addition, any missing or unreasonable data tagged during an Exploratory Data Analysis step was imputed using the prices from the previous day (forward filling).

#### Modeling
As part of the model builing exercise, linear regression and time series methods were used as shown below.

![ModelMehtods](/img/GoldPrices/ModelMethods.png "Methods")

The model building process had gone through the following steps, where each step went through multiple iterations as needed:
1. Use daily prices from 2000 to 2015 to train linear regression models. Dependent variables tested as part of this step include: next day's price, next week's price, next month's price, and percentage change for next day's price. These models were then tested using the daily prices for 2016 to 2018.  The outcome of this step was to use next days price as the dependent variable and use only one-year for model testing instead of three years. It was noted during this step that the performance of the model deterioates as farther the testing data gets from the training data.  
2. Use daily prices from 2000 to 2015 to train linear regression models with costs log-transformed and scaled to center over zero.  Test using the daily prices for the year 2016.
3. Use daily prices from 2000 to 2015 to train univariate time series models using gold prices with and without log-transformation. Test using the daily prices for the year 2016.

For time series models, only autoregressive (AR) models were used. For two time series functional forms explored, with and without log-tranformed gold prices, the optimal order was determined to be 3 preceding days based on the Akaike Informational Criterion (AIC). 

#### Model Evaluation
A critical step in evaluating the model performance for a time series vaiable like gold price is to compare the results against a naive solution.  In case of a non-seasonal variable like gold price (as opposed to a seasonable variable like cotton prices or traffic volumes), the naive forecast is <i>tomorrow's gold price is same as today's</i>. Naive solution has limited utility as an investment strategy or decision making tool, but it makes a benchmark solution for model evalutation. For model building steps 1 to 3 mentioned in the previous section, the metric used in model selection and comparing agaist the naive solution is R-square value.

Once a model was selected, that functional form was used in performaning a chronological training/testing using an expanding window concept as shown below. Chronological training/testing was done for daily prices for every individual year from 2006 to 2018, using models that were trained with the data from the preceding years beginning with year 2000. For this chronological testing, Mean Absolute Error (MAE) was used for comparing against the naive solution.

![Expanding Window](/img/GoldPrices/ExpandingWindow.png "Train/Test")
<hr>

### Results
Test results for the year 2016 from the model building step for simple linear regression are presented below.  This model was trainined using the daily gold prices from 2000 to 2015, and produced the best R-squared value compared against the naive forecast (0.9719 vs 0.9715). The model tracks pretty well with the actual 2016 gold prices, as demonstrated below:

![Linear 2016](/img/GoldPrices/Year2016_timemap.png "Linear Model: 2016 Test")

A scatterplot of forecasted 2016 gold prices vs actual gold prices is shown below. 

![Scatter Linear 2016](/img/GoldPrices/Yr2016_Gold.png "Linear Model Scatter Plot: 2016 Test")

Chronological training/test evaluation performed using the linear regression models are presented below for the years 2006 to 2012. For every single year in this timeframe, the forecasts are closer to the actual than the naive forecasts. Average forecast deviation is measured using the mean absolute error (MAE).

![2006 to 2012](/img/GoldPrices/Upto2012_v2.png "Test Results for 2006 to 2012")

Chronological training/test results for 2006 to 2018 are shown below. Notice how the model failed to capture the shifting trends in the gold market occured in 2013. 

![2006 to 2018](/img/GoldPrices/AllYears_v2.png "Test Results for 2006 to 2018")

<b> What happened in 2013? </b><br>
A huge swift in investments from precious metals to equites resulted in near 30% loss in gold prices in 2013 ending a 12-year bull market. This drop is primarily related to investors speculating US Federal Reserves cutting down their bond purchases removing any need to purchase gold as a hedge against US Dollar and inflation ([WSJ Article](https://www.wsj.com/articles/no-headline-available-1388504140)).

The model estitmated using 2000 to 2012 training data was unable to fully capture the changing market conditions between equties, t-bills, US dollar, Federal Monetary policies, and the commodities markets.
<hr>
### Conclusions
The above historical analysis to model gold prices using linear regression and time series resulted in the following conclusions:
1. For time series variavbles, always compare against a naive solution whether its possible to beat it or not.
2. Check how far your model can hold and extend the trends. Not all models can completely generalize a highly complex system like stocks and commodity markets. Aim for a smaller timeframe and update frequently.
3. Multiple models: have multiple models. Even if time series didn't performe better than the naive forecasts, it would have captured the 2013 gold price drops better than the linear regression that did well against naive forecast.
<br>
<br>
<hr>
### References
1. [Forecasting at Uber: An Introduction](https://eng.uber.com/forecasting-introduction/) 
2. [Sami et al, <i>Predicting Future Gold Rates using Machine Learning Approach</i>, International Journal of Advanced Computer Science and Applications,
Vol. 8, No. 12, 2017](http://thesai.org/Downloads/Volume8No12/Paper_13-Predicting_Future_Gold_Rates.pdf)
3. [Gold Falls 28% in 2013, Ends 12-Year Bull Run](https://www.wsj.com/articles/no-headline-available-1388504140)












