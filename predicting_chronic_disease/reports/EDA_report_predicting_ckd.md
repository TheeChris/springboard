# Predicting Chronic Kidney Disease
## Exploratory Data Analysis and Inferential Statistics
*Note: The code used in this analysis can be found in [this Jupyter Notebook](EDA_predicting_chronic_disease.ipynb)*

With nearly 400 features, it would be easy to get lost in the vast volume of potential interactions. For the purpose of exploratory data analysis and statistical inference, I first wanted to focus on finding a few different types of features. To do this, I started with an extra trees classifier. This classifier fit 10 random decision trees on various sub-samples of the data in order to predict the features’ contributions to the variance in rates of CKD. I ran this classifier multiple times and found, not surprisingly, that the top 10 features were always other chronic health diseases (with the top 3 always being stroke, diabetes, and coronary heart disease). 

![20 Most Important Features](img/important_features.png)

However, a few interesting findings from the feature importance rankings were surrounding workforce participation, type of health insurance, and elder populations living alone. It is interesting that workforce participation (in the form of Labor Force Participation Rate and Employment / Population Ratio) were always in the top 20, but indicators such as specific income levels were always absent. This may indicate that having one or more individuals in a household out of the workforce may be a better indicator as to the risk of chronic kidney disease than the actual household income level. <img src="img/ecdf_labor_ckd.png" alt="ECDF of labor participation rates" width="60%" align="right" />For now, however, I examined whether the Labor Force Participation Rate had a significant impact on rates of CKD. I started by splitting the tracts into 3 groups of high, low, and normal rates of CKD (high and low were defined as being 2 standard deviations away from the mean). From the ECDF we can see that there are not a lot of census tracts with low rates of CKD (only 108 of 27,000) and the low rates do not appear to be normally distributed (in fact, they seem to be bimodal). However, the distribution does appear to be quite separate and descriptive statistics back this up, although there are large degrees of standard error.

|                   | Mean CKD Rate | Mean Labor Participation Rate | Standard Error of Labor Participation |
| ----------------- | ------------- | ----------------------------- | ------------------------------------- |
| Low Rates of CKD  | 0.88%         | 58.94%                        | 25.97%                                |
| High Rates of CKD | 4.912%        | 49.32%                        | 9.63%                                 |

Assuming a null hypothesis that there is no difference in mean labor participation rates between groups with low and high rates of CKD and a significance level of 0.01, we calculate a 99% confidence interval for the difference of means between -6.55 and + 6.44. With our sample difference of means of 9.62, we calculate a z-score of 3.83, which translates to an estimated p-value of 0.0001. In addition, our 99% confidence interval for the difference of means calculated from the sample data is between +3.09 and +16.22, which does not include zero. For these reasons, we reject the null hypothesis and find that there may be a difference in mean labor participation rates between groups of high and low CKD. 

<img src="img/pairs_bs_labor.png" alt="Pairs Bootstrap Regression" width="60%" align="right" />A bootstrap analysis only further solidifies this rejection of the null hypothesis by providing a 99% confidence interval of the difference of mean assuming the null hypothesis to be between -0.065 and + 0.066, with a bootstrap sample mean difference in means of 9.62 and a z-score of 384.31, giving a p-value of virtually 0.0. This implies that low rates of labor participation can be correlated with higher rates of CKD.

Next, we examine whether the type of insurance coverage (public, private, or no insurance) may have an effect on rates of CKD. <img src="img/pairs_bs_insurance.png" alt="Pairs Bootstrap Regression of Insurance Type" width="60%" align="right" />We will assume a null hypothesis that the type of insurance does not affect the rate of CKD. (i.e. the rate of CKD is the same no matter what type of insurance you have). Examining a scatter plot of the data with 10,000 bootstrapped slopes, we can see that private health insurance is very strongly negatively correlated with rates of CKD and nearly orthogonal to the slope of public health insurance. Interestingly, public health insurance appears to be more positively correlated with CKD than no health insurance. This is corroborated by a bootstrap analysis of the difference of means and difference of means slopes. 

|                         | Sample Difference | 99% Confidence Interval | z-score | p-value |
| ----------------------- | ----------------- | ----------------------- | ------- | ------- |
| Difference of Means     | 4.54              | (4.29, 4.82)            | 44.15   | 0.0     | 
| Difference of Mean Slope| 4.78              | (4.773, 4.780)          | 3714.96 | 0.0     |
| Difference of Pearson r | 0.1473            | (0.1471, 0.1474)        | 2518.91 | 0.0     |

The difference between the slope of public and no insurance is likely due to the fact that older people are more likely to have public insurance (Medicare) than no insurance as compared to younger populations, and since age is strongly correlated with rates of CKD, there is a potential collinear effect. A bootstrap analysis of private health insurance shows a 99% confidence interval for the slope between -17.78 and -17.13, indicating a strong negative correlation. We, therefore, reject the null hypothesis and find evidence to believe that the type of insurance may have an effect on rates of CKD. While it makes intuitive sense that the lack of health insurance would lead to higher rates of CKD, it seems implausible that an increased average age would alone account for the positive correlation with CKD.

<img src="img/live_alone_ckd.png" alt="joint plot of elderly living alone and CKD rate" width="60%" align="left" />The third variable of interest was the percentage of people over the age of 65 who are living alone. Examining the scatterplot with the regression line, we can see there appears to be a positive correlation. The hypothesis was first tested using the sample data and then 10,000 bootstrapped samples. The results are shown in the table below. Since the data appears to be skewed to the left for both variables, the Spearman ⍴ was also calculated. Spearman ⍴ at 0.4149 is slightly lower than the Pearson r, but with a p-value of 0.0, still well beyond the significance level. Since our mean slope and correlation coefficient both remain above 0 in the 99% confidence interval, we will reject the null hypothesis. While it is possible that this is the result of collinearity between the percent of people over the age of 65 and the percent of that same population living alone, a quick analysis showed that there was a positive correlation between the two, but a correlation coefficient of 0.61 indicates that there is room for variability in the data.

<table align="center", width="25%">
  <tr>
    <th colspan="2">Sample Data</th>
  </tr>
  <tr>
    <td>Slope</td>
    <td>3.29</td>
  </tr>
  <tr>
    <td>Pearson r</td>
    <td>0.4459</td>
  </tr>
  <tr>
    <td>p-value</td>
    <td>0.000</td>
  </tr>
  <tr>
    <td>Standard error</td>
    <td>0.0405</td>
  </tr>
</table>

<table align="center", width="50%">
  <tr>
    <th colspan="2">Bootstrap Samples</th>
    <th>Confidence Interval</th>
  </tr>
  <tr>
    <td>Mean Slope</td>
    <td>3.29</td>
     <td>(3.137, 3.454)</td>
  </tr>
  <tr>
    <td>Pearson r</td>
    <td>0.4459</td>
    <td>(0.428, 0.463)</td>
  </tr>
  <tr>
    <td>r²</td>
    <td>0.1989</td>
    <td>(0.183, 0.215)</td>
  </tr>
</table>

As a final note, I found it interesting that either the percent of males or females living below poverty almost always found a place at the bottom of the 20 most important features. Most of the time it was females living below poverty, which made me wonder if there was an interaction effect <img src="img/sex_poverty_ckd_scatter_bs.png" alt="Comparative bootstrap regression of poverty and gender" width="60%" align="left" />between poverty and gender regarding the rate of CKD. To test the null hypothesis that gender does not play a role in how poverty affects the rate of CKD, I tested the difference in mean slopes and mean Pearson r coefficient using bootstrap analysis. The results lead us to reject the null hypothesis and assume that gender and poverty status may provide more insight into the rate of CKD than poverty alone.

|                              | 99% Confidence Interval | Mean Difference | z-score | p-value |
| ---------------------------- | ----------------------- | --------------- | ------- | ------- |
| Difference of Mean Slope     | (0.7019, 0.7108)        | 0.7063          | 413.63  | 0.0000  |
| Difference of Mean Pearson r | (0.0080, 0.0085)        | 0.0083          | 93.35   | 0.0000  |

In conclusion, labor participation rate, types of insurance, people over the age of 65 living alone, and gender combined with poverty status all seem to be correlated with the rate of CKD and may provide predictive power in a statistical model.
