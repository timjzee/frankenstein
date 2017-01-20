# frankenstein
Authorship attribution of Frankenstein

**Hypothesis**: Frankenstein contains small chunks of text (< 100 consecutive words) written by Percy Bysshe Shelley (PBS).


![alt text](https://github.com/timjzee/frankenstein/blob/master/sample_size.png?raw=true "Hypothesis")


**Results**: The percentage of samples classified as PBS at different sample sizes support the hypothesis if the classifier is trained on Part-of-Speech bigram frequency. This is evidenced by the result that, as sample sizes get smaller, the percentage of PBS classifications go up before they go down (see *tags* in figure below). This is due to the way the sampling process interacts with the chunks of PBS-authored texts in Frankenstein, see figure above.

![alt text](https://github.com/timjzee/frankenstein/blob/master/percentage_testset.png?raw=true "Percentage of PBS classification")

When the classifier is trained on function word frequency, the proportion of PBS classified samples at different sample sizes can be explained by the monotonically increasing relationship between accuracy (or F-score) and sample size (see figure below). Because much more samples were written by MWS compared to PBS, at lower accuracies more samples actually written by MWS are misclassified as being authored by PBS than the other way around. This results in the monotonically decreasing curve for *words* in the figure above.

![alt text](https://github.com/timjzee/frankenstein/blob/master/f_score_trainset.png?raw=true "F-score of cross-validation")
