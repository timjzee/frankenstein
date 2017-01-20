# frankenstein
Authorship attribution of Frankenstein

**Hypothesis**: Frankenstein contains small chunks of text (< 100 consecutive words) written by Percy Bysshe Shelley (PBS).

![alt text](https://github.com/timjzee/frankenstein/blob/master/sample_size.png?raw=true "Hypothesis")

**Results**: The percentage of samples classified as PBS at different sample sizes support the hypothesis if the classifier is trained on Part-of-Speech bigram frequency (see figure below).

![alt text](https://github.com/timjzee/frankenstein/blob/master/percentage_testset.png?raw=true "Percentage of PBS classification")

When the classifier is trained on function word frequency the proportion of PBS classified samples at different sample sizes can be explained by the monotonically increasing relationship between accuracy (or F-score) and sample size (see figure below). Because much more samples were written by MWS compared to PBS, the percentage of PBS classified samples becomes higher as sample sizes get lower.

![alt text](https://github.com/timjzee/frankenstein/blob/master/f_score_trainset.png?raw=true "F-score of cross-validation")
