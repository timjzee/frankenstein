known_dataset <- read.csv("/Users/tim/GitHub/frankenstein/results/check_unbalanced_PBS/equalized_results_10-cv_BOTH.csv")

known_dataset$MWS_precision = known_dataset$true_MWS_pred_MWS / (known_dataset$true_MWS_pred_MWS + known_dataset$true_PBS_pred_MWS)
known_dataset$PBS_precision = known_dataset$true_PBS_pred_PBS / (known_dataset$true_PBS_pred_PBS + known_dataset$true_MWS_pred_PBS)
known_dataset$macro_precision = (known_dataset$MWS_precision + known_dataset$PBS_precision) / 2
known_dataset$MWS_recall = known_dataset$true_MWS_pred_MWS / (known_dataset$true_MWS_pred_MWS + known_dataset$true_MWS_pred_PBS)
known_dataset$PBS_recall = known_dataset$true_PBS_pred_PBS / (known_dataset$true_PBS_pred_PBS + known_dataset$true_PBS_pred_MWS)
known_dataset$macro_recall = (known_dataset$MWS_recall + known_dataset$PBS_recall) / 2
known_dataset$macro_f_score = 2 * ((known_dataset$macro_precision * known_dataset$macro_recall) / (known_dataset$macro_precision + known_dataset$macro_recall))
known_dataset$accuracy = (known_dataset$true_MWS_pred_MWS + known_dataset$true_PBS_pred_PBS) / 100

known_dataset2 = known_dataset[known_dataset$sample_size < 800,]
known_dataset2$sample_size = factor(known_dataset2$sample_size)
known_dataset2$fold = factor(known_dataset2$fold)
levels(known_dataset2$feature_type)[levels(known_dataset2$feature_type)=="tokens"] <- "words"

known_aov = with(known_dataset2, aov(macro_f_score ~ sample_size * feature_type + Error(fold / (sample_size * feature_type))))
summary(known_aov)

par(mar=c(4, 4, 2, 4), cex=0.9, cex.lab=1, cex.axis=1)

interaction.plot(known_dataset2$sample_size, known_dataset2$feature_type, known_dataset2$macro_f_score, xlab = "Sample Size", ylab = "F-score", trace.label = "", leg.bty = "o")

##################

franken_dataset = read.csv("/Users/tim/GitHub/frankenstein/results/check_unbalanced_PBS/franken_results_groups_BOTH.csv")

franken_dataset2 = franken_dataset[franken_dataset$sample_size < 800,]
franken_dataset2$sample_size = factor(franken_dataset2$sample_size)
franken_dataset2$group = factor(franken_dataset2$group)
levels(franken_dataset2$feature_type)[levels(franken_dataset2$feature_type)=="tokens"] <- "words"


franken_aov = with(franken_dataset2, aov(percentage_PBS ~ sample_size * feature_type + Error(group / (sample_size * feature_type))))
summary(franken_aov)

interaction.plot(franken_dataset2$sample_size, franken_dataset2$feature_type, franken_dataset2$percentage_PBS, xlab = "Sample Size", ylab = "Percentage classified as PBS", trace.label = "", leg.bty = "o")

#################

known_aov2 = with(known_dataset2, aov(accuracy ~ sample_size * feature_type + Error(fold / (sample_size * feature_type))))
summary(known_aov2)

interaction.plot(known_dataset2$sample_size, known_dataset2$feature_type, known_dataset2$accuracy, xlab = "Sample Size", ylab = "Accuracy", trace.label = "", leg.bty = "o")
