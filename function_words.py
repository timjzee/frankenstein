import pickle


function_words = ["a", "about", "above", "across", "after", "again", "against", "ah", "alas", "all", "almost", "along", "already", "also", "although", "am", "amid", "amidst", "among", "an", "and", "another", "any", "are", "around", "as", "aside", "at", "away", "be", "been", "before", "behind", "being", "below", "beneath", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "d", "did", "do", "does", "down", "during", "each", "either", "enough", "ere", "even", "ever", "every", "except", "few", "first", "five", "following", "for", "from", "given", "had", "half", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "however", "i", "if", "in", "into", "is", "it", "its", "itself", "last", "least", "less", "like", "little", "many", "may", "me", "meanwhile", "midst", "might", "mine", "more", "most", "much", "must", "my", "myself", "nay", "near", "neither", "next", "no", "none", "nor", "not", "now", "o", "of", "off", "oh", "on", "once", "one", "or", "our", "out", "over", "past", "part", "re", "s", "several", "shall", "she", "should", "since", "so", "some", "ten", "than", "that", "the", "thee", "their", "them", "themselves", "then", "there", "therefore", "these", "they", "thine", "this", "those", "thou", "though", "thousand", "three", "through", "thus", "thy", "till", "to", "too", "towards", "two", "under", "until", "up", "upon", "us", "various", "very", "was", "we", "were", "what", "whatever", "when", "where", "whether", "which", "while", "whilst", "who", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yes", "yet", "you", "your", "yours", "yourself"]

# print(len(function_words))
path = "/Users/tim/GitHub/frankenstein/"
f = open(path + "function_words.pck", "wb")
pickle.dump(function_words, f)
f.close()
