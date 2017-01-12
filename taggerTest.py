import re
from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


test_samples = ["Red thunder-clouds, borne on the wings of the midnight whirlwind, floated, at fits, athwart the crimson-coloured orbit of the moon; the rising fierceness of the blast sighed through the stunted shrubs, which, bending before its violence, inclined towards the rocks whereon they grew: over the blackened expanse of heaven, at intervals, was spread the blue lightning's flash; it played upon the granite heights, and, with momentary brilliancy, disclosed the terrific scenery of the Alps, whose gigantic and mishapen summits, reddened by the transitory moon-beam, were crossed by black fleeting fragments of the tempest-clouds. The rain, in big drops, began to descend, and the thunder-peals, with louder and more deafening crash, to shake the zenith, till the long-protracted war, echoing from cavern to cavern, died, in indistinct murmurs, amidst the far-extended chain of mountains. In this scene, then, at this horrible and tempestuous hour, without one existent earthy being whom he might claim as friend, without one resource to which he might fly as an asylum from the horrors of neglect and poverty, stood Wolfstein; -- he gazed upon the conflicting elements; his youthful figure reclined against a jutting granite rock; he cursed his wayward destiny, and implored the Almighty of Heaven to permit the thunderbolt, with crash terrific and exterminating, to descend upon his head, that a being useless to himself and to society might no longer, by his existence, mock Him whone'er made aught in vain.", "A door, which before had appeared part of the solid rock, flew open at the chieftain's touch, and the whole party advanced into the spacious cavern. Over the walls of the lengthened passages putrefaction had spread a bluish clamminess; damps hung around, and, at intervals, almost extinguished the torches, whose glare was scarcely sufficient to dissipate the impenetrable obscurity. After many devious windings they advanced into the body of the cavern: it was spacious and lofty. A blazing wood fire threw its dubious rays upon the mishapen and ill-carved walls. Lamps suspended from the roof, dispersed the subterranean gloom, not so completely however, but that ill-defined shades lurked in the arched distances, whose hollow recesses led to different apartments. The gang had sate down in the midst of the cavern to supper, which a female, whose former loveliness had left scarce any traces on her cheek, had prepared. The most exquisite and expensive wines apologized for the rusticity of the rest of the entertainment, and induced freedom of conversation, and wild boisterous merriment, which reigned until the bandits, overcome by the fumes of the wine which they had drank, sank to sleep. Wolfstein, left again to solitude and silence, reclining on his mat in a corner of the cavern, retraced, in mental, sorrowing review, the past events of his life: ah! that eventful existence whose fate had dragged the heir of a wealthy potentate in Germany from the lap of luxury and indulgence, to become a vile associate of viler bandits, in the wild and trackless deserts of the Alps."]

new_list = []
for test_text in test_samples:
    sentences = re.split(r'([;.!?][ "])', test_text)
    sentences2 = []
    for sentence_index in range(0, len(sentences) - 1, 2):
        sentences2.append(sentences[sentence_index] + sentences[sentence_index + 1])
    sentences2.append(sentences[-1])
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences2]
    tagged_sentences = pos_tag_sents(tokenized_sentences)
    tag_list = [tpl[1] for sentence in tagged_sentences for tpl in sentence]
    tag_string = " ".join(tag_list)
    new_list.append(tag_string)

tag_regex = re.compile(r'[^ ]+')
count_vect = CountVectorizer(analyzer='word', token_pattern=tag_regex, lowercase=False, ngram_range=(2, 2))
counts = count_vect.fit_transform(new_list)
counts.shape
count_vect.get_feature_names()
count_vect.vocabulary_.get('NNP NN')
