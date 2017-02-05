import pickle

sample_size = input("Sample size?")
group_num = input("Group number?")
index = int(input("Sample index?"))

f = open("/Users/tim/GitHub/frankenstein/sampled_texts/franken_samples/samples_{}-g{}.pck".format(sample_size, group_num), "rb")
samples = pickle.load(f)
f.close()

print(samples[index])
