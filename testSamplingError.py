import random
import math
import matplotlib.pyplot as plt

f, ax = plt.subplots(1)
TEXT_SIZE = 40000
CHUNK_SIZES = [25, 50, 100, 200]
PROPORTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
ITERATIONS = 1000


def initializeText():
    """Creates a list of writer labels each of which represents a word."""
    txt = ["A" for i in range(TEXT_SIZE)]
    return txt


def insertB():
    """Insert chunks of writer B in the text."""
    num_chunks = int((PROPORTION_B * TEXT_SIZE) / CHUNK_SIZE)
    while num_chunks != 0:
        start_index = random.randint(0, TEXT_SIZE - 1)
        end_index = start_index + CHUNK_SIZE
        if end_index <= (TEXT_SIZE - 1) and text[start_index] == "A" and text[end_index] == "A":
            text[start_index:end_index] = ["B" for i in range(CHUNK_SIZE)]
            num_chunks -= 1


def sampleText(initialize):
    """Samples text, calculates writer proportions, and labels samples."""
    if initialize:
        percentages = {}
    for sample_size in [25, 50, 100, 200, 400]:
        sampled_text = [text[i:i + sample_size] for i in range(0, TEXT_SIZE, sample_size)]
        labels = []
        for sample in sampled_text:
            if sample.count("A") > sample.count("B"):
                labels.append("A")
            else:
                labels.append("B")
        if initialize:
            percentages[sample_size] = labels.count("B") / len(labels) * 100
        else:
            b_percentages[sample_size] += labels.count("B") / len(labels) * 100
    if initialize:
        return percentages


def adjustPercentages():
    """Adjusts mean percentages based on measured accuracy at different sample sizes."""
    adj_percentages = {}
    bline_percentages = {}
    for k in mean_percentages:
        accuracy = (0.75 - math.log2(25) / 20) + math.log2(k) / 20
        bline_percentages[k] = (1 - accuracy) * 100
        adj_percentage = mean_percentages[k] * accuracy + (100 - mean_percentages[k]) * (1 - accuracy)
        adj_percentages[k] = adj_percentage
    return adj_percentages, bline_percentages


def drawPercentages():
    """Draws percentages."""
    if counter == 1:
        ax.plot([k for k in baseline_percentages], [baseline_percentages[k] for k in baseline_percentages], "b")
    ax.plot([k for k in adjusted_percentages], [adjusted_percentages[k] for k in adjusted_percentages], "r", linewidth=0.5)


def getType():
    gtype = ""
    while not (gtype == "chunks" or gtype == "proportion"):
        gtype = input("Choose which parameter to vary (chunks/proportion)? ")
    return gtype


graph_type = getType()
if graph_type == "chunks":
    PROPORTIONS = [0.1]
else:
    CHUNK_SIZES = [100]
counter = 0
for PROPORTION_B in PROPORTIONS:
    for CHUNK_SIZE in CHUNK_SIZES:
        counter += 1
        text = initializeText()
        insertB()
        b_percentages = sampleText(True)
        for i in range(ITERATIONS):
            text = initializeText()
            insertB()
            sampleText(False)
        mean_percentages = {k: b_percentages[k] / ITERATIONS for k in b_percentages}
        print(mean_percentages)
        adjusted_percentages, baseline_percentages = adjustPercentages()
        print(adjusted_percentages)
        drawPercentages()
plt.show()
