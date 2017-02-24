import pickle
import sys
import turtle
from tkinter import *


SAMPLE_SIZE = 25


def getSamplingInfo():
    """Gets number of samples per group."""
    samples_path = "/Users/tim/GitHub/frankenstein/sampled_texts/check_PBS/franken/"
    try:
        f = open("{}samples_{}-g1.pck".format(samples_path, SAMPLE_SIZE), "rb")
    except:
        print("Can't find sample file.")
        sys.exit()
    samples = pickle.load(f)
    f.close()
    try:
        g = open("{}samples_25-g1.pck".format(samples_path, SAMPLE_SIZE), "rb")
    except:
        print("Can't find sample file.")
        sys.exit()
    max_samples = pickle.load(g)
    g.close()
    return len(samples), len(max_samples)


def getPBSSamples():
    """Gets samples classified as PBS."""
    results_path = "/Users/tim/GitHub/frankenstein/results/check_unbalanced_PBS/"
    try:
        f = open(results_path + "franken_results_TOKENS_PBS_samples.csv", "r")
    except:
        print("Can't find results file.")
        sys.exit()
    prev_group = 0
    prev_ss = "0"
    sample_dict = {}
    sublist = []
    for line in f:
        line_list = line[:-1].split(",")
        if line_list[0] == str(SAMPLE_SIZE):
            if line_list[1] != prev_group:
                if len(sublist) != 0:
                    sample_dict[int(prev_group)] = sublist
                sublist = []
            sublist.append(int(line_list[2]))
            prev_group = line_list[1]
        else:
            if prev_ss == str(SAMPLE_SIZE):
                sample_dict[int(prev_group)] = sublist
        prev_ss = line_list[0]
    f.close()
    return sample_dict


def drawDistribution():
    "Use a turtle to draw a distribution of PBS samples in Frankenstein."
    bar_length = max_samples_per_group + 2
    bar_height = 40
    offset = 10
    horizontal_pixels = bar_length + 2 * offset
    vertical_pixels = 10 * bar_height + 11 * offset
    scrn = turtle.Screen()
    turtle.setup(width=horizontal_pixels, height=vertical_pixels)
    scrn.setworldcoordinates(0, 0, horizontal_pixels, vertical_pixels)
    percy = turtle.Turtle()
    percy.speed(0)
    percy.hideturtle()
    percy.penup()
    percy.setpos(horizontal_pixels - offset, vertical_pixels)
    percy.pendown()
    for group in range(1, 11):
        percy.penup()
        percy.right(90)
        percy.forward(offset + bar_height)
        percy.right(90)
        percy.forward(bar_length - 1)
        percy.right(180)
        percy.pendown()
        for distance in [bar_length, bar_height, bar_length, bar_height]:
            percy.forward(distance)
            percy.left(90)
        for sample in range(1, samples_per_group + 1):
            for i in range(int(SAMPLE_SIZE / 25)):
                percy.forward(1)
                if group in PBS_samples:
                    if sample in PBS_samples[group]:
                        percy.left(90)
                        percy.forward(bar_height - 1)
                        percy.right(180)
                        percy.forward(bar_height - 1)
                        percy.left(90)
        percy.forward(1)
    scrn.getcanvas().postscript(file="input_{}.eps".format(SAMPLE_SIZE))
    scrn.mainloop()


samples_per_group, max_samples_per_group = getSamplingInfo()
PBS_samples = getPBSSamples()
drawDistribution()
