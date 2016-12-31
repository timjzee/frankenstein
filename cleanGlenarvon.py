import os
import re
from lxml import html


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def getFileNames():
    """Gets html file names"""
    input_path = "/Users/tim/OneDrive/Master/Text_Mining/project/texts/glenarvon_html/"
    temp_list = os.listdir(input_path)
    name_list = [i for i in temp_list if i[-4:] == "html"]
    name_list.sort(key=natural_keys)                                            # see http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    return name_list, input_path


def extractText(html_code):
    """Parses html code to extract certain text."""
    html_tree = html.fromstring(html_code)
    chapter_list = html_tree.find_class("chapter")
    chapter_text = chapter_list[0].text_content()
    return chapter_text


def cleanText(raw_text):
    """Cleans text."""
    temp1 = re.sub(r'.+\n', '', raw_text, count=1)                              # Deletes the first line (Chapter title)
    temp2 = re.sub(r'\n+', ' ', temp1)                                          # Get rids of newline characters
    temp3 = re.sub(r'[“”]', '"', temp2)                                         # Standardizes quotation marks p. 1
    temp4 = re.sub(r'[‘’]', "'", temp3)                                         # Standardizes quotation marks p. 2
    temp5 = re.sub(r'—', ' -- ', temp4)                                         # Standardizes em-dashes
    temp6 = re.sub(r' {2,}', ' ', temp5)                                        # Removes excess whitespaces
    return temp6


def loadFiles():
    """Loads all html files and loops through them"""
    all_chapters = []
    for name in names:
        f = open(file_path + name, "r", encoding="utf-8")
        html_file = f.read()
        f.close()
        chap_text = extractText(html_file)
        new_text = cleanText(chap_text)
        all_chapters.append(new_text)
    concatenated_chapters = " ".join(all_chapters)
    return concatenated_chapters


def writeText():
    ouput_path = "/Users/tim/OneDrive/Master/Text_Mining/project/texts/"
    f = open(ouput_path + "LAMB-glenarvon-clean.txt", "w", encoding="utf-8")
    f.write(concat_chaps)
    f.close()


names, file_path = getFileNames()
concat_chaps = loadFiles()
writeText()
