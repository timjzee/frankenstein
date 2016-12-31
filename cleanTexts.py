import re


def getInput():
    input_path = "/Users/tim/OneDrive/Master/Text_Mining/project/texts/"
    input_name = "PBS-zastrozzi-gutAU.txt"
    f = open(input_path + input_name, "r", encoding="utf-8-sig")
    text = f.read()
    f.close()
    return text, input_path, input_name


def checkQuotation(paragraph):
    """Checks whether paragraph is a quotation and indicates whether (part of) the paragraph needs to be deleted."""
    if re.fullmatch(r'  .*\[[0-9]\]', paragraph, re.DOTALL):
        return "delete"
    elif re.fullmatch(r'\[[0-9]\].*', paragraph, re.DOTALL):
        return "delete"
    elif re.search(r'".*"\[[0-9]\]\Z', paragraph, re.DOTALL):
        return re.search(r'".*"\[[0-9]\]\Z', paragraph, re.DOTALL).re
    elif re.search(r'\n--[A-Z].*\Z', paragraph):
        return "delete"
    else:
        return "keep"


def checkTitle(paragraph):
    """Checks whether paragraph is a title or not and returns a Boolean."""
    titles = ["CHAP", "SONG", "CONC", "BALL", "LETT", "INTR", "VOL", "THE END"]
    if re.fullmatch(r'(' + r'|'.join(titles) + r').*', paragraph):
        return True
    elif re.fullmatch(r'[IVXL]+\.', paragraph):
        return True
    else:
        return False


def analyseParagraphs():
    """Splits text into paragraphs and cleans up paragraph-specific mess."""
    paragraphs = re.split(r'(\n|\r\n){2,}', input_text)
    remaining_paragraphs = []
    for par in paragraphs:
        quotation_status = checkQuotation(par)
        title_status = checkTitle(par)
        if title_status is False and quotation_status != "delete":
            if quotation_status == "keep":
                remaining_paragraphs.append(par)
            else:
                new_par = re.sub(quotation_status, "", par)
                remaining_paragraphs.append(new_par)
    return remaining_paragraphs


def analyseText():
    """Cleans whole text of remaining mess."""
    joined_pars = "".join(remaining_pars)
    temp_text1 = re.sub(r'(?<!-)\n(?!-)', ' ', joined_pars)
    temp_text2 = re.sub(r'\n', '', temp_text1)
    temp_text3 = re.sub(r'[_*]', '', temp_text2)
    temp_text4 = re.sub(r'-{2,}', ' -- ', temp_text3)
    temp_text5 = re.sub(r'[“”]', '"', temp_text4)
    temp_text6 = re.sub(r'[‘’]', "'", temp_text5)
    upper2lower = lambda pat: pat.group(1).lower()
    temp_text7 = re.sub(r'(?<![.?!"] | ")([A-Z])(?=[A-Z])', upper2lower, temp_text6)
    temp_text8 = re.sub(r'(?<=[a-zA-Z])([A-Z])', upper2lower, temp_text7)
    temp_text9 = re.sub(r' {2,}', ' ', temp_text8)
    temp_text10 = re.sub(r'\[[0-9]\]', '', temp_text9)
    return temp_text10


def writeOutput():
    new_name = re.sub(r'(?<=-)[^-]+(?=\.txt)', 'clean', file_name)
    f = open(file_path + new_name, 'w', encoding='utf-8')
    f.write(output_text)
    f.close()


input_text, file_path, file_name = getInput()
remaining_pars = analyseParagraphs()
output_text = analyseText()
writeOutput()
