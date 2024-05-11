import matplotlib.pyplot as plt
import os
import re
import shutil
import sys

DATA_DIR = "../data"
INPUT_DOC_FP = os.path.join(DATA_DIR, "snowflake-book.txt")
RAW_CHAPTER_DIR = os.path.join(DATA_DIR, "raw-chapters")
COOKED_CHAPTER_DIR = os.path.join(DATA_DIR, "chapters")

# split text into chapters
shutil.rmtree(RAW_CHAPTER_DIR, ignore_errors=True)
os.makedirs(RAW_CHAPTER_DIR, exist_ok=True)

chapter_id = 0
fout = open(os.path.join(
    RAW_CHAPTER_DIR, "chapter-{:0d}.txt".format(chapter_id)),
    "w", encoding="utf-8")
with open(INPUT_DOC_FP, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("CHAPTER "):
            m = re.match(r"CHAPTER (\d+)", line.strip())
            assert m is not None
            chapter_id = int(m.group(1))
            fout.close()
            fout = open(os.path.join(
                RAW_CHAPTER_DIR, "chapter-{:0d}.txt".format(chapter_id)),
                "w", encoding="utf-8")
            fout.write(line)
            continue
        fout.write(line)

fout.close()

# # distribution of line lengths
# # <= 40 chars keep on own line >40 merge to next line
# line_lengths = []
# for raw_chapter_fn in os.listdir(RAW_CHAPTER_DIR):
#     if raw_chapter_fn == "chapter-0.txt":
#         continue
#     raw_chapter_fp = os.path.join(RAW_CHAPTER_DIR, raw_chapter_fn)
#     with open(raw_chapter_fp, "r", encoding="utf-8") as f:
#         for line in f:
#             line_lengths.append(len(line.strip()))
# plt.hist(line_lengths)
# plt.title("Distribution of line lengths")
# plt.xlabel("line lengths (#-chars)")
# plt.ylabel("counts")
# _ = plt.show()


def join_para(para):
    para_j = " ".join(para)
    para_j = re.sub(r"\s+", " ", para_j)
    para_j = re.sub(r"\-\s", "", para_j)
    return para_j


def maybe_page_num(line):
    m1 = re.match(".*?\|\s\d+", line)
    m2 = re.match("\d+\s\|.*", line)
    return m1 is not None or m2 is not None


# create cooked directory and cook raw files into it
if not os.path.exists(COOKED_CHAPTER_DIR):
    shutil.rmtree(COOKED_CHAPTER_DIR, ignore_errors=True)
    os.makedirs(COOKED_CHAPTER_DIR, exist_ok=True)
else:
    print("Directory chapters exists already, physically delete to proceed!")
    sys.exit(-1)

for raw_chapter_fn in os.listdir(RAW_CHAPTER_DIR):
    if raw_chapter_fn == "chapter-0.txt":
        continue
    raw_chapter_fp = os.path.join(RAW_CHAPTER_DIR, raw_chapter_fn)
    cooked_chapter_fp = os.path.join(COOKED_CHAPTER_DIR, raw_chapter_fn)
    cooked_question_fp = os.path.join(
        COOKED_CHAPTER_DIR, raw_chapter_fn.replace("chapter", "questions"))
    with open(raw_chapter_fp, "r", encoding="utf-8") as fin, \
         open(cooked_chapter_fp, "w", encoding="utf-8") as fout_c, \
         open(cooked_question_fp, "w", encoding="utf-8") as fout_q:
        at_end_of_chapter = False
        cooked_lines, para = [], []
        for line in fin:
            line = line.strip()
            if line.startswith("Knowledge Check"):
                at_end_of_chapter = True
            if not at_end_of_chapter:
                if line.startswith("---- Page ") or \
                        len(line.strip()) == 0 or \
                        line.startswith("Snowflake  "):
                    continue
                if maybe_page_num(line):
                    continue
                if len(line) < 40 and line.istitle():
                    if len(para) > 0:
                        cooked_lines.append(join_para(para))
                        para = []
                    cooked_lines.append(line)
                else:
                    para.append(line)
            else:
                fout_q.write("{:s}\n".format(line))
        fout_c.write("\n\n".join(cooked_lines) + "\n")
