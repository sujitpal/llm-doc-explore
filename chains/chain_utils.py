import numpy as np
import os
import re
import xmltodict

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from typing import Generic, List, Tuple, TypeVar, Union


T = TypeVar("T")


class Result(GenericModel, Generic[T]):
    value: T = Field(alias="result")


def extract_chapter_from_name(question_fn: str,
                              is_question: bool = True) -> str:
    if is_question:
        m = re.match(r"questions-(\d+).txt", question_fn)
    else:
        m = re.match(r"chapter-(\d+).txt", question_fn)
    if m is not None:
        return m.group(1)
    else:
        return ""


def extract_questions(chapters_dir: str) -> List[str]:
    questions = []
    for question_fn in os.listdir(chapters_dir):
        if not question_fn.startswith("questions-"):
            continue
        chapter = extract_chapter_from_name(question_fn)
        question_fp = os.path.join(chapters_dir, question_fn)
        with open(question_fp, "r", encoding="utf-8") as f:
            for line in f:
                question_text = line.strip()[2:]
                questions.append((chapter, question_text))
    return questions


def sample_question(questions: List[str],
                    num_samples: int = 1
                    ) -> Union[Tuple[int, str], List[Tuple[int, str]]]:
    idxs = np.arange(len(questions))
    sample_idxs = np.random.choice(idxs, size=num_samples, replace=False)
    sample_idxs = set(sample_idxs.tolist())
    sample_questions = [q for i, q in enumerate(questions) if i in sample_idxs]
    if num_samples == 1:
        return sample_questions[0]
    else:
        return sample_questions


def read_template_from_file(prompt_fn: str) -> str:
    prompt_fp = os.path.join("../prompts", prompt_fn)
    with open(prompt_fp, "r", encoding="utf-8") as f:
        prompt_template_text = f.read()
    return prompt_template_text


def parse_response(response):
    response = response.strip()
    start_tag, end_tag = "<result>", "</result>"
    is_valid = response.startswith(start_tag) and response.endswith(end_tag)
    if not is_valid:
        pattern = f"(?:{start_tag})(.*)(?:{end_tag})"
        p = re.compile(pattern, re.DOTALL)
        m = p.search(response)
        if m is not None:
            response = start_tag + m.group(1) + end_tag
    try:
        resp_dict = xmltodict.parse(response)
    except Exception as e:
        print("response:", response)
        raise e
    result = Result(**resp_dict)
    return result
