import re
import math
from fractions import Fraction

def normalize_number_format(string):
    """쉼표 제거 및 `th` 접미사 처리"""
    if string is None:
        return None
    string = string.replace(",", "")  # 쉼표 제거
    string = re.sub(r"(\d+)th", r"\1", string)  # "12th" → "12"
    return string.strip()


def convert_decimal_to_fraction(string):
    """소수를 분수로 변환"""
    try:
        if string is None:
            return None
        
        value = float(string)
        
        # 무한대 또는 NaN 값 방지
        if not math.isfinite(value):
            return string
        
        fraction = Fraction(value).limit_denominator(1000)  # 1000까지 제한하여 근사값 방지
        return f"\\frac{{{fraction.numerator}}}{{{fraction.denominator}}}"
    except ValueError:
        return string
    
def last_boxed_only(sample):
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def only_until_first_boxed_from_tokens(string, tokens):
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    
    cum_length = 0
    for i, t in enumerate(tokens):
        cum_length += len(t)
        if cum_length >= idx:
            break
    
    return tokens[:i]



def clean_numbers(sample):
    if not sample:
        return None
    new_sample = list()
    for s in sample:
        new_sample.append(_clean_numbers(s))

    return tuple(new_sample)

def _clean_numbers(string):
    """
    Clean Numbers in the given string

    >>> _clean_numbers(None, "Hello 123")
    'Hello 123'
    >>> _clean_numbers(None, "Hello 1234")
    'Hello 1,234'
    >>> _clean_numbers(None, "Hello 1234324asdasd")
    'Hello 1,234,324asdasd'
    """
    num_prev_digits = 0
    new_string = ""
    for i, c in enumerate(string):
        # isdigit() doesnt work here because of weird unicode chars.
        if c in {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}:
            num_prev_digits += 1
        else:
            if num_prev_digits > 3:
                # Some fixing
                string_number = new_string[-num_prev_digits:]
                new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))
            num_prev_digits = 0
        new_string += c

    if num_prev_digits > 3:
        # Some fixing
        string_number = new_string[-num_prev_digits:]
        new_string = new_string[:-num_prev_digits] + "{0:,}".format(int(string_number))

    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def remove_text_env(string):
    """LaTeX의 \text{} 환경을 제거하고 내부 문자열만 남김"""
    return re.sub(r"\\text\{(.*?)\}", r"\1", string)

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = remove_text_env(string)
    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        #pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

class NotEqual:
    def __eq__(self, other):
        return False
    
def check_equivalence(answer, ground_truth, question=None):
    """답변이 정답과 동등한지 확인"""
    if answer == "NO_ANSWER" or answer is None:
        return False
    
    # 1️⃣ 숫자 형식 정규화
    normalized_answer = normalize_number_format(answer)
    normalized_ground_truth = normalize_number_format(ground_truth)

    # 2️⃣ 분수 & 소수 변환
    converted_answer = convert_decimal_to_fraction(normalized_answer)
    converted_ground_truth = convert_decimal_to_fraction(normalized_ground_truth)

    # 3️⃣ 기존 math_parsingutil.is_equiv() 호출
    return (
        is_equiv(normalized_answer, normalized_ground_truth) or
        is_equiv(converted_answer, converted_ground_truth)
    )

    
def extract_boxed(text):
    # 먼저 \boxed{ 형태를 찾음
    start_idx = text.find("\\boxed{")
    if start_idx != -1:
        start_idx += len("\\boxed{")
        count = 1
        end_idx = start_idx
        while end_idx < len(text) and count > 0:
            if text[end_idx] == '{':
                count += 1
            elif text[end_idx] == '}':
                count -= 1
            end_idx += 1
        if count == 0:
            return text[start_idx:end_idx-1].strip()
    
    # \boxed 다음에 {} 없이 바로 오는 숫자 처리
    start_idx = text.find("\\boxed ")
    if start_idx != -1:
        start_idx += len("\\boxed ")
        end_idx = start_idx
        while end_idx < len(text) and text[end_idx] not in (" ", "$", ".", ",", "}"):
            end_idx += 1
        return text[start_idx:end_idx].strip()

    return None
    
def extract_answer(text):
    boxed_content = extract_boxed(text)
    if boxed_content:
        ans = boxed_content.strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    final_pattern = r"(?i)final answer is[:\s]+\$?(.+?)(?:\$|\n|$)"
    matches = re.findall(final_pattern, text)
    if matches:
        ans = matches[-1].strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    ans_pattern = r"(?i)answer[:\s]+\$?(.+?)(?:\$|\n|$)"
    matches = re.findall(ans_pattern, text)
    if matches:
        ans = matches[-1].strip()
        if ans == "X":
            return "NO_ANSWER"
        return ans
    return None

def check_equivalence(answer, ground_truth, question=None):
    """답변이 정답과 동등한지 확인"""
    if answer == "NO_ANSWER" or answer is None:
        return False
    
    if ground_truth is None:
        print("Warning: ground_truth is None")  # 디버깅 로그 추가
        return False  # 또는 다른 예외 처리

    # 1️⃣ 숫자 형식 정규화
    normalized_answer = normalize_number_format(answer)
    normalized_ground_truth = normalize_number_format(ground_truth)

    # 2️⃣ 분수 & 소수 변환
    converted_answer = convert_decimal_to_fraction(normalized_answer)
    converted_ground_truth = convert_decimal_to_fraction(normalized_ground_truth)

    # 3️⃣ 기존 math_parsingutil.is_equiv() 호출
    return (
       is_equiv(normalized_answer, normalized_ground_truth) or
       is_equiv(converted_answer, converted_ground_truth)
    )
