# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize

from nltk import wordpunct_tokenize
from io import StringIO
# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#############################################################################

PATTERN_VAR_EQUAL = re.compile("(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile("for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")


def repair_program_io(code):
    """
    修复从交互式 Python 会话（IPython 或 Python REPL）复制的代码。
    移除提示符和输出指示符，保留实际代码。

    参数:
        code (str): 含有交互提示符和输出指示符的原始代码字符串。

    返回:
        tuple: 包含修复后的代码字符串和代码块列表的元组。
    """
    # 为第一种情况定义正则表达式模式
    pattern_case1_in = re.compile("In ?\[\d+]: ?")  # flag1
    pattern_case1_out = re.compile("Out ?\[\d+]: ?")  # flag2
    pattern_case1_cont = re.compile("( )+\.+: ?")  # flag3

    # reg patterns for case 2
    pattern_case2_in = re.compile(">>> ?")  # flag4
    pattern_case2_cont = re.compile("\.\.\. ?")  # flag5

    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    lines = code.split("\n")
    lines_flags = [0 for _ in range(len(lines))]

    code_list = []  # a list of strings

    # match patterns
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # pdb.set_trace()
    # repair
    if lines_flags.count(0) == len(lines_flags):  # no need to repair
        repaired_code = code
        code_list = [code]
        bool_repaired = True

    elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or \
            re.match(re.compile("(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        '''在这个 if 判断内的 while 循环中，代码假设 pre_idx 总是小于 lines_flags 的长度，这可能导致索引溢出错误。添加索引检查以避免溢出错误。'''
        if lines_flags[0] == 0:
            flag = 0
            while (flag == 0):
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""  # clean

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                # clean sub_block record
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        # avoid missing the last unit
        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    if not bool_repaired:  # not typical, then remove only the 0-flag lines after each Out.
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list

'''
在 get_vars_heuristics 中直接使用正则表达式匹配变量名，存在误匹配的风险。在使用正则表达式时应小心构建模式，确保不误匹配其他内容。


在函数内部使用全局变量（如 tokenize 和 token），可能导致不可预期的行为。将这些变量封装在函数内或作为参数传递，避免全局依赖。


代码中存在被注释掉的调试信息和不必要的 print 语句。使用适当的日志记录方法（例如 logging 模块），并确保在生产代码中移除或禁用调试信息。



'''



def get_vars(ast_root):
    """
    从代码的 AST（抽象语法树）中提取变量名。

    参数:
        ast_root (ast.AST): 抽象语法树的根节点。

    返回:
        list: 变量名的排序列表。
    """
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})


def get_vars_heuristics(code):
    """
    当 AST 解析失败时，启发式地从代码中提取变量名。

    参数:
        code (str): 代码字符串。

    返回:
        set: 变量名集合。
    """
    varnames = set()
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]

    # best effort parsing
    start = 0
    end = len(code_lines) - 1
    bool_success = False
    while not bool_success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            bool_success = True
    # print("Best effort parse at: start = %d and end = %d." % (start, end))
    varnames = varnames.union(set(get_vars(root)))
    # print("Var names from base effort parsing: %s." % str(varnames))

    # processing the remaining...
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            # matching PATTERN_VAR_EQUAL
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # remove "="
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

            # matching PATTERN_VAR_FOR
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # remove "for" and "in"
                varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))

    return varnames


def PythonParser(code):
    """
    解析Python代码，提取代码中的变量名和标记化的代码列表。如果解析失败，则尝试修复代码后再次解析，如果仍然失败，则使用启发式方法获取变量名。

    参数:
        code (str): Python 代码。

    返回:
        tuple: 包含标记列表、变量提取失败标志和标记失败标志的元组。
    """
    bool_failed_var = False
    bool_failed_token = False

    try:
        root = ast.parse(code)
        varnames = set(get_vars(root))
    except:
        repaired_code, _ = repair_program_io(code)
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except:
            # failed_var_qids.add(qid)
            bool_failed_var = True
            varnames = get_vars_heuristics(code)

    tokenized_code = []

    def first_trial(_code):

        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # fetch the next term
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True
                # print("Failed line: ")
                # print sys.exc_info()
                # tokenize the error line with wordpunct_tokenizer
                code_lines = code.split("\n")
                # if lineno <= len(code_lines) - 1:
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]  # error line
                    # print("Failed code line: %s" % failed_code_line)
                    if posno < len(failed_code_line) - 1:
                        # print("Failed position: %d" % posno)
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(
                            failed_code_line)  # tokenize the failed line segment
                        # print("wordpunct_tokenizer tokenization: ")
                        # print(tokenized_failed_code_line)
                        # append to previous tokenizing outputs
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token


#############################################################################

'''
在 revert_abbrev 函数中使用的正则表达式模式可能会误匹配某些缩写形式，且 " 使用不正确。使用更准确的模式，并正确处理撇号 ' 以避免误匹配。

在 get_wordpos 函数中，未处理 wordnet 可能未导入或未初始化的问题。确保正确导入所需的模块，并添加相应的错误处理。

在 process_nl_line 函数中使用的正则表达式可能匹配不准确，尤其是括号处理部分。使用非贪婪匹配来确保匹配正确。

在 process_sent_word 函数中，某些正则表达式模式可能导致误匹配或性能问题。优化正则表达式模式以提高匹配准确性和效率。



'''



#############################################################################
# 缩略词处理
def revert_abbrev(line):
    """
    将输入字符串中的常见英语缩写形式还原为其完整形式。
    恢复代码中的缩略词，将其转换为完整形式。使用字典存储缩略词和对应的完整形式，并在代码中进行替换操作。

    参数:
    line (str): 包含英语句子的字符串，可能包含缩写形式。

    返回:
    str: 还原缩写后的字符串，其中所有缩写形式被替换为其对应的完整形式。
    """
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


# 获取词性
def get_wordpos(tag):
    """
    根据给定的词性标签（tag）返回对应的WordNet词性。

    参数:
    tag (str): 词性标签，如'JJ', 'VB', 'NN'等。

    返回:
    wordnet.synset.SynsetType 或 None: 如果tag以'J'开头，返回wordnet.ADJ；
                                       如果tag以'V'开头，返回wordnet.VERB；
                                       如果tag以'N'开头，返回wordnet.NOUN；
                                       如果tag以'R'开头，返回wordnet.ADV；
                                       否则返回None。

    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# ---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    对输入的英文句子进行预处理，去除其中的冗余和特殊字符，并将驼峰命名转换为下划线命名。

    参数:
    line (str): 包含英文句子的字符串，可能包含缩写、特殊字符和驼峰命名。

    返回:
    str: 预处理后的字符串，其中缩写被还原，特殊字符被去除，驼峰命名被转换为下划线命名。

    """
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^(|^)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除开始和末尾空格
    line = line.strip()
    return line


# ---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    """
    对输入的英文句子进行一系列处理，包括单词查找、替换特殊字符和数字、小写化、词性标注和词干提取。对句子进行分词、标记化、词性标注、词形还原和词干提取等操作，返回处理后的单词列表。

    参数:
    line (str): 包含英文句子的字符串。

    返回:
    list: 经过处理后的单词列表，其中每个单词可能经过了小写化、词性标注、词干提取等处理。


    """
    # 找单词
    line = re.findall(r"\w+|[^\s\w]", line)
    line = ' '.join(line)
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z_])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words = line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


#############################################################################

'''
filter_all_invachar 和 filter_part_invachar 函数中的正则表达式使用了不正确的模式，可能导致不必要的替换。使用更清晰和准确的正则表达式模式，并保持一致性。

filter_all_invachar 和 filter_part_invachar 函数功能几乎相同，可以提取公共部分减少重复代码。合并两个函数以减少重复代码。

python_code_parse 函数中的异常处理过于宽泛，容易掩盖潜在错误。具体捕获可能出现的异常，并添加日志记录或错误处理逻辑。


'''

def filter_all_invachar(line):
    """
    去除字符串中的非常用符号并规范化字符。

    参数:
        line (str): 输入的字符串。

    返回:
        str: 处理后的字符串。
    """
    # 去除非常用符号；防止解析有误
    assert isinstance(line, object)
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    """
    去除输入字符串中的非常用符号，以防止解析时出现错误。

    参数:
    line (str): 待处理的字符串。

    返回:
    str: 处理后的字符串，其中非常用符号被替换为空格。


    """
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


########################主函数：代码的tokens#################################
def python_code_parse(line):
    """
    解析Python代码行，去除非常用字符并转换为tokens列表。

    参数:
    line (str): 待解析的Python代码行。

    返回:
    list: 包含tokens的列表，每个token是代码中的单词或特殊字符。
    如果解析失败，则返回字符串'-1000'。

    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)  # 新增加
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    '''
    line = filter_part_invachar(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    '''
    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
        # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'


########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def python_query_parse(line):
    """
    解析Python查询字符串，去除非常用字符并进行自然语言处理。

    参数:
    line (str): 待解析的Python查询字符串。

    返回:
    list: 包含处理后的单词的列表，不包括括号。

    注意:
    1. 使用`filter_all_invachar`函数去除非常用字符。
    2. 调用`process_nl_line`（未在此处定义）进行自然语言处理相关的行处理。
    3. 调用`process_sent_word`（未在此处定义）进行分词处理。
    4. 遍历分词后的列表，将包含括号的单词替换为空字符串。
    5. 去除列表中的空字符串和空格。

    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[()]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def python_context_parse(line):
    """
    解析Python上下文代码，去除非常用字符并进行自然语言处理。

    参数:
    line (str): 待解析的Python上下文代码。

    返回:
    list: 包含处理后的单词的列表。

    注意:
    1. 使用`filter_part_invachar`函数去除非常用字符。
    2. 调用`process_nl_line`（未在此处定义）进行自然语言处理相关的行处理。
    3. 调用`process_sent_word`（未在此处定义）进行分词处理。
    4. 去除列表中的空字符串和空格。


    """
    line = filter_part_invachar(line)
    # 在这一步的时候驼峰命名被转换成了下划线
    line = process_nl_line(line)
    print(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list


#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    print(python_query_parse(
        'python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    print(python_context_parse(
        'How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    print(python_code_parse(
        'if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    print(python_code_parse(
        "diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    print(python_code_parse(
        '  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
