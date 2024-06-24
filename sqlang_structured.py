# -*- coding: utf-8 -*-
import re
import sqlparse #0.4.2

#骆驼命名法
import inflection

#词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#词干提取
from nltk.corpus import wordnet

#############################################################################
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }

scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])

#---------------------子函数1：代码的规则--------------------
def tokenizeRegex(s):
    """
    使用正则表达式扫描器对输入的字符串 s 进行扫描，并返回扫描结果。

    参数:
    s (str): 要进行扫描的输入字符串。

    返回:
    list: 包含扫描结果的列表，通常是正则表达式匹配到的子串或标记的列表。
    """
    results = scanner.scan(s)[0]
    return results

#---------------------子函数2：代码的规则--------------------
class SqlangParser():
    @staticmethod
    def sanitizeSql(sql):
        """
        对输入的 SQL 语句进行预处理，确保其格式符合解析要求。

        参数:
        sql (str): 要预处理的 SQL 语句。

        返回:
        str: 预处理后的 SQL 语句。

        注意:
        - 去除首尾空格并将所有字符转换为小写。
        - 如果 SQL 语句不以分号结尾，添加分号。
        - 在括号前后添加空格以便于后续解析。
        - 将 'index', 'table', 'day', 'year', 'user', 'text' 等关键字替换为带有 '1' 后缀的单词，以避免冲突。
        - 移除 SQL 语句中的 '#' 字符。
        """
        s = sql.strip().lower()
        if not s[-1] == ";":
            s += ';'
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')
        return s

    def parseStrings(self, tok):
        """
        解析 SQL 语句中的字符串常量，并根据配置替换其值。

        参数:
        tok (sqlparse.sql.TokenList 或 sqlparse.sql.Token): 要解析的 SQL token。

        注意:
        - 递归处理 TokenList 类型的 token。
        - 如果 token 类型是字符串，根据 regex 配置替换其值。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"

    def renameIdentifiers(self, tok):
        """
        重命名 SQL 语句中的标识符（列名和表名），以避免冲突。

        参数:
        tok (sqlparse.sql.TokenList 或 sqlparse.sql.Token): 要重命名的 SQL token。

        注意:
        - 递归处理 TokenList 类型的 token。
        - 根据列名和表名的映射表进行重命名。
        - 处理整数、浮点数和十六进制数的特定标记。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]

        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def __hash__(self):
        """
        计算 SqlangParser 对象的哈希值，基于其 token 列表。

        返回:
        int: SqlangParser 对象的哈希值。
        """
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))

    def __init__(self, sql, regex=False, rename=True):
        """
        初始化 SqlangParser 对象，解析和处理 SQL 语句。

        参数:
        sql (str): 要解析的 SQL 语句。
        regex (bool): 是否使用正则表达式解析字符串常量。
        rename (bool): 是否重命名标识符以避免冲突。

        注意:
        - 初始化各种映射表和计数器。
        - 预处理 SQL 语句。
        - 解析 SQL 语句并构建解析树。
        - 识别并处理 SQL 语句中的子查询、函数、表和字符串常量。
        - 根据配置重命名标识符。
        """

        self.sql = SqlangParser.sanitizeSql(sql)

        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex

        self.parseTreeSentinel = False
        self.tableStack = []

        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]

        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])

        self.parseStrings(self.parse[0])

        if rename:
            self.renameIdentifiers(self.parse[0])

        self.tokens = SqlangParser.getTokens(self.parse)

    @staticmethod
    def getTokens(parse):
        """
        获取解析树中的所有 token，并将其展平为一个列表。

        参数:
        parse (list): 解析树，包含 SQL 表达式的列表。

        返回:
        list: 展平后的 token 列表，每个 token 作为一个字符串。
        """
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    def removeWhitespaces(self, tok):
        """
        移除 SQL 解析树中的空白 token。

        参数:
        tok (sqlparse.sql.TokenList 或 sqlparse.sql.Token): 要处理的 SQL token。

        注意:
        - 递归处理 TokenList 类型的 token。
        - 将非空白的 token 保存到临时列表中，并更新原 token 列表。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)

            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)

    def identifySubQueries(self, tokenList):
        """
        识别并标记 SQL 解析树中的子查询。

        参数:
        tokenList (sqlparse.sql.TokenList): 要处理的 SQL token 列表。

        返回:
        bool: 是否识别到子查询。
        """
        isSubQuery = False

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    def identifyLiterals(self, tokenList):
        """
        识别并标记 SQL 解析树中的字面量和特定 token 类型。

        参数:
        tokenList (sqlparse.sql.TokenList): 要处理的 SQL token 列表。

        """
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
                tok.ttype = COLUMN

    def identifyFunctions(self, tokenList):
        """
        识别并标记 SQL 解析树中的函数调用。

        参数:
        tokenList (sqlparse.sql.TokenList): 要处理的 SQL token 列表。

        """
        for tok in tokenList.tokens:
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    def identifyTables(self, tokenList):
        """
        识别并标记 SQL 解析树中的表名。

        参数:
        tokenList (sqlparse.sql.TokenList): 要处理的 SQL token 列表。

        """
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]

            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE

            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True

            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)

            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    def __str__(self):
        """
        返回 SQL 解析树中 token 的字符串表示。

        返回:
        str: SQL 解析树的字符串表示。
        """
        return ' '.join([str(tok) for tok in self.tokens])

    def parseSql(self):
        """
        获取 SQL 解析树中所有 token 的字符串表示。

        返回:
        list: 包含 SQL 解析树中所有 token 字符串表示的列表。
        """
        return [str(tok) for tok in self.tokens]
#############################################################################

#############################################################################
#缩略词处理
def revert_abbrev(line):
    """
    将输入字符串中的英文缩写还原为完整形式。

    参数:
    line (str): 包含缩写形式的输入字符串。

    返回:
    str: 缩写还原后的字符串。


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

#获取词性
def get_wordpos(tag):
    """
    根据词性标记获取对应的 WordNet 词性。

    参数:
    tag (str): 词性标记。

    返回:
    str 或 None: 对应的 WordNet 词性，如果没有对应的词性，则返回 None。


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

#---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    对输入的自然语言句子进行预处理。

    参数:
    line (str): 包含自然语言文本的输入字符串。

    返回:
    str: 预处理后的字符串。

    注意:
    - 处理各种缩写形式并还原为完整形式。
    - 替换制表符和换行符为空格，并去除多余的空格。
    - 将骆驼命名转换为下划线命名。
    - 去除括号内的内容。
    - 去除句子末尾的句号和空格。
    """
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line=line.strip()
    return line


#---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    """
    对输入的句子进行分词和词性标注。

    参数:
    line (str): 输入的句子。

    返回:
    list: 处理后的单词列表。

    """
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
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
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words= line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    #词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list=[]
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

def filter_all_invachar(line):
    """
    去除输入字符串中的所有非常用符号。

    参数:
    line (str): 输入字符串。

    返回:
    str: 去除非常用符号后的字符串。

    注意:
    - 保留的符号包括：数字、字母、横杠、下划线、单引号、双引号、括号、换行符等。
    - 去除多余的横杠和下划线。
    - 替换或去除其他符号。

    """
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
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
    去除输入字符串中的部分非常用符号。

    参数:
    line (str): 输入字符串。

    返回:
    str: 去除部分非常用符号后的字符串。
    """
    #去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

########################主函数：代码的tokens#################################
def sqlang_code_parse(line):
    """
    对输入的代码进行预处理和解析，返回标记列表。

    参数:
    line (str): 输入的代码字符串。

    返回:
    list: 处理后的代码标记列表，如果解析失败返回 '-1000'。

    注意:
    - 去除部分非常用符号。
    - 替换小数和其他模式。
    - 使用 SqlangParser 进行 SQL 解析。
    - 将骆驼命名转换为下划线命名。
    - 处理和转换代码标记为小写形式。

    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)

    line = re.sub('>>+', '', line)#新增加
    line = re.sub(r"\d+(\.\d+)+",'number',line)#新增加 替换小数

    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typedCode = query.parseSql()
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower()  for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def sqlang_query_parse(line):
    """
    对输入的查询语句进行预处理和解析，返回标记列表。

    参数:
    line (str): 输入的查询语句字符串。

    返回:
    list: 处理后的查询语句标记列表。

    注意:
    - 去除所有非常用符号。
    - 进行自然语言句子的预处理。
    - 对句子进行分词和词性标注。
    - 去除分词结果中的括号内容。
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空

    return word_list


def sqlang_context_parse(line):
    """
    对输入的上下文进行预处理和解析，返回标记列表。

    参数:
    line (str): 输入的上下文字符串。

    返回:
    list: 处理后的上下文标记列表。

    注意:
    - 去除部分非常用符号。
    - 进行自然语言句子的预处理。
    - 对句子进行分词和词性标注。

    """
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    # 解析可能为空
    return word_list

#######################主函数：句子的tokens##################################


if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



