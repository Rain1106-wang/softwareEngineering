import pickle
'''
重复参数：get_vocab(total_data2, total_data2) 应为 get_vocab(total_data1, total_data2)。

调试信息：print 调试信息应在生产代码中移除，或替换为适当的日志记录。

直接保存集合：直接将集合转换为字符串保存，读取时需要特殊处理。可以考虑使用 json 序列化和反序列化集合，以便更安全和便于解析。

错误的函数调用：在 __main__ 块中，函数名 final_vocab_processing 应为 vocab_processing。'''

def get_vocab(corpus1, corpus2):
    """
    从两个语料库中提取词汇表。

    参数:
    corpus1 (list): 第一个语料库中的文档列表。
    corpus2 (list): 第二个语料库中的文档列表。

    返回:
    set: 包含两个语料库中所有唯一单词的集合。
    """
    word_vocab = set()
    for corpus in [corpus1, corpus2]:
        for i in range(len(corpus)):
            word_vocab.update(corpus[i][1][0])
            word_vocab.update(corpus[i][1][1])
            word_vocab.update(corpus[i][2][0])
            word_vocab.update(corpus[i][3])
    print(len(word_vocab))
    return word_vocab


def load_pickle(filename):
    """
    从 pickle 文件中加载数据。

    参数:
    filename (str): pickle 文件的路径。

    返回:
    object: 加载的数据。
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def vocab_processing(filepath1, filepath2, save_path):
    """
    处理文本文件中的词汇表。

    参数:
    filepath1 (str): 第一个文本文件的路径。
    filepath2 (str): 第二个文本文件的路径。
    save_path (str): 保存处理后词汇表的路径。

    """
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    word_set = get_vocab(total_data2, total_data2)

    excluded_words = total_data1.intersection(word_set)
    word_set = word_set - excluded_words

    print(len(total_data1))
    print(len(word_set))

    with open(save_path, 'w') as f:
        f.write(str(word_set))


if __name__ == "__main__":
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
