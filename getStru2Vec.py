import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *

def multipro_python_query(data_list):
    """
    并行处理Python查询数据，使用python_query_parse函数解析每个查询。

    参数:
    data_list (list): 包含多个查询数据的列表，每个元素是一个查询字符串。

    返回:
    list: 处理后的Python查询数据列表，每个元素是解析后的查询结果。
    """
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    """
    并行处理Python代码数据，使用python_code_parse函数解析每段代码。

    参数:
    data_list (list): 包含多个代码数据的列表，每个元素是一个代码字符串。

    返回:
    list: 处理后的Python代码数据列表，每个元素是解析后的代码结果。
    """
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    """
    并行处理Python上下文数据，使用python_context_parse函数解析上下文。

    参数:
    data_list (list): 包含多个上下文数据的列表，每个元素是一个上下文字符串。

    返回:
    list: 处理后的Python上下文数据列表，每个元素是解析后的上下文结果。
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result

def multipro_sqlang_query(data_list):
    """
    并行处理SQL查询数据，使用sqlang_query_parse函数解析每个查询。

    参数:
    data_list (list): 包含多个查询数据的列表，每个元素是一个查询字符串。

    返回:
    list: 处理后的SQL查询数据列表，每个元素是解析后的查询结果。
    """
    return [sqlang_query_parse(line) for line in data_list]

def multipro_sqlang_code(data_list):
    """
    并行处理SQL代码数据，使用sqlang_code_parse函数解析每段代码。

    参数:
    data_list (list): 包含多个代码数据的列表，每个元素是一个代码字符串。

    返回:
    list: 处理后的SQL代码数据列表，每个元素是解析后的代码结果。
    """
    return [sqlang_code_parse(line) for line in data_list]

    return [sqlang_code_parse(line) for line in data_list]

def multipro_sqlang_context(data_list):
    """
    并行处理SQL上下文数据，使用sqlang_context_parse函数解析上下文。

    参数:
    data_list (list): 包含多个上下文数据的列表，每个元素是一个上下文字符串。

    返回:
    list: 处理后的SQL上下文数据列表，每个元素是解析后的上下文结果。
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

def parse(data_list, split_num, context_func, query_func, code_func):
    """
    使用多进程并行处理数据列表，将其拆分为子列表，并分别处理上下文、查询和代码。

    参数:
    data_list (list): 要处理的数据列表，每个元素是一个数据记录。
    split_num (int): 拆分数量，每个子列表的大小。
    context_func (function): 用于处理上下文数据的函数。
    query_func (function): 用于处理查询数据的函数。
    code_func (function): 用于处理代码数据的函数。

    返回:
    tuple: 包含处理后的上下文数据、查询数据和代码数据的元组。
    """
    pool = multiprocessing.Pool()
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
    results = pool.map(context_func, split_list)

    # 并行处理上下文数据
    context_data = [item for sublist in results for item in sublist]
    print(f'context条数：{len(context_data)}')

    # 并行处理查询数据
    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]
    print(f'query条数：{len(query_data)}')

    # 并行处理代码数据
    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]
    print(f'code条数：{len(code_data)}')

    pool.close()
    pool.join()

    return context_data, query_data, code_data

def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    """
    主函数，用于加载数据，进行并行处理，并将处理后的数据保存到文件。

    参数:
    lang_type (str): 编程语言类型，如'python'或'sql'。
    split_num (int): 拆分数量，每个子列表的大小。
    source_path (str): 源数据文件路径，文件中包含待处理的数据。
    save_path (str): 处理后数据的保存路径，保存为Pickle文件。
    context_func (function): 用于处理上下文数据的函数。
    query_func (function): 用于处理查询数据的函数。
    code_func (function): 用于处理代码数据的函数。
    """
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    # 使用多进程并行处理数据
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]

    # 组合处理后的数据
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 将处理后的数据保存到文件
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)

    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code)
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code)
