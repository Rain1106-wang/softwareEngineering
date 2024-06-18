
# softwareEngineering
20211060236 王思雨

## 目录

- [一、项目说明](#一项目框架)
- [二、文件说明](#二文件说明)
  - [2.1 embeddings_process.py文件](#embeddings_processpy文件)
  - [2.2 getSru2Vec.py文件](#getStru2Vecpy文件)
  - [2.3 process_single_corpus.py文件](#process_single_corpuspy文件)
  - [2.4 python_structured.py文件](#python_structuredpy文件)
  - [2.5 sqlang_structured.py文件](#sqlang_structuredpy文件)
  - [2.6 word_dict.py文件](#word_dictpy文件)

## 一、项目说明

此项目的python文件是对文本数据进行预测处理。要求对所有函数进行注释。
```
|── hnn_processing  
│     └── embeddings_process.py  
│     └── getStru2Vec.py
│     └── process_single_corpus.py
│     └── python_structured.py
│     └── sqlang_structured.py
│     └── word_dirt.py
```


## 二、文件说明

### embeddings_process.py 

#### 1. 概述
将词向量文件转换为二进制格式，并从大词典中提取特定于语料的词典，并构建词向量矩阵。然后提供了一些函数来处理语料数据的序列化和索引操作。

#### 2. 具体功能
- `trans_bin`：将词向量文件转换为二进制格式（.bin）文件。
- `get_new_dict`：从大词典中获取特定于语料的词典，并构建词向量矩阵。将无法找到词向量的词存储在失败词列表中，并保存词向量和词典为文件。
- `get_index`：根据词在词典中的位置，获取词的索引。对于代码类型的文本，会在开头和结尾添加特殊标记索引；对于其他类型的文本，会在开头添加空白文本或未知词的索引。
- `Serialization`：将训练、测试或验证语料进行序列化处理。将每个文本中的词转换为对应的索引，并进行填充操作，将其长度统一。然后将处理后的数据保存到文件中。
- `get_new_dict_append`：从现有的词典和词向量中获取特定于语料的新词典，并将其添加到词向量中。加载现有的词典和词向量文件以及包含要添加词的文件，并将新词加入词典和词向量中，保存为新的文件。
---
### getStru2Vec.py文件

#### 1. 概述
实现了并行分词的功能，使用多进程并行处理Python和SQL语料中的文本数据，并进行分词操作。

#### 2. 具体功能
- `multipro_python_query`：对Python语料中的查询文本进行解析和分词处理。
- `multipro_python_code`：对Python语料中的代码文本进行解析和分词处理。
- `multipro_python_context`：对Python语料中的上下文文本进行解析和分词处理。
- `multipro_sqlang_query`：对SQL语料中的查询文本进行解析和分词处理。
- `multipro_sqlang_code`：对SQL语料中的代码文本进行解析和分词处理。
- `multipro_sqlang_context`：对SQL语料中的上下文文本进行解析和分词处理。
- `parse_python`：针对Python语料，调用上述解析函数进行分词处理，并返回分词结果。
- `parse_sqlang`：针对SQL语料，调用上述解析函数进行分词处理，并返回分词结果。
- `parse_data_multiprocess`：将数据切分为多个小块，并利用多线程进行并行处理，提高分词速度。
- `main`：主函数，根据语言类型调用相应的解析函数进行分词处理，并将处理结果保存到文件中。
- `test`：测试函数，用于验证读取和保存的数据是否一致。
---
### process_single_corpus.py文件

#### 1. 概述
实现了Python代码和自然语言句子的解析和处理功能。

#### 2. 具体功能
- `load_pickle`：加载pickle文件的数据。
- `single_list`：计算一个列表中某个元素出现的次数。
- `data_staqc_processing`：针对staqc数据，根据问题ID判断单候选和多候选问题，将数据分别保存到不同的文件中。
- `data_large_processing`：针对large数据，根据问题ID判断单候选和多候选问题，将数据分别保存到不同的文件中。
- `single_unlable2lable`：将有标签的数据转换为带有标签的形式，并按照问题ID和标签进行排序，将数据保存到文件中。
---

### python_structured.py文件

#### 1. 概述
用于对数据进行处理和转换，包括加载数据、统计问题的单候选和多候选情况，将数据分为单候选和多候选部分，以及将有标签的数据生成为带有标签的形式。

#### 2. 具体功能
- `repair_program_io`：修复交互式代码的输入输出格式，将其转换为常规代码格式。根据代码的行特征，将代码分为不同的块，并返回修复后的代码字符串和代码块列表。
- `PythonParser`：解析Python代码，提取代码中的变量名和标记化的代码列表。如果解析失败，则尝试修复代码后再次解析，如果仍然失败，则使用启发式方法获取变量名。
- `revert_abbrev`：恢复代码中的缩略词，将其转换为完整形式。使用字典存储缩略词和对应的完整形式，并在代码中进行替换操作。
- `get_wordpos`：根据词性标记获取对应的WordNet词性，用于词性还原和词干提取。
- `process_nl_line`：对自然语言句子进行去冗处理，包括恢复缩略词、去除多余的空格和换行符、驼峰命名转换为下划线命名、去除括号内的内容等操作。
- `process_sent_word`：对句子进行分词、标记化、词性标注、词形还原和词干提取等操作，返回处理后的单词列表。
- `filter_invachar`：根据正则表达式模式去除句子中的非法字符。使用正则表达式匹配非常用字符，并将其替换为空格。
- `python_code_parse`：对Python代码进行解析，提取代码中的标记化的代码列表。根据代码的语法规则和标点符号，将代码分割为单词，并进行标准化和词干提取等操作。
- `python_query_parse`：对自然语言查询进行解析，提取查询中的单词列表。首先去除非法字符和括号，然后进行去冗处理、分词、标记化、词性标注、词形还原和词干提取等操作。
- `python_context_parse`：对自然语言上下文进行解析，提取上下文中的单词列表。首先去除非法字符，然后进行去冗处理、分词、标记化、词性标注、词形还原和词干提取等操作。

---

### sqlang_structured.py文件

#### 1. 概述
完成一个SQL语言解析器的功能，用于对SQL代码进行解析和处理。

#### 2. 具体功能
- `sanitizeSql`：用于对输入的SQL语句进行预处理，去除多余的空格和符号，并在语句末尾添加分号。
- `tokenizeRegex`：使用正则表达式模式将字符串进行分词，返回分词后的结果。
- `removeWhitespaces`：去除SQL语句中的空白符号。
- `identifySubQueries`：识别子查询并将其标记为特定类型。
- `identifyLiterals`：识别SQL语句中的关键字、字面量和其他标记，并将其相应地标记为特定类型。
- `identifyFunctions`：识别SQL语句中的函数，并将其标记为特定类型。
- `identifyTables`：识别SQL语句中的表和列，并将其相应地标记为特定类型。
- `parseStrings`：解析SQL语句中的字符串，并将其标记为特定类型。
- `renameIdentifiers`：对标识符（表名、列名）进行重命名，以避免冲突。
- `process_nl_line`：用于对自然语言句子进行处理，包括恢复缩写、去除多余的空格和括号，并去除句子末尾的点号。
- `get_wordpos`：根据词性标注的结果，返回对应的词性。
- `process_sent_word`：对句子进行分词、词性标注、词性还原和词干提取，返回处理后的单词列表。
- `filter_all_invachar`：用于过滤句子中的非常用符号。
- `filter_part_invachar`：用于过滤代码中的非常用符号。
- `sqlang_code_parse`：用于解析SQL代码并返回代码中的标记列表。
- `sqlang_query_parse`：用于解析SQL查询句子并返回句子中的标记列表。
- `sqlang_context_parse`：用于解析SQL上下文句子并返回句子中的标记列表。

---

### word_dict.py文件

#### 1. 概述
该代码用于构建词典，通过遍历语料库中的数据，将所有单词添加到一个集合中，从而构建词汇表。在构建最终词汇表时，首先加载已有的词汇表，然后获取新的词汇表，并找到新的单词。最后，将新的单词保存到最终词汇表文件中。

#### 2. 具体功能
- `get_vocab`：根据给定的两个语料库，获取词汇表。该函数遍历语料库中的数据，并将所有单词添加到一个集合中，最终返回词汇表。
- `load_pickle`：从pickle文件中加载数据并返回。
- `vocab_processing`：用于处理语料库文件和保存词汇表的文件路径。该函数调用load_pickle()函数加载语料库数据，然后调用get_vocab()函数获取词汇表，并将词汇表保存到指定的文件路径中。
- `final_vocab_processing`：首先从文件中加载已有的词汇表，然后调用get_vocab()函数获取新的词汇表。将新的词汇表与已有词汇表进行比较，找到新的单词，并将其保存到指定的文件路径中。
