---
title: "Datawhale AI 夏令营"
author: "BlackJack"
date: "2024.8.14"
output:
  html_document:
  toc: true
---

## 前言

**之前用百度的千帆训练过一次大模型，当时使用了RAG，这次来探索下使用讯飞的大模型进行训练。**

**也是第一次参加Datawhale的AI夏令营，有什么写的不好或者有问题的欢迎指出！**
**代码的注释copy了Datawhale的文档，不过大部分都在下面写了些我自己的理解**

*Q: 为什么需要微调*

> 大模型需要微调的主要原因是为了在特定任务或领域中提高模型的性能。虽然预训练的大模型（如 GPT、BERT 等）在广泛的文本语料库上进行了训练，具有强大的通用语言理解能力，但它们可能并不完全适应特定任务或特定领域的数据。因此，微调过程应运而生，旨在进一步调整模型参数，使其更好地适应特定任务的需求。
> 具体来说，微调有以下几个主要原因：

 1. **适应特定任务**
   大模型通常是在大规模的通用数据集上进行预训练的，这些数据可能包括各种各样的文本内容。然而，实际应用中，任务的要求通常是特定的。例如，预训练的大模型可能善于理解通用文本，但如果任务是法律文本的分类或医学文本的情感分析，直接使用预训练模型的效果可能不佳。微调通过在特定任务的数据集上进一步训练模型，使其能够更好地执行该任务。
 2. **提高模型的准确性**
   微调可以显著提高模型在特定任务上的性能。通过在目标任务的数据集上进行微调，模型能够更好地捕捉任务相关的特征和模式，从而提高预测的准确性。例如，在图像分类任务中，微调可以让模型更好地识别特定类别的图像。
 3. **减少计算成本**
   微调比从头训练一个新模型所需的计算资源要少得多。预训练一个大模型往往需要大量的数据和计算能力，而微调通常只需要在较小的数据集上进行少量训练。因此，微调可以在大幅度提高特定任务性能的同时，节省计算成本。
 4. **适应特定领域的术语或风格**
   不同领域的数据可能有不同的术语、表达方式或风格。预训练的大模型可能不完全掌握这些特定领域的语言特性。通过微调，模型可以适应特定领域的语言习惯，从而提高对领域特定任务的处理能力。
 5. **避免灾难性遗忘**
   微调有助于在特定任务上保持模型的通用能力，同时增强特定任务的性能。它能够在不显著丧失预训练时学习的通用知识的前提下，专注于提高模型在特定任务上的表现。

*Q：常见的微调方法有哪些？* (这部分不太懂问GPT的，不过今天看到task2发现里面写了，那里写得更好)
常见的微调方法主要包括以下几种：

1. **全模型微调**
   - **概述**: 对预训练模型的所有参数进行重新训练。
   - **应用场景**: 适用于数据量充足的情况下，希望在特定任务上获得最佳性能。
2. **冻结部分层微调**
   - **概述**: 仅微调预训练模型的部分参数（通常是靠近输出层的几层），而冻结其他层的参数。
   - **应用场景**: 数据量较小或计算资源有限的情况下，适用于特定任务与预训练任务有一定相关性的场景。
3. **任务特定的微调**
   - **概述**: 在预训练模型的基础上，添加任务特定的层或模块（如线性层、注意力层等）进行微调。
   - **应用场景**: 适用于需要增强特定任务能力的场景，如文本分类、问答系统。
4. **逐层微调**
   - **概述**: 从输出层开始，逐渐解冻并微调模型的各层参数。
   - **应用场景**: 适用于需要逐步适应新任务的复杂任务。
5. **轻量微调**
   - **概述**: 通过微调部分参数或添加轻量模块（如 Adapter、LoRA）来减少计算成本。
   - **应用场景**: 适用于资源有限的场景，如移动设备或边缘计算。
6. **多任务微调**
   - **概述**: 在多个相关任务上同时微调模型，使其学习共享的特征。
   - **应用场景**: 适用于跨任务学习或需要在多个任务上同时表现良好的场景。

# 初识 baseline

进入链接报名后，点击运行一下：[星火大模型驱动阅读理解题库构建挑战赛](https://aistudio.baidu.com/projectdetail/8225663)

## 数据源

进入BML codelab，给了我们这些：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240812110440.png)
由于在赛题描述中说明数据集为科大讯飞的保密信息，故这里不对数据集进行展示，仅对格式进行说明：

>[!info] 训练集
> **英语**：共64条数据，数据格式为：阅读文本-选项-答案，数据源来自各年高考英语试题，粗略看了看，ABCD篇似乎都有
> **语文**：共64条数据，数据格式和数据源同上，内容主要为现代文阅读1和现代文阅读2，但与英语的数据集不同的是，语文的选项和答案里面还有主观题的部分。

>[!info] 测试集/测试样例数据
>语文和英语一样，只有客观题，数据格式均为 input-target(output)，其中target里面既有对应的问题、选项，还有相应答案。

## baseline内容

整个内容大致为：

- 数据处理
- 模型微调
- 本地测试
- 在线提交
  
### 数据处理

```python
!pip install pandas openpyxl
```

**解释**: `!` 是一个魔法命令，它让你可以在Jupyter Notebook中直接运行Shell命令。这样，用户可以一边写代码，一边安装必要的库。

- `pandas`：一个用于数据操作和分析的 Python 库，广泛应用于数据科学、金融分析、统计分析等领域。`pandas` 提供了高效的数据结构和数据分析工具，能够轻松处理和分析结构化数据（如表格数据）。详情用法可以看Datawhale的教程[**Joyful Pandas**](https://inter.joyfulpandas.datawhale.club/Home.html)
- `openpyxl` 一个用于读写 Excel 文件的 Python 库。它支持 Microsoft Excel 2010 及更高版本的 `.xlsx` 文件格式，允许你在 Python 程序中创建、读取、修改和保存 Excel 文件。`openpyxl` 是处理 Excel 表格的常用工具，适合需要自动化处理 Excel 数据的场景。

#### 测试与清洗代码

```python
# coding~

import pandas as pd
import re  # 正则表达式模块

# 读取Excel文件
df = pd.read_excel('训练集-语文.xlsx')
df = df.replace('．', '.', regex=True)
df = df.replace('（', '(', regex=True)

# 读取第二行（即第三行）“选项”列的内容
second_row_option_content = df.loc[2, '选项']

# 显示第二行“选项”列的内容
print(second_row_option_content)
```

`replace`函数：

1. `df.replace('．', '.', regex=True)`:
   - **功能**: 将 `DataFrame` 中的所有 `．`（全角的“点”）替换为 `.`（半角的“点”）。
   - **`regex=True`**: 表示使用正则表达式进行替换，尽管这里并不涉及复杂的正则表达式，但通过指定 `regex=True`，可以让 `replace` 方法更灵活地处理替换操作。
2. **`df.replace('（', '(', regex=True)`**:
   - **功能**: 将 `DataFrame` 中的所有 `（`（全角的“左括号”）替换为 `(`（半角的“左括号”）。
   - **`regex=True`**: 同样，这里使用正则表达式模式进行替换。

这些操作的主要目的是将 `DataFrame` 中的全角字符（常见于日文或中文文本）替换为对应的半角字符。全角字符通常比半角字符占据更多的空间，并且在处理国际化文本时，可能会遇到需要标准化这些字符的情况。使用 `replace` 方法可以快速地替换这些字符，从而实现数据的标准化。

#### 问题抽取

刚刚我们在数据格式里面发现语文试题的训练集里面含有主观题，但是测试集只要求给出客观题及答案，所以我们需要把主观题部分剔除，防止给模型带来不良影响
这里主要需要使用正则表达式模块`re`

```python
def chinese_multiple_choice_questions(questions_with_answers):
    # 输入的题目文本
    text = questions_with_answers
    
    question_pattern = re.compile(r'\d+\..*?(?=\d+\.|$)', re.DOTALL)
    # 这一行的作用是定义一个正则表达式模式，用于匹配题目。
    # 该模式匹配以数字开头并跟随一个点的字符串，直到下一个数字和点字符或者字符串结束。
    
    choice_pattern = re.compile(r'([A-D])\s*(.*?)(?=[A-D]|$|\n)', re.DOTALL)
    # 这一行定义了另一个正则表达式模式，用于匹配选择题的选项。
    # 该模式匹配以字母[A-D]开头的字符串，并在遇到下一个[A-D]、字符串结束符或者换行符时停止匹配。
    
    # 找到所有问题
    questions = question_pattern.findall(text)
    # 使用question_pattern从输入文本中找到所有的问题。
    # 这些问题包括选择题和简答题。

    # 初始化选择题和简答题列表
    multiple_choice_questions = []
    short_answer_questions = []

    # 处理每个问题
    for id, question in enumerate(questions):
        # 遍历提取的问题，每个问题都可能是选择题或简答题。
        
        # 检查是否是选择题，因为选择题中包含[A-D]的选项
        if re.search(r'[A-D]', question):
            # 如果题目中有[A-D]，则认为这是一个选择题。
            
            # 如果有选项，提取出选项的内容
            choices = choice_pattern.findall(question)
            # 使用choice_pattern从题目中提取出选项（A、B、C、D）及其对应的内容。
            
            # 这里提取了题目的内容，因为每个题目都会有一个类似“(X分)”的标记
            # 以左括号为目标，截取选择题选项中的内容
            question_text = re.split(r'\n', question.split('(')[0])[0]
            # 提取题目的文本部分，忽略后面的标记（如分数标记）。

            pattern_question = re.compile(r'(\d+)\.(.*)')
            # 定义了一个正则表达式，用于匹配题目编号和题目内容。
            
            # 这里清洗了选择题的编号，重新用循环中的id进行编号。
            # 如果不做这一步可以发现给定的数据中编号是乱序的。
            matches_question = str(id + 1) + '.' + pattern_question.findall(question_text)[0][1]
            # 将题目编号重新排序，并生成新的题目字符串。
            
            # 将每个问题和选项以字典的形式存入方便我们处理
            multiple_choice_questions.append({
                'question': matches_question,
                'choices': choices
            })
            # 将题目和选项以字典形式添加到multiple_choice_questions列表中。
            
        else:
            # 如果题目中没有[A-D]，则认为这是一个简答题。
            short_answer_questions.append(question.strip())
            # 将简答题去掉首尾空格后添加到short_answer_questions列表中。

    # 最后返回抽取后的选择题字典列表
    return multiple_choice_questions
    # 返回包含选择题的字典列表，每个字典包含题目和选项。
```

这段代码的作用是解析包含选择题和简答题的文本，提取出选择题的内容并返回一个包含题目和选项的字典列表。

`re`模块是Python中用于处理正则表达式的标准库。正则表达式是一种强大的字符串匹配工具，允许用户定义复杂的搜索模式，以便在文本中查找、替换或提取特定的内容。`re`模块提供了一系列函数，用于编译正则表达式、在字符串中查找匹配项、替换字符串中的匹配项等操作。

##### `re`模块的常用函数

1. **`re.compile(pattern, flags=0)`**:
   - 用于编译正则表达式，返回一个模式对象，可以在后续的匹配操作中重复使用。
   - `pattern`是正则表达式的字符串形式，`flags`是可选参数，用于控制匹配行为（如忽略大小写、多行匹配等）。
2. **`re.search(pattern, string, flags=0)`**:
   - 搜索整个字符串，返回第一个匹配的对象。如果没有找到匹配项，则返回`None`。
   - `pattern`是要匹配的正则表达式，`string`是要搜索的目标字符串。
3. **`re.findall(pattern, string, flags=0)`**:
   - 返回所有匹配的子串组成的列表。如果没有匹配项，则返回空列表。
   - 该方法会查找字符串中所有符合正则表达式的部分。
4. **`re.match(pattern, string, flags=0)`**:
   - 尝试从字符串的起始位置匹配一个模式，如果匹配成功，返回一个匹配对象，否则返回`None`。
5. **`re.split(pattern, string, maxsplit=0, flags=0)`**:
   - 根据正则表达式中的匹配项来拆分字符串，返回一个子串列表。
6. **`re.sub(pattern, repl, string, count=0, flags=0)`**:
   - 用于替换字符串中所有符合正则表达式的部分，用另一个字符串替换。

##### `re`模块在代码中的使用

在代码中，`re`模块用于匹配题目和选项：

1. **`question_pattern = re.compile(r'\d+\..*?(?=\d+\.|$)', re.DOTALL)`**:
   - 这一行使用`re.compile`编译了一个正则表达式，用于匹配题目。
   - `\d+\.`匹配数字后跟随一个点字符，`.*?`表示非贪婪匹配任意字符，直到`(?=\d+\.|$)`遇到下一个数字和点字符或字符串结束。
   - `re.DOTALL`是一个标志，使`.`能够匹配包括换行符在内的任意字符。
2. **`choice_pattern = re.compile(r'([A-D])\s*(.*?)(?=[A-D]|$|\n)', re.DOTALL)`**:
   - 这一行编译了另一个正则表达式，用于匹配选择题的选项。
   - `([A-D])`匹配字母A到D，`\s*`匹配任意数量的空白字符，`(.*?)`非贪婪匹配选项内容，直到遇到`(?=[A-D]|$|\n)`下一个选项字母、字符串结束符或换行符。
3. **`re.search(r'[A-D]', question)`**:
   - 这一行使用`re.search`在题目字符串中查找是否存在[A-D]，以判断该题目是否为选择题。
4. **`matches_question = str(id+1)+'.'+ pattern_question.findall(question_text)[0][1]`**:
   - 这行代码的作用是提取题目中的文本部分，并根据循环的索引（`id`）重新编号，然后将编号和题目内容拼接在一起。为了帮助你理解，我们先拆解这行代码并通过一个具体的例子来说明。
  
###### 代码拆解

假设我们有一个题目字符串：

```python
question_text = "12. What is the capital of France?"
```

匹配过程：

1. **定义正则表达式**:

   ```python
   pattern_question = re.compile(r'(\d+)\.(.*)')
   ```

2. **使用 `findall` 提取编号和内容**:

   ```python
   matches = pattern_question.findall(question_text)
   # matches = [('12', ' What is the capital of France?')]
   ```

    这里，`matches` 是一个包含一个元组的列表，元组的第一个元素是`12`（题号），第二个元素是`What is the capital of France?`（题目内容）。

3. **提取题目内容**:
  
   ```python
   question_content = matches[0][1]  # ' What is the capital of France?'
   ```

    这里提取了题目的内容部分。

4. **重新编号并拼接**:

   ```python
   id = 0  # 假设这是循环的第一次
   matches_question = str(id + 1) + '.' + question_content
   # matches_question = '1. What is the capital of France?'
   ```

最终，`matches_question` 的值是 `'1. What is the capital of France?'`，即将原本题目的编号`12`替换为`1`。

#### 贪婪匹配与非贪婪匹配

在正则表达式中，**贪婪匹配**（greedy matching）和**非贪婪匹配**（non-greedy matching，也称为懒惰匹配，lazy matching）是两种不同的匹配策略，它们决定了正则表达式在匹配时获取多少内容。

##### 贪婪匹配（Greedy Matching）

贪婪匹配尽可能多地匹配字符。这意味着当有多个可能的匹配时，贪婪模式会尝试匹配尽可能长的字符串。

###### 例子

```python
import re

text = "abc123def456"

# 贪婪匹配
match = re.search(r'\d+.*\d+', text)
print(match.group())  # 输出：123def456
```

在这个例子中，`\d+`匹配一个或多个数字字符，`.*`匹配任意数量的任意字符（包括0个），所以贪婪匹配会扩展匹配范围，直到遇到最后一个数字字符。

##### 非贪婪匹配（Non-greedy Matching）

非贪婪匹配尽可能少地匹配字符，即在能满足匹配条件的情况下，会尝试匹配尽可能短的字符串。在正则表达式中，通常通过在量词后面添加`?`来实现非贪婪匹配。

###### 例子2

```python
import re

text = "abc123def456"

# 非贪婪匹配
match = re.search(r'\d+?.*?\d+?', text)
print(match.group())  # 输出：123def4
```

在这个例子中，`+?`和`*?`是非贪婪的，它们会尽量少地匹配内容。因此，`123`匹配了第一个数字序列，`.*?`匹配了`def`，最后的`\d+?`只匹配了数字`4`。

###### 匹配过程

1. **首先匹配`\d+?`**：
   - `\d+?`在"123def456"中，从左到右扫描，首先遇到`1`，这是一个数字。因为非贪婪匹配会尽量少地匹配字符，所以它首先尝试只匹配`1`。
   - 然而，正则表达式的其余部分（`.*?\d+?`）需要继续匹配，为了满足整个模式，`1`不足以使整个正则表达式匹配成功。
   - 因此，`+?`扩展了匹配范围，包含了`2`，然后`3`，最终`123`被匹配为第一个`\d+?`，因为这仍然满足尽量少的匹配原则，同时使后续部分有机会成功匹配。
2. **然后匹配`.*?`**：
   - 接下来，`.*?`开始匹配，尽量少地匹配字符。它从`d`开始匹配，并逐步扩展到匹配`def`。
   - `.*?`匹配到`def`后，它会停下尝试，因为后面的`\d+?`需要匹配一个数字。
3. **最后匹配第二个`\d+?`**：
   - 此时，剩下的字符串是`456`。第二个`\d+?`是非贪婪匹配，它尽量少地匹配数字字符。因此，它只匹配了`4`，因为这已经足够满足整个正则表达式的匹配。

###### 总结

- 第一个`\d+?`非贪婪匹配最终匹配了`123`，因为这是最少的匹配量，能使后续部分也成功匹配。
- 第二个`\d+?`非贪婪匹配只匹配了`4`，因为它只需要匹配一个数字来满足正则表达式的整体要求。

#### 抽取问题结果

这部分也是用正则表达式进行处理，不过对于中文试题由于存在简答题，所以需要进行些特别处理：

```python
def chinese_multiple_choice_answers(questions_with_answers):
    questions_with_answers = questions_with_answers.replace(" ", "").replace("\n", "")
    
    print(questions_with_answers)
    print("\n")
    # 使用正则表达式匹配答案
    choice_pattern = re.compile(r'(\d+)\.([A-Z]+)')
    short_pattern = re.compile(r'(\d+)\.([^A-Z]+)')

    # 找到所有匹配的答案
    choice_matches = choice_pattern.findall(questions_with_answers)
    short_matches = short_pattern.findall(questions_with_answers)

    # 将匹配结果转换为字典
    choice_answers = {int(index): answer for index, answer in choice_matches}
    short_answers = {int(index): answer for index, answer in short_matches}

    # 按序号重新排序
    sorted_choice_answers = sorted(choice_answers.items())
    sorted_short_answers = sorted(short_answers.items())
    
    answers = []

    # 输出结果
    
    print("选择题答案：")
    for id in range(len(sorted_choice_answers)):
        answers.append(f"{id+1}. {sorted_choice_answers[id][1]}")
    return answers
```

这里正则表达式的解释：

- `choice_pattern`: 用于匹配选择题答案，例如 "1.A"、"2.BC" 等。`\d+` 匹配数字序号，`[A-Z]+` 匹配一个或多个大写字母（即答案）。
- `short_pattern`: 用于匹配其他类型的答案，例如填空题或简答题，这些答案中不会有大写字母。这里的 `[^A-Z]+` 匹配**除了大写字母以外的任意字符**。
`findall`返回的是一个包含元组的列表，每个元组包含两个元素：`index`和`answer`
例如：

```python
choice_matches = [('1', 'A'), ('2', 'BC'), ('3', 'AB')]
```

因此列表推导式创建的字典对应的形式为：

```python
choice_answers = {1: 'A', 2: 'BC', 3: 'AB'}
```

遍历列表时候，将元组的第一个元素赋值给`index`，第二个元素赋值给`answer`
测试一下是否能真的提取:
![image.png|550](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814102354.png)

#### 对答案列进行处理

```python
df['答案_processed'] = df['答案'].map(chinese_multiple_choice_answers)
```

这句话是使用 Pandas 的 `map` 函数对数据框 `df` 中的某一列 `答案` 进行处理，并将处理后的结果存储在新列 `答案_processed` 中。

- `df['答案']`: 这是数据框 `df` 中名为 `答案` 的列。假设这一列包含了多个问题的答案字符串。
- `map(chinese_multiple_choice_answers)`: `map` 函数会将 `chinese_multiple_choice_answers` 这个函数应用到 `df['答案']` 列的每一个元素上。`map` 会逐个遍历 `df['答案']` 列中的每一个值，并将这个值作为参数传递给 `chinese_multiple_choice_answers` 函数，然后将函数的返回结果赋给对应的行。
- `df['答案_processed']`: 这表示创建或更新数据框 `df` 中的一个新列，名为 `答案_processed`。这个新列将存储 `chinese_multiple_choice_answers` 函数对 `df['答案']` 中每一个元素处理后的结果。

假设 `df['答案']` 列中的数据如下：

```plaintext
1.A 2.BC 3.AB
4.D 5.AC 6.B
```

在应用 `map(chinese_multiple_choice_answers)` 后，`chinese_multiple_choice_answers` 函数会处理每一行的答案字符串，提取并格式化选择题答案，然后将处理后的结果存储到新列 `答案_processed` 中。
最终，`df` 可能会变成这样：

| 答案          | 答案_processed        |
| ------------- | -------------------- |
| 1.A 2.BC 3.AB | [1. A, 2. BC, 3. AB] |
| 4.D 5.AC 6.B  | [4. D, 5. AC, 6. B]  |

这里，`答案_processed` 列中每个元素都是 `chinese_multiple_choice_answers` 函数处理后的结果，通常是一个包含格式化答案的列表。

#### prompt工程

```Python
def get_prompt_cn(text):
    prompt = f'''
    你是⼀个⾼考选择题出题专家，你出的题有⼀定深度，你将根据阅读文本，出4道单项选择题，包含题目选项，以及对应的答案，注意：不⽤给出原文，每道题由1个问题和4个选项组成，仅存在1个正确答案，请严格按照要求执行。 阅读文本主要是中文，你出的题目需要满足以下要点，紧扣文章内容且题干和答案为中文：
    
    ### 回答要求
    (1)理解文中重要概念的含义
    (2)理解文中重要句子的含意
    (3)分析论点、论据和论证方法
    
    
    ### 阅读文本
    {text}
    '''
    
    return prompt   
```

最开始的prompt不需要更改，可以对回答要求部分进行尝试修改，阅读文本实际上是我们的阅读材料

#### 中文处理主函数

```python
def process_cn(df): 
    # 定义好返回列表
    res_input = []
    res_output = []

    for id in range(len(df)):
        # 逐个遍历每行的选项、答案、阅读文本的内容
        data_options = df.loc[id, '选项']
        data_answers = df.loc[id,'答案']
        data_prompt = df.loc[id,'阅读文本']
        # 处理选项部分，抽取出选择题题目及选项
        data_options = chinese_multiple_choice_questions(data_options)
        # 处理答案部分，抽取出选择题答案
        data_answers = chinese_multiple_choice_answers(data_answers)
        # 抽取阅读材料组合成input内容
        data_prompt = get_prompt_cn(data_prompt)
        # print(data_options)
        # print(data_answers)
        # 做数据验证，因为训练数据格式不能确定每组数据都能被正常处理（会有一部分处理失败）
        # 我们验证一下两个列表的长度 如果相同代表数据处理正确
        if(len(data_answers)==len(data_options)):
            # 定义output的数据字符串
            res = ''
            # 处理选择题目中的每个数据，逐个拼入到output字符串
            for id_,question in enumerate(data_options):
            # 首先放入题目
                res += f'''
{question['question']}?
                '''+'\n'
                # 然后找到选择题的每个选项，进行choices列表循环
                for choise in question['choices']:
                # 逐个将选项拼接到字符串
                    res = res+ choise[0] + choise[1]+ '\n'
                #  最后将答案拼接到每个选择题的最后
                # 以 答案：题号.选项的格式
                res = res + '答案:' + str(data_answers[id_].split('.')[-1])  + '\n'
            # 最后将处理得到的input、output数据存入到列表
            res_output.append(res)
            res_input.append(data_prompt)
        # break
    return res_input,res_output
```

将`res`打印出来是这样的：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814104500.png)
可以看到实现了题目编号重新排序，选项重新排序，答案贴在题目和选项后面这三个功能

### 英语处理

大部分跟中文处理一样，只有一点点微调
一开始处理的时候看到有个把A换成A的。。。实在没懂为什么这么做，看note才知道原来那个是OCR识别的俄文。。。好吧也是学到了

```Python
import pandas as pd

# 读取Excel文件
df = pd.read_excel('训练集-英语.xlsx')
# 英文数据处理中有一部分ocr识别的题目，这种题目中看上去是字母A，但是实际为俄文的字母，，
# 所以开始使用全局匹配做了清洗……
df = df.replace('．', '.', regex=True).replace('А.', 'A.', regex=True).replace('В.', 'B.', regex=True).replace('С.', 'C.', regex=True).replace('D.', 'D.', regex=True)
# df = df.replace('（', '(', regex=True)

# 读取第二行（即第三行）“选项”列的内容
second_row_option_content = df.loc[0, '选项']

# 显示第二行“选项”列的内容
print(second_row_option_content)
```

#### 英文问题抽取

这里需要处理的是英语中出现ACBD这种选项乱序的问题

```Python
import re

# 示例文本
text = second_row_option_content

def get_questions(text):
    # 数据清洗，将所有换行改为两个空格方便统一处理
    text = text.replace('\n', '  ')+'  '
    # print(text)
    # 正则表达式模式
    # 通过匹配以数字开头然后带一个点，为题干
    # 然后抽取选项A  以A开头 后面带一个点 最后以两个空格结尾
    # 为什么是两个空格？部分数据换行时为换行符，我们已经换成了两个空格，有些是以多个空格分割，我们默认为两个空格
    # 接着匹配B C D选项内容
    # 最后有一个
    pattern = re.compile(r'(\d+\..*?)(A\..*?\s{2})([B-D]\..*?\s{2})([B-D]\..*?\s{2})(D\..*?\s{2})', re.DOTALL)

    # 查找所有匹配项
    matches = pattern.findall(text)

    # 存储结果的字典列表
    questions_dict_list = []

    # 打印结果
    for match in matches:
        question, option1, option2, option3, option4 = match
        pattern_question = re.compile(r'(\d+)\.(.*)')
        # 第一个为选择题的题目 提前存到question_text 
        question_text = pattern_question.findall(question.strip())[0][1]
        
        # 提取选项字母和内容
        options = {option1[0]: option1, option2[0]: option2, option3[0]: option3, option4[0]: option4}
        
        question_dict = {
            'question': question_text,
            # 这一步就是防止ACBD这种乱序，我们进行重新匹配，将可能是ACBD的数据以首字母按位置排好号
            'options': {
                'A': options.get('A', '').strip(),
                'B': options.get('B', '').strip(),
                'C': options.get('C', '').strip(),
                'D': options.get('D', '').strip()
            }
        }
        
        questions_dict_list.append(question_dict)
    # 最后获得
    return questions_dict_list

# 调用函数并打印结果
questions = get_questions(text)
for q in questions:
    print(q)
```

看看发生了什么：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814110230.png)
第一个框内容是最原始的数据，发现会有ACBD乱码
于是先对数据进行简单处理，去掉换行符，得到第二个框内容(注意这一步处理的时候添加了两个空格在每个选项间，后面还需要对这个进行处理)
用正则表达式进行处理：

```python
pattern = re.compile(r'(\d+\..*?)(A\..*?\s{2})([B-D]\..*?\s{2})([B-D]\..*?\s{2})(D\..*?\s{2})', re.DOTALL)
```

- `\d+\.`: 匹配题号，例如 "1." 或 "23."。
- `.*?`: 匹配题号后的题干内容，`?` 是非贪婪匹配。
- `A\..*?\s{2}`: 匹配选项 A，`A\.` 匹配 "A."，`.*?` 匹配选项内容，`\s{2}` 表示两个空格作为选项之间的分隔符。
- `[B-D]\..*?\s{2}`: 匹配选项 B, C, D（D 是最后一个选项，所以不会匹配后续的空格）。
- `re.DOTALL` 标志用于让 `.` 可以匹配换行符，这在处理包含换行的题干时很有用。
随后进行匹配，得到如下结果：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814110932.png)
列表中的每个元素都是一个元组，包含一个匹配的选择题和它的选项。

再对内部内容进行进一步处理：

```python
for match in matches:
 question, option1, option2, option3, option4 = match
 pattern_question = re.compile(r'(\d+)\.(.*)')
 question_text = pattern_question.findall(question.strip())[0][1]
```

- 遍历 `matches` 列表中的每个匹配项 `match`，解包为 `question`（题干）和四个选项。
- 使用正则表达式 `pattern_question` 提取题干中的题号和文本内容，将题号过滤掉，保留纯粹的题目内容。
  
```python
options = {option1[0]: option1, option2[0]: option2, option3[0]: option3, option4[0]: option4}
```

建立字典，其中键是选项字母（A, B, C, D），值是对应的选项内容。
为啥是这样？举个例子：

```python
match = (
    '21. What is an advantage of MacBike?', 
    'A. It gives children a discount.', 
    'B. It offers many types of bikes.', 
    'C. It organizes free cycle tours.', 
    'D. It has over 2,500 rental shops.  '
)
```

在这段代码中，`option1`, `option2`, `option3`, `option4` 分别对应 `match` 中的四个选项：

- `option1 = 'A. It gives children a discount.'`
- `option2 = 'B. It offers many types of bikes.'`
- `option3 = 'C. It organizes free cycle tours.'`
- `option4 = 'D. It has over 2,500 rental shops. '`
  
这段代码的目的是创建一个字典 `options`，其中键是选项的标识符（A, B, C, D），而值是完整的选项文本。让我们来看看这段代码是如何工作的：

- **`option1[0]: option1`**:
  - `option1[0]` 提取选项 `option1` 的第一个字符，即 `'A'`。
  - `option1` 是完整的选项字符串 `'A. It gives children a discount.'`。
  - 所以字典中的键值对为 `'A': 'A. It gives children a discount.'`。
后面的同理，于是得到最终的字典如下：

```python
options = {
    'A': 'A. It gives children a discount.',
    'B': 'B. It offers many types of bikes.',
    'C': 'C. It organizes free cycle tours.',
    'D': 'D. It has over 2,500 rental shops.'
}
```

调整可能出现的乱序：

```python
question_dict = {
    'question': question_text,
    # 这一步就是防止ACBD这种乱序，我们进行重新匹配，将可能是ACBD的数据以首字母按位置排好号 
    'options': { 
        'A': options.get('A', '').strip(), 
        'B': options.get('B', '').strip(), 
        'C': options.get('C', '').strip(), 
        'D': options.get('D', '').strip() 
    } 
}
```

- **`options.get('A', '')`**:
  - `options` 是一个字典，存储了选项的标识符（例如 `'A'`、`'B'`、`'C'`、`'D'`）及其对应的文本内容。
  - `get('A', '')` 尝试从字典 `options` 中获取键 `'A'` 对应的值。
  - 如果 `options` 字典中存在键 `'A'`，就会返回其对应的值（例如 `'A. It gives children a discount.'`）。
  - 如果 `options` 字典中不存在键 `'A'`，则返回一个默认值，这里是空字符串 `''`。
  
- **`strip()`**:
  - `strip()` 方法用于去除字符串前后的空白字符。
  - 假设 `options.get('A', '')` 返回的是 `'A. It gives children a discount. '` （注意末尾有一个空格），使用 `strip()` 后，字符串会变成 `'A. It gives children a discount.'`。

#### 英文处理主函数

抽取问题结果和prompt设置没有太多特别的，这里不作阐述，介绍英文处理方法

```Python
def process_en(df): 
    res_input = []
    res_output = []
    for id in range(len(df)):
        data_options = df.loc[id, '选项']
        data_answers = df.loc[id,'答案']
        data_prompt = df.loc[id,'阅读文本']
        data_options = get_questions(data_options)
        data_answers = get_answers(data_answers)
        data_prompt = get_prompt_en(data_prompt)
        # print(data_options)
        # print(data_answers)

        if(len(data_answers)==len(data_options)):
            res = ''
            for id,question in enumerate(data_options):
                res += f'''
                {id+1}.{question['question']}
                {question['options']['A']}
                {question['options']['B']}
                {question['options']['C']}
                {question['options']['D']}
                answer:{data_answers[id]}
                '''+'\n'
            res_output.append(res)
            res_input.append(data_prompt)
    return res_input,res_output
    # break
```

最终处理结果展示如下：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814113028.png)

### 数据合并

因为微调数据只能导入jsonl或者csv文件，所以我们需要将原本的数据进行合并
微调需要150条数据，数据处理后得到有效数据为102，从中文抽取30条，英文抽取20条组成152条数据作为微调数据(就是说有的数据是重复的)。

```python
# 将两个列表转换为DataFrame
df_new = pd.DataFrame({'input': cn_input+cn_input[:30]+en_input+en_input[:20], 'output': cn_output+cn_output[:30]+en_output+en_output[:20]})
```

### 导出jsonl文件

```python
import json

# 打开一个文件用于写入 JSONL，并设置编码为 UTF-8
with open('output.jsonl', 'w', encoding='utf-8') as f:
    # 遍历每一行并将其转换为 JSON
    for index, row in df_new.iterrows():
        row_dict = row.to_dict()
        row_json = json.dumps(row_dict, ensure_ascii=False,)
        # 将 JSON 字符串写入文件，并添加换行符
        f.write(row_json + '\n')
```

- **`with open('output.jsonl', 'w', encoding='utf-8') as f:`**
  - 以写模式 (`'w'`) 打开文件 `output.jsonl`。如果文件不存在，将会创建它。
  - `encoding='utf-8'` 指定文件的编码为 UTF-8，这对于处理包含非 ASCII 字符的文本是很重要的。
  - 使用 `with` 语句来确保在操作完成后自动关闭文件。
- **`for index, row in df_new.iterrows():`**
  - `df_new.iterrows()` 是 `pandas` 提供的一个方法，用于逐行遍历 DataFrame。它返回一个迭代器，生成 `(index, row)` 对，其中 `index` 是当前行的索引，`row` 是一个包含该行数据的 `pandas` Series 对象。
- **`row_dict = row.to_dict()`**
  - 将 `row`（一个 `pandas` Series 对象）转换为字典 `row_dict`。字典的键是列名，值是该行对应的单元格值。
- **`row_json = json.dumps(row_dict, ensure_ascii=False,)`**
  - 使用 `json.dumps()` 方法将 `row_dict` 转换为 JSON 字符串 `row_json`。`ensure_ascii=False` 参数确保生成的 JSON 字符串可以包含非 ASCII 字符（如中文字符），而不是将它们转义为 Unicode 码点。
- **`f.write(row_json + '\n')`**
  - 将 JSON 字符串 `row_json` 写入到文件 `f` 中，并在每个 JSON 字符串的末尾添加换行符 `'\n'`。这样每一行的 JSON 数据将单独占据文件的一行。
这样就输出了jsonl文件，将这个文件丢到(<https://training.xfyun.cn/dataset/datasetIndex)即可进行微调>

#### jsonl和json的区别

JSON（JavaScript Object Notation）和 JSONL（JSON Lines）都是用来存储和传输结构化数据的格式，但它们有一些显著的区别。

##### JSON 文件

- **格式**：JSON 文件是一种结构化的文本文件，整个文件内容是一个合法的 JSON 对象或数组。它的结构通常是嵌套的，包含键值对、数组等，可以存储复杂的层次化数据。

  **示例：**

  ```json
  {
      "name": "John Doe",
      "age": 30,
      "city": "New York",
      "children": [
          {
              "name": "Jane Doe",
              "age": 10
          },
          {
              "name": "Jake Doe",
              "age": 7
          }
      ]
  }
  ```

- **用途**：JSON 文件通常用于存储单个对象或对象数组，适合用作配置文件、API 响应格式、数据传输等。

##### JSONL 文件

- **格式**：JSONL（JSON Lines）文件是一种文本文件，其中每一行都是一个独立的 JSON 对象。不同于标准的 JSON 文件，JSONL 文件没有根对象或数组，文件中的每一行都是一个独立的 JSON 对象。

  **示例：**

  ```json
  {"name": "John Doe", "age": 30, "city": "New York"}
  {"name": "Jane Doe", "age": 10, "city": "New York"}
  {"name": "Jake Doe", "age": 7, "city": "New York"}
  ```

- **用途**：JSONL 文件适合存储和处理大量结构相同的记录（比如日志文件、数据集等）。每一行的数据都可以独立处理，这使得 JSONL 格式在流式处理、大规模数据集等场景中特别有用。它也可以方便地逐行读取和写入数据，而无需加载整个文件。
  
##### 区别总结

1. **结构**：
   - **JSON** 文件是一个完整的 JSON 对象或数组，适合存储嵌套的数据结构。
   - **JSONL** 文件由多行独立的 JSON 对象组成，适合存储结构相同的多条记录。
2. **读取方式**：
   - **JSON** 文件通常需要一次性读取整个文件，才能解析为一个 JSON 对象或数组。
   - **JSONL** 文件可以逐行读取，每一行都是一个独立的 JSON 对象，可以流式处理。
3. **使用场景**：
   - **JSON** 文件适合存储复杂、嵌套的配置、API 响应等。
   - **JSONL** 文件适合存储大量独立的记录，尤其在大数据处理或日志文件中。

## 大语言模型介绍

### 大语言模型的概念

**大语言模型（英文：Large Language Model，缩写LLM**）也称大型语言模型，是一种人工智能模型，旨在理解和生成人类语言。
通常，大语言模型 (LLM) 指包含数 **十亿**（**Billion**或更多）参数的语言模型，这些模型在大量的文本数据上进行训练，例如国外的有GPT-3 、GPT-4、PaLM 、Galactica 和 LLaMA 等，国内的有ChatGLM、文心一言、通义千问、讯飞星火等。

### 大模型的能力和特点

1. **大模型的能力**
    大语言模型（LLM）与以前的预训练语言模型（PLM）的主要区别在于其涌现能力。这种能力在小型模型中不明显，但在大型模型中显著。例如：

    - **上下文学习**：首次由GPT-3引入，允许模型在提供自然语言指令或多个任务示例的情况下，通过理解上下文并生成相应输出来执行任务。
    - **指令遵循**：通过指令微调，LLM可以根据任务指令执行未见过的任务，展示出强大的泛化能力。
    - **逐步推理**：通过"**思维链（Chain of Thought, CoT）**"策略，LLM能够解决多步推理任务，例如数学问题。

2. **大模型的特点**
   - 巨大的规模：参数规模达数十亿甚至数千亿，使其能捕捉更多语言知识和复杂语法结构。
   - 预训练和微调：在大规模无标签文本数据上预训练，然后通过有标签数据微调，适应特定任务。
   - 上下文感知：具备强大的上下文感知能力，能够理解和生成依赖前文的文本内容。
   - 多语言支持：支持多种语言，促进跨文化和跨语言的应用。
   - 多模态支持：一些LLM支持文本、图像和语音的多模态数据。
   - 涌现能力：在大规模模型中表现出明显的性能提升，能处理更复杂的任务。
   - 多领域应用：广泛应用于文本生成、自动翻译、信息检索、摘要生成、聊天机器人等多个领域。
   - 伦理和风险问题：需要谨慎处理生成有害内容、隐私问题和认知偏差等伦理和风险问题。

## 微调介绍

### 什么是模型微调？

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814121425.png)
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814121511.png)

相当于给你一个预训练模型（Pre-trained model），基于这个模型微调（Fine Tune）。
**预训练模型** **就是** 已经用数据集训练好了的模型。

### 两种 Finetune 范式

1. 增量预训练微调 (***Continue PreTraining***)
使用场景：让基座模型学习到一些新知识，如某个垂类领域的常识
训练数据：文章、书籍、代码等
2. 指令跟随微调 (***Supervised Finetuning***)
使用场景：让模型学会对话模板，根据人类指令进行对话
训练数据：高质量的对话、问答数据

### 为什么要*微调*？

相对于从头开始训练(Training a model from scatch)，微调可以**省去大量计算资源和计算时间**，提高了计算效率,甚至提高准确率。
**普通预训练模型**的特点是：用了大型数据集做训练，已经具备了*提取浅层基础特征和深层抽象特征*的能力。
**不做微调**：
（1）从头开始训练，需要大量的数据，计算时间和计算资源。
（2）存在模型不收敛，参数不够优化，准确率低，模型泛化能力低，容易过拟合等风险。
**使用微调**：避免了上述可能存在的问题。

### 什么情况下使用*微调*？

（1） 你要使用的数据集和预训练模型的**数据集相似**
如果不太相似，效果可能就没有那么好了，特征提取是不同的，所以相应的参数训练后也是不同的。
（2）自己搭建或者使用的模型正确率太低。
（3）数据集相似，但数**据集数量太少**。
（4）**计算资源太少**。

### 不同数据集下使用微调

- 数据集1 - **数据量少，但数据相似度非常高**在这种情况下，我们所做的只是**修改最后几层**或最终的softmax图层的输出类别。
- 数据集2 - **数据量少，数据相似度低**在这种情况下，我们可以**冻结预训练模型的初始层**（比如k层），并再次训练剩余的（$n-k$）层。由于新数据集的相似度较低，因此根据新数据集对较高层进行重新训练具有重要意义。
- 数据集3 - **数据量大，数据相似度低**在这种情况下，由于我们有一个大的数据集，我们的神经网络训练将会很有效。但是，由于我们的数据与用于训练我们的预训练模型的数据相比有很大不同。使用预训练模型进行的预测不会有效。因此，最好根据你的数据**从头开始训练**神经网络（Training from scatch）。
- 数据集4 - **数据量大，数据相似度高**这是理想情况。在这种情况下，预训练模型应该是最有效的。使用模型的最好方法是保留模型的体系结构和模型的初始权重。然后，我们可以使用在预先训练的模型中的权重来重新训练该模型。
  
### 微调指导事项

1. 通常的做法是截断预先训练好的网络的*最后一层*（softmax层），并用与我们自己的问题相关的新的softmax层替换它。例如，ImageNet上预先训练好的网络带有1000个类别的softmax图层。如果我们的任务是对*10个类别*的分类，则网络的新softmax层将由10个类别组成，而不是1000个类别。然后，我们在网络上运行预先训练的权重。确保执行交叉验证，以便网络能够很好地推广。
2. 使用*较小的学习率*来训练网络。由于我们预计预先训练的权重相对于随机初始化的权重已经相当不错，我们不想过快地扭曲它们太多。通常的做法是使*初始学习率*比用于从头开始训练（Training from scratch）的初始学习率*小10倍*。
3. 如果数据集数量过少，我们进来只训练最后一层，如果数据集数量中等，*冻结预训练网络的前几层的权重也是一种常见做法*。

> 这是因为前几个图层捕捉了与我们的新问题相关的通用特征，如曲线和边。我们希望保持这些权重不变。相反，我们会让网络专注于学习后续深层中特定于数据集的特征。

### LoRA

LoRA是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)。

#### LoRA 的优势

- 可以针对不同的下游任务构建小型 LoRA 模块，从而在共享预训练模型参数基础上有效地切换下游任务。
- LoRA 使用自适应优化器（Adaptive Optimizer），不需要计算梯度或维护大多数参数的优化器状态，训练更有效、硬件门槛更低。
- LoRA 使用简单的线性设计，在部署时将可训练矩阵与冻结权重合并，不存在推理延迟。
- LoRA 与其他方法正交，可以组合。

#### LoRA 的原理

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240814121536.png)

- <https://github.com/microsoft/LoRA?tab=readme-ov-file>
- <https://arxiv.org/pdf/2106.09685>
- <https://huggingface.co/docs/peft/quicktour>

但这里我们不需要写代码进行LoRA微调，而是在讯飞开放平台上通过上传我们刚刚导出的jsonl文件，通过创建数据集，点击“去训练”，设置学习率和训练轮数进行微调，默认方法就是LoRA
