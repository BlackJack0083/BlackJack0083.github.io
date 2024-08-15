## 前情提要

越训练分越低了。。。
前面尝试了3次不同的调节学习率和学习次数，发现结果都不怎么样。。。
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240815123401.png)
一个是初始的学习率0.0008和10次训练
后面调成了学习率0.0005和0.0001，epochs次数为15次
发现好像学习率调大一点会好？不过没继续尝试了，每周只有4次提交

Task3给出了一些提高效果的方法，让我们尝试一下（感觉task3 写得没有前面那么清楚）

# 数据增强和评分

## 数据增强

这里给出用星火大模型实现数据增强的方案
首先需要去申请：[星火大模型MAX api领取地址](https://console.xfyun.cn/sale/buy?wareId=9108&packageId=9108009&serviceName=Spark3.5%20Max&businessId=bm35)

申请后点【开始调试】，选择【spark Max】模型，然后记录APPID，APISECRET，APIKEY

### 进入代码

前面的部分没有区别，到调用大模型的时候有点区别：

```python
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

#星火认知大模型Spark Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
#星火认知大模型Spark Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'

def call_sparkai(prompt):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].text
```

跟之前的对比下：

```python
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

SPARKAI_URL = 'wss://xingchen-api.cn-huabei-1.xf-yun.com/v1.1/chat'
#星火认知大模型调用秘钥信息，请结合飞书文档，前往讯飞微调控制台（https://training.xfyun.cn/modelService）查看
SPARKAI_APP_ID = '8101c601'
SPARKAI_API_SECRET = 'ZGVhNGUxNTllZWY1ZTc0MTcxOTczYTE5'
SPARKAI_API_KEY = 'd6afdaec1f39f84ad15a242592c87307'
serviceId = 'xspark13b6k'  
resourceId = '4855463986962432'

if __name__ == '__main__':
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=serviceId,
        model_kwargs={"patch_id": resourceId},
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    print(a.generations[0][0].text)
```

发现`spark_llm_domain`函数现在变成了一个domain值，不同版本大模型有各自的domain值，然后没有了`model_kwargs`这个参数(猜测是跟微调有关，我们这里的Spark Max是通用模型，没有微调，所以不用输入这些特定参数)

### 数据增强思路

- 在原本生成结果进行优化
- 生成一些补充数据作为增强

#### 大模型完成答案生成

这里的想法是让大模型自己根据训练集里面的阅读文本、题目选项和答案进行整理，一方面是把简答题去掉只保留选择，另一方面是如果选择不够的话就自己出几道来补充。

首先还是需要先定义prompt
```python
reading = df.loc[2, '阅读文本']
question = df.loc[2, '选项']
answer = df.loc[2, '答案']


cankao_content = '''
1. 以下哪个选项是“具身认知”的定义？
A. 认知在功能上的独立性、离身性构成了两种理论的基础。
B. 认知在很大程度上是依赖于身体的。
C. 认知的本质就是计算。
D. 认知和心智根本就不存在。

答案：B

2. 以下哪个实验支持了“具身认知”的假设？
A. 一个关于耳机舒适度的测试。
B. 一个关于眼睛疲劳程度的测试。
C. 一个关于人类感知能力的实验。
D. 一个关于人类记忆力的实验。

答案：A

3. 以下哪个选项是“离身认知”的教育观的特点？
A. 教育仅仅是心智能力的培养和训练，思维、记忆和学习等心智过程同身体无关。
B. 教育观认为身体仅仅是一个“容器”，是一个把心智带到课堂的“载体”。
C. 教育观认为知识经验的获得在很大程度上依赖于我们身体的体验性。
D. 教育观认为知识经验的获得在很大程度上依赖于我们大脑的记忆能力。

答案：A

4. 以下哪个选项是“具身认知”带来的教育理念和学习理念的变化？
A. 更强调全身心投入的主动体验式学习。
B. 更注重操作性的体验课堂，在教学过程中将学生的身体充分调动起来，这在教授抽象的概念知识时尤为重要。
C. 更强调教师的教学方法和学生的学习方法。
D. 更注重教师的教学技巧和学生的学习技巧。

答案：A'''

def get_adddata_prompt_zero(reading, cankao_content, question, answer):
    prompt = f'''你是一个高考英语阅读题出题专家，请阅读材料，需要参考参考内容 按照要求将题目、选项、答案对其补充完整。

###阅读材料
{reading}


###要求
1.需要将序号对应的题目与答案做匹配。
2.匹配后格式按照问题、ABCD四个选项顺序、答案的结构组合，按照参考内容格式输出。
3.如果选择题目数量不够四个请根据阅读材料及出题思路再生成题目，总题目达到四个。
4.题目中不能出现任何不合理的词汇、语法错误。
5.如果有简答题目与答案请忽略这部分内容，只处理选择题目。

###参考内容
{cankao_content}

###题目
{question}

###答案
{answer}
'''
    return prompt
```

这里的意思应该是让它对输入的题目和答案按照参考内容的格式进行匹配？去掉简答题部分，然后如果没有达到4个题目的话就往后面补充？

我们喂给它一篇文章，看看给出的结果：
![image.png|381](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240815141043.png)
看着还行。。。？
但是不对啊，我不是应该不够才让它补充么，原本的题目怎么不见了，全部都是它自己创造的了？？

再捉个虫，明明出的是语文题，为什么prompt给的是英语阅读题出题专家。。
总体评价为寄，可能是prompt有点问题(可惜怎么改都还是失败)。

#### 大模型完成增强数据

##### 选项与答案补全

由于之前生成的数据中我们处理的数据(指的是baseline1 中的output.jsonl)不一定满足四个题目与答案，这里我们需要将答案补全。
这里的`get_adddata_prompt_rebuild` 函数几乎和上面的一样，只是对prompt进行了修改，要求补充一些题目并按照参考内容格式输出，这里处理的内容是baseline1的`output.jsonl`，不同于上面的操作，这里输出是先输出output里面已有的题目，然后大模型再在后面补充。

```python
import json
from loguru import logger
import time
import re

# 这个函数似乎没有用到
def api_retry(query):
    # 最大尝试次数
    max_retries = 3
    # 再次尝试等待时间
    retry_delay = 120  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return call_sparkai(query)
        except Exception as e:
            attempts += 1   
            if attempts < max_retries:
                logger.warning(f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed for text: {query}. Error: {e}")
                raise

# 抽取一个例子示范
data = list_data[32]
# 处理解析后的数据
input = data.get('input')
output = data.get('output')
text = input

# 使用正则表达式匹配阅读文本后的内容
match = re.search(r'### 阅读文本\n(.*)', text, re.DOTALL)
reading = match.group(1)
prompt = get_adddata_prompt_rebuild(reading,cankao_content,output)
rebuild_output = call_sparkai(prompt)
print(output,'\n###########\n',rebuild_output)
```

这里的结果还行：

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240815152021.png)

##### 对刚刚答案生成的数据进行扩增

保持阅读题目不变，生成新的QA，使得数据集进行扩增

## 评分

还是通过大模型来评分：
会发现我们又对prompt进行了修改

```python
judgement = f'''
你是一个高考阅读题目出题专家，你需要根据下面要求结合阅读文章对题目及答案这样的出题情况进行打分，根据要求一步一步打分，得到有效分数后你将得到100万元的报酬，给出最终得分情况，以“总分:XX分”的形式返回。

### 阅读文章
{reading}

### 题目及答案
{QA}

### 要求

1. 判断给出的题目及答案，题目是否为四道，如果不满足四道，少一道题扣10分，如果每个题目没有答案，少一个答案扣5分。
1. 给出题目选项与答案匹配正确度给分，通过阅读文章每分析道题目正确，则给5分，如果错误给0分。四道题满分20分。
2. 给出题目与选项在阅读文章中的匹配程度给分，每道题目符合阅读文章且选择答案复合题目并可用通过阅读文章分析得到，完全符合给3分，完全不符合给0分。四道题满分12分。
3. 给出题目与选项是否符合高考难度，每道题目与答案是否符合高考的难度，完全符合给3分，完全不符合给0分。四道题满分12分。
4. 给出最终得分情况,对上面三个分数进行求和得到总分，以“总分:XX分”的形式返回，三个问题满分共44分。
'''

score = call_sparkai(judgement)
score

import re

text = score.replace(' ', '')

# 使用正则表达式匹配阅读文本后的内容

match = re.search(r'总分：(\d+)分', text)

if match:
    content = match.group(1)
    print(int(content))
else:
    print("未找到匹配的内容")
```

结果展示：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240815152431.png)


大模型目前可以弥补一些人类评分的痛点，提升评分效率。掌握这个方法对日后完成评价类任务有很大帮助。评分技术不光用在agent设计，还可以优化推荐算法等等，帮你提升算法质量。

1. 人类评分的痛点
	- 主观性和不一致：不同评分者可能因个人标准和偏见导致评分不一致。
	- 时间和资源密集：手动评分耗时且需要大量人力资源，限制了评分任务的可扩展性和效率。
	- 疲劳和认知限制：评分者易受疲劳和认知限制影响，影响评分质量和一致性。
	- 缺乏细致反馈：难以提供针对绩效特定方面的详细反馈。
2. AI在评分方面的优势
	- 一致性和标准化：LLMs通过训练和微调，确保评分的一致性。
	- 效率和可扩展性：AI系统能快速处理大量数据，提高评分效率。
	- 客观性和公正性：减少人类主观性和偏见，促进公平。
	- 细致且可操作的反馈：提供针对绩效各方面的详细反馈。

补充学习资料：
https://dev.to/tarek_eissa/large-language-models-llms-in-scoring-tasks-and-decision-making-3gko
https://huggingface.co/learn/cookbook/en/llm_judge

# 实操

那么现在我们来实操一下数据增强的思路二(思路一感觉实在有点奇怪，，，)

