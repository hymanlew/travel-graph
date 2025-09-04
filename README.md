
## 项目介绍

AI旅游助手， LangGraph 实现的智能多代理旅行助手系统，通过主助理与多个专业子助理（航班、酒店、租车、游览）的协作，自动化处理复杂的旅行预订、查询与管理任务。提供个性化的旅行规划、陪伴和分享服务，让您的旅程充满乐趣并留下难忘回忆。

基于星火**大模型的文生文、图生文以及文生语音**等技术，量身定制一份满意的旅行计划。无论您期望体验何种旅行目的地、天数、行程风格（如紧凑、适中或休闲）、预算以及随行人数，我们的助手都能为您精心规划行程并生成详尽的旅行计划表，包括每天的行程安排、交通方式以及需要注意的事项等。

此外，我们还采用RAG技术，专为提供实用全方位信息而设计，包括景点推荐、活动安排、餐饮、住宿、购物、行程推荐以及实用小贴士等。目前，我们的知识库已涵盖全国各地区、城市的旅游攻略信息，为您提供丰富多样的旅行建议。

您还可以随时拍摄旅途中的照片，并通过我们的应用上传。应用将自动为您生成适应不同社交媒体平台（如朋友圈、小红书、抖音、微博）的文案风格，让您轻松分享旅途中的点滴，与朋友们共同感受旅游的乐趣。

 **功能模块**

- 旅游规划助手
- 旅游问答助手
- 旅行文案助手
- 各助理的介绍:
  
  1. 主助理, 负责将任务分配给不同的子助理
  2. 航班子助理
  3. 酒店子助理
  4. 租车子助理
  5. 游览子助理

**技术亮点**

- 充分**使用星火大模型API矩阵能力**，包含星火大模型、图片理解、语音合成、语音识别、文生图、embedding等
- 旅游规划、文案生成 Prompt 高效设计，**ReAct 提示框架**设计
- RAG 创新：根据用户 query 动态加载，读取文本；BM25检索、向量检索的**混合检索；重排模型**高效使用
- 多模态生成：**图生文**，**文生图**，**TTS**，**ASR** 和**数字人**视频合成
- 旅游问答 Agent 实现：**查询天气**、**附近搜索**、**联网搜索**
- 生成语音，生成图片和数字人视频全部可预览查看、下载，提高体验

## 项目流程图

![](img\LvBan流程图.png)

![](img\旅行助手流程图-main.png)

## 项目演示

- 旅游规划助手
  

![](img\旅游规划助手v2.0.gif)


- 旅游问答助手
  

![](img\旅游问答助手v2.0.gif)


- 旅游文案助手
  
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E6%96%87%E6%A1%88%E5%8A%A9%E6%89%8Bv2.0.gif" alt="Demo gif" >
</p>

<details>
<summary>LvBan_v1.5项目展示</summary>
<br>

- 旅游规划助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%A7%84%E5%88%92%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>


- 旅游问答助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%97%AE%E7%AD%94%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>

- 旅游文案助手
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%96%87%E6%A1%88%E5%8A%A9%E6%89%8B.gif" alt="Demo gif" >
</p>

</details>



## 🎉 NEWS
- [2024.09.26] **发布LvBan v2.0**演示视频：[https://www.bilibili.com/video/BV1wPxuepEBG](https://www.bilibili.com/video/BV1wPxuepEBG)
- [2024.08.13] 项目介绍视频发布：[B站](https://www.bilibili.com/video/BV1pxYye6ECE)

## 🗂️ 部署步骤

- [详细指南](#2-详细指南)
  - [数据、模型及工具选型](#21-数据、模型及工具选型)
  - [基于本地旅游攻略pdf文本文件的RAG系统](#22-基于本地旅游攻略pdf文本文件的RAG系统)
  - [多模态生成：图生文，TTS和数字人视频合成](#23-多模态生成：图生文，TTS和数字人视频合成)
  - [旅游问答智能体(Agent)实现](#24-旅游问答智能体(Agent)实现)
- [案例展示](#3-案例展示)
- [人员贡献](#4-人员贡献)
- [ 致谢](#5-致谢)

### 在线体验

- 目前已将 `LvBan v2.0` 版本部署到modelscope平台，地址: [https://www.modelscope.cn/studios/NumberJys/LvBan](https://www.modelscope.cn/studios/NumberJys/LvBan)

### 本地部署

```bash
git clone https://github.com/yaosenJ/LvBanGPT.git
cd LvBanGPT
conda create -n LvBanGPT python=3.10.0 -y
conda activate  LvBanGPT
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# 下载重排模型
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
base_path = './model/rerank_model/'
# download repo to the base_path directory using git
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_rerank.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

# 修改.env文件，添加自己相关的key

gradio app.py
python3 app.py
```
![](img\成功页面.png)

### PAI-DSW 部署

选择`ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.14.0`魔搭GPU镜像，启动环境。地址: https://www.modelscope.cn/my/mynotebook/preset

## 线上技术实现

### 模型工具选型

- 数据集：全国各地区及景点旅游攻略pdf文本文件
  
- 大语言模型：星火大模型(Spark3.5 Max)
- 图片理解模型：星火图片理解模型
- 图片生成模型，星火文生图模型
- 语音合成模型：星火语音合成模型
- 语音识别模型：星火语音识别模型
- 向量模型：星火文本向量模型

![](img\讯飞开放平台1.png)

### 旅游 RAG 系统

该项目的RAG系统，首先从用户的查询中提取关键信息（如城市名称、地点名称等），并通过这些信息检索匹配的pdf文件，提取相关内容并计算其嵌入向量。然后利用BM25检索和向量检索技术，筛选出与用户查询相似度较高的文本块。在此基础上，利用重排序模型对这些文本块进行进一步排序，最终选择最相关的内容提供给星火大模型。星火大模型根据这些上下文信息，生成对用户问题的准确回答。其详细技术实现流程：

**文本处理与文档匹配**

- 城市提取（extract_cities_from_text）：使用jieba进行中文分词，提取query文本中提及的地名（城市名称）。
- PDF文件匹配（find_pdfs_with_city）：根据提取的城市名称，在指定目录下寻找包含这些城市名称的pdf文件。
  

**嵌入生成与文档处理**

- PDF内容提取与分割（embedding_make）：
  - 1. 根据用户输入的文本，调用get_embedding_pdf函数提取相关的PDF文件。
  - 2. 从提取的pdf文件中读取文本内容，并对内容进行清理和分割，使用RecursiveCharacterTextSplitter将文本按(chunk_size=1000, chunk_overlap=300)进行切分，以便后续处理。
  - 3. 使用BM25Retriever对切分后的文本块进行初步检索，获得与用户问题最相关前20个文档片段。
    

**嵌入计算与相似度匹配**

- 嵌入计算：
  - 1. 通过加载的EmbeddingModel(星火文本向量模型)，为用户的查询问题和检索到的文档片段生成嵌入向量。
  - 2. 使用余弦相似度（cosine_similarity）计算查询问题与文档片段之间的相似度。
  - 3. 根据相似度选择最相关的前10个文档片段。
    

**文档重排序与生成回答**

- 重排序（rerank）：加载预训练的重排序模型(BAAI/bge-reranker-large)，对初步选出的文档片段进行进一步排序，选择出最相关的3个片段。
- 生成回答：
  - 1. 将重排序后的文档片段整合，并形成模型输入（通过指定的格式，将上下文和问题整合）。
  - 2. 调用ChatModel(星火大语言模型)生成最终回答，并返回给用户。

### 多模态及数字人合成

![](img\多模态生成v2.0.png)


通过将文本数据处理成音频数据后同视频一起输入，先使用脚本处理视频，该脚本首先会预先进行必要的预处理，例如人脸检测、人脸解析和 VAE 编码等。对音频和图片通过唇同步模型处理，生成对应唇形的照片，匹配所有的音频，最终将音频与生成的图片合成为视频输出。

<h3 id="2-4"> 旅游问答智能体(Agent)实现</h3>

- 查询天气Agent: 利用星火大模型（Spark3.5 Max）和 和风天气API实现联网搜索Agent。
- 附近搜索Agent: 利用星火大模型（Spark3.5 Max）和高德地图API实现附近搜索Agent。该Agent系统可以根据用户输入的文本请求，星火大模型自动判断是否需要调用高德地图API。若提问关于附近地址查询问题，则调用地图服务来获取地点信息和附近POI，目的帮助用户查询特定地点的周边设施、提供地址信息等，反之，其他问题，不调用高德地图API。
- 联网搜索Agent：利用星火大模型（Spark3.5 Max）和 Travily 搜索引擎API实现联网搜索Agent。

<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/Agent.png" style="zoom:40%;" />

### 语音识别对话

运行asr.py即可
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/asr.png" style="zoom:40%;" />
<center><img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/asr_record.png" style="zoom:40%;" />
### 案例展示

- 旅游规划助手
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E6%97%85%E6%B8%B8%E8%A7%84%E5%88%92v2.0.png" alt="Demo" >
- 知识库问答(RAG:true)
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E7%9F%A5%E8%AF%86%E5%BA%93%E9%97%AE%E7%AD%94v2.0_1.png" alt="Demo" >
- 知识库问答(RAG:false)
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E7%9F%A5%E8%AF%86%E5%BA%93%E9%97%AE%E7%AD%94v2.0_2.png" alt="Demo" >
- 附近查询&联网搜索&天气查询
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E5%AE%9E%E5%86%B5%E6%9F%A5%E8%AF%A2v2.0.png" alt="Demo" >
- 旅游文案助手
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/LvBan_v2.0/img/%E6%97%85%E6%B8%B8%E6%96%87%E6%A1%88v2.0.png" alt="Demo" >

```htaccess
<details>
<summary>LvBan_v1.5案例展示</summary>
<br>
<h2 id="3"> 案例展示 </h2>
<p align="center">
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%97%85%E6%B8%B8%E6%94%BB%E7%95%A5.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/RAG.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E5%A4%A9%E6%B0%94%E6%9F%A5%E8%AF%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E9%99%84%E8%BF%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E8%81%94%E7%BD%91%E6%90%9C%E7%B4%A2.png" alt="Demo" >
  <img src="https://github.com/yaosenJ/LvBanGPT/blob/main/img/%E6%96%87%E6%A1%88%E7%94%9F%E6%88%90.png" alt="Demo" >
</p>
</details>
```

## 本地图项目流程

### 定义部分

#### 主入口节点(fetch_user_info)

首先由Start节点进入主入口(fetch_user_info), 对应执行的函数为查询数据库中, 与用户航班相关的一切信息, 并将结果存入state的user_info中, 结果格式如下:

```python
{
    "ticket_no": "ABC1234567",
    "book_ref": "BR0001",
    "flight_id": "987",
    "flight_no": "UA101",
    "departure_airport": "JFK",
    "arrival_airport": "LAX",
    "scheduled_departure": "2025-06-01 08:00:00",
    "scheduled_arrival": "2025-06-01 11:30:00",
    "seat_no": "12A",
    "fare_conditions": "Economy"
}
```

#### 子助理入口节点(enter_update_flight)

- 接下来, 对每个子助理, 构建入口节点 enter_up_flight(以机票子助理为例)
- 入口节点负责提示当前的助理自己的身份, 以及自己的任务, 注意==提示是放在tool_message中的, 以防止对系统与用户消息的污染==
- 四个子助理都要创建入口节点, 因此存在大量可复用的部分; 然而LangChain的绑定在node中的函数要求是必须仅以state作为输入, 因此==使用闭包, 达到"实际上传了多个参数"的效果==

#### 子助理更新节点(update_flight): 仍以机票助手为例==(核心)==

- 这个node负责更新航班相关信息, ==准确地说, 是"判断要调用哪些工具, 进行哪些更新", 但并不能实际进行这些更新!==
- node绑定的不再是一个函数, 而是一个具有call方法的类(允许将其当作函数使用, 适用于更复杂的，需要进行更多定制的场景
- 这里的函数以一个runnable对象作为init的输入, 并将其保存在self的参数中, 再在call方法中使用(call方法的传参仍然必须是规定格式, 即只能传入state和config)
- 这里的runnable对象实际上就是一个负责更新节点的LLM, 通过LangChain绑定了对应的提示词和工具 
  - 其提示词的功能为: 告知模型自己负责更新机票相关的操作, 并且把之前存在user_info中的机票相关信息传给模型
  - 绑定的工具包括:
    - ==安全工具(无需用户确认: 搜索航班)==
    - ==敏感工具(需要用户确认: 改签航班 & 取消航班)==
    - ==CompleteOrEscalate工具: 提示子助理什么时候应该返回主助理==
      - 属性
        - cancel: 是否离开当前身份, 返回主助理
        - reason: 这样做的理由
      - 示例(通过Config类)

```python
class CompleteOrEscalate(BaseModel):  # 定义数据模型类 —— 只要继承了BaseModel，就可以被当做工具使用（信号型）
    """
    一个工具，用于标记当前任务为已完成和/或将对话的控制权升级到主助理，
    主助理可以根据用户的需求重新路由对话。
    """

    cancel: bool = True  # 默认取消任务
    reason: str  # 取消或升级的原因说明

    # 示例：分别展示了可能会调用类的三种情况 —— 任务完成，用户意图变更与权限不足、需要调用其它工具
    class Config:  # 内部类 Config: json_schema_extra: 这个字段包含了一些示例数据
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "用户改变了对当前任务的想法。",
            },
            "example2": {
                "cancel": True,
                "reason": "我已经完成了任务。",
            },
            "example3": {
                "cancel": False,
                "reason": "我需要搜索用户的电子邮件或日历以获取更多信息。",
            },
        }
```

- 函数的操作是对runnable对象生成的结果进行进一步控制, 写了一个循环, 当其没有生成有效结果时, 会提示模型重新生成, 直到生成符合要求和规范的结果, 才终止循环, 返回结果, 这样的好处是保证了结果的健壮性, 降低因为偶发因素导致结果出错的概率

==与此前的确定性连边不同, 在更新节点之后, 需要进行条件路由(因为之后跳转到什么节点是不确定的, 可能是工具节点也可能是离开节点)==

#### 实际的工具调用节点(update_flight_safe_tools, update_flight_sensitive_tools): 对数据库增删改查

前面的更新节点已经得到了要调用什么工具, 则只需要经过路由判断, 即可来到对应的实际工具调用节点

- 实际调用工具函数时, 用create_tool_node_with_fallback(工具) 的方式, 目的是增加错误处理的相关操作

#### 离开节点(leave_skill)

同样是根据路由判断, 当CompleteOrEscalate工具中cancel的值为true, 就来到这个节点

- 其函数要实现的功能包括:
  - 提示模型正在返回主助理, 明确自己的身份并回顾之前的历史记录, 以更好地执行自己的任务
  - 对state的字段进行更新: state中的dialog_state指示了当前模型的身份, 此时由于是退出当前节点, 因此对其字段命名为"pop", 系统会自动识别出这是一个出栈命令而不是一个对字段的赋值

```python
return {
            # 更新对话状态为弹出 —— 这里的 pop不是具体值，而是表示要弹出（被识别为一个“命令”）,因此不冲突
            # 具体的值在state中定义，只能是那五个
            "dialog_state": "pop",
            "messages": messages,  # 返回消息列表
        }
```

#### 主助理节点(primary_assistant)

==**为什么要设置主助理:** 因为任务是多步骤的, 尽管一开始的fetch_user_info可以连接到各个节点, 但需要在各个助手间反复切换时, 就需要一个主助理协助调度==

其对应的函数中操作为: 设置提示词, 让模型判断应该将任务交给哪个子助手来完成

其可以使用的工具包括:

- 各个用于跳转到子助手的工具节点(信号型工具)
- 查询公司政策的工具
- 外部工具如tavily_tool
- 查询航班的工具(因为所有的信息都根据航班来进行, 所以主助理应该可以调用这个工具来查询需要的信息)

```
primary_assistant_tools = [
    tavily_tool,  # 假设TavilySearchResults是一个有效的搜索工具
    search_flights,  # 搜索航班的工具
    lookup_policy,  # 查找公司政策的工具
]
```

#### 各个节点之间的跳转关系

- 一开始的Start节点必然跳转到fetch_user_info
- fetch_user_info可能跳转到主助理与各个子助理, (如果state栈空则主助理)
- 主助理可能跳转到各个子助理的入口节点, 也可能跳转到主助理对应的工具节点
- 各个子助理的入口节点必然跳转到各个子助理节点
- 各个子助理节点可能跳转到工具节点或者leave节点
- leave节点必然跳转到主助理节点

### 执行部分

设置一个无限循环: 每次都由用户先输入问题, 再进入流程让模型进行处理

循环开始时, 判断用户的消息是不是表达"退出", 如果是, 直接结束流程

否则正式执行流程: 

```python
events = graph.stream({'messages': ('user', question)}, config, stream_mode='values')
```

使用stream方式, 这会返回发生中断之前的所有会话, 打印之

当中断发生后, 询问用户是否批准当前操作

- 如果批准, 直接继续执行(由于config中保存了会话id和用户id, 因此在中断后取得之前的config, 就可以回到原来的处理流程, 无需手动管理state, 非常方便)

```python
if user_input.strip().lower() == "y":
    # 之前的流中断了，因此要继续执行 —— 由于之前只是中断，因此会自动继续利用之前的state，继续流程
    events = graph.stream(None, config, stream_mode='values') // state的位置传None即可
```

- 如果不批准, 也要继续执行, 但多加一个tool消息(也是将局部作用域的提示伪装成工具调用返回给模型, 其中不仅提示模型请求被用户拒绝, 还将用户的输入作为拒绝的原因返回给模型)

## 项目亮点

### 将对子助理的身份提示放在tool_message中

实际上就是把人为对模型的提示"伪装成一条工具调用的返回结果", 从而给模型一个局部作用域的身份提示, 既避免了使用系统提示造成的污染(多个子助理各自修改系统提示, 由于其是全局的, 造成身份信息的混乱), 也避免了伪装成用户消息对用户消息造成的污染

### 使用闭包创建各个子助理的入口节点

四个子助理都要创建入口节点, 因此存在大量可复用的部分; 然而LangChain的绑定在node中的函数要求是必须仅以state作为输入, 因此==使用闭包, 达到"实际上传了多个参数"的效果==

### 将工具划分为安全工具与敏感工具

对于需要增删改查的操作, 认为是敏感操作, 流程图在执行到对应位置时会被人为中断, 向用户确认之后才继续执行

```python
graph = builder.compile(
    # 检查点：如果工作流中发生中断或失败，memory 将用于恢复工作流的状态。
    checkpointer=memory,
    # 工作流执行到这些节点时会中断，并向用户确认
    # 中断意味着流的停止，且由于这是一个人为造成的中断，模型仍然可以基于原本的定义得知其“如果不中断的话，下一个节点是什么”
    # 此时current_state.next就会为true
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ]
)
```

### 使用循环 + 格式控制来确保得到符合规范的输出结果

模型是具有不稳定性的, 如果直接将单次LLM生成的结果返回, 其可能会由于各种意外因素导致不正确, 因此采取无限循环 + 格式判断的方式, 当且仅当模型返回了正确的格式, 才将结果返回. 

### 自定义信号型工具来提示子助理是否返回主助理

使用继承自BaseModel的类作为工具

### 用MemorySaver保存历史记录

每轮对话的记录会被自动添加到state的message字段中, 无需手动维护

此外, 在主程序中使用了 LangGraph 提供的 **持久化内存机制**: 

```python
memory = MemorySaver()
```

每一次状态更新后，它会把状态（如 `messages`）**自动保存到内存或磁盘**，支持==中断恢复、跨轮调用==.




<h2 id="5"> 致谢</h2>

感谢科大讯飞股份有限公司、共青团安徽省委员会、安徽省教育厅、安徽省科学技术厅和安徽省学生联合会联合举办的“**2024「星火杯」大模型应用创新赛**”！

感谢科大讯飞提供星火大模型API矩阵能力，包含星火大模型、星火语音识别大模型、图片理解、图片生成、超拟人语音合成、embedding等

感谢Datawhale及其教研团队在项目早期提供的基础教程

感谢Datawhale Amy大佬及其学习交流群的小伙伴们的支持和意见反馈！

感谢A100换你AD钙奶成员们的技术支持和反馈帮助！

感谢上海人工智能实验室，书生浦语大模型实战营的算力和计算平台支持！



