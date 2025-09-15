import uuid
from typing import Annotated
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from tools.flights_tools import fetch_user_flight_information
from graph_chat.assistant import primary_assistant_tools, sensitive_tool_names, sensitive_tools, create_assistant_node
from graph_chat.draw_png import draw_graph
from graph_chat.state import State
from utils.init_db import update_dates
from tools.tools_handler import create_tool_node_with_fallback, _print_event

# 定义了一个流程图的构建对象
builder = StateGraph(State)

def get_user_info(state: State):
    """
    查数据库，获取用户的航班信息并更新状态字典。
    参数:
        state (State): 当前状态字典。
    返回:
        dict: 包含用户信息的新状态字典。
    """
    return {"user_info": fetch_user_flight_information.invoke({})}

def human_approval_node(state: State):
    """
    处理用户批准或拒绝的节点
    """
    # 获取当前状态
    current_state = graph.get_state(config)
    # resume_value 就是 Command(resume=...) 中传递的值
    resume_value = current_state.values.get("__resume__", "")

    if resume_value == "approved":
        # 处理批准的情况
        print("工具调用已获批准，将继续执行...")
        # 这里可以添加任何批准后的特殊处理逻辑
        return {"messages": state["messages"]}
    else:
        # 处理拒绝的情况
        print(f"工具调用被拒绝，原因: {resume_value}")
        # 这里可以添加任何拒绝后的特殊处理逻辑
        # 例如，添加一个工具消息来表示拒绝
        rejection_message = ToolMessage(
            content=resume_value,
            tool_call_id=state["messages"][-1].tool_calls[0]["id"] if state["messages"][-1].tool_calls else ""
        )
        return {"messages": state["messages"] + [rejection_message]}


# 新增：fetch_user_info节点首先运行，这意味着我们的助手可以在不采取任何行动的情况下看到用户的航班信息
builder.add_node('fetch_user_info', get_user_info)
# 自定义函数代表助手节点，Runnable，或者一个自定义的类都可以是节点
builder.add_node('assistant', create_assistant_node())

# 添加一个名为"tools"的节点，该节点创建了一个带有回退机制的工具节点
# builder.add_node('tools', create_tool_node_with_fallback(primary_assistant_tools))
# 如果是不需要异常额外处理的场景，推荐使用 ToolNode：使用 ToolNode 替代自定义的工具节点实现
builder.add_node('safe_tools', ToolNode(primary_assistant_tools))
# 添加敏感工具节点
builder.add_node("sensitive_tools", create_tool_node_with_fallback(sensitive_tools))
# 添加处理批准的节点
builder.add_node('approval_handler', human_approval_node)

def route_conditional_tools(state: State) -> str:
    """
    根据当前状态，来决定下一个要执行的节点
    :param state: 当前的状态
    :return: str 下一个要执行节点的名字
    """
    # tools_condition 会根据助手节点的输出决定是否调用工具。如果需要调用工具，结果会自动传递到 tools 节点执行，然后继续流程
    next_node = tools_condition(state)
    if next_node == END:
        return END

    ai_message = state['messages'][-1]
    if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
        tool_call = ai_message.tool_calls[0]
        if tool_call['name'] in sensitive_tool_names:  # 敏感的工具调用
            return 'sensitive_tools'
        return 'safe_tools'
    return END


# 定义边：这些边决定了控制流如何移动
builder.add_edge(START, 'fetch_user_info')
builder.add_edge('fetch_user_info', "assistant")

# 从"assistant"节点根据条件判断添加到其他节点的边
builder.add_conditional_edges(
    "assistant",
    #  # 使用 lambda 函数判断是否需要工具调用，效果与 tools_condition 相同
    #     lambda state: "tools" if state["messages"][-1].tool_calls else "__end__",
    #     {"tools": "tools", "__end__": "__end__"}
    route_conditional_tools,
    ['safe_tools', 'sensitive_tools', END]
)

# 从"tools"节点回到"assistant"节点添加一条边
builder.add_edge("safe_tools", "approval_handler")
builder.add_edge("sensitive_tools", "approval_handler")
builder.add_edge("approval_handler", "assistant")

# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
memory = MemorySaver()

# 编译状态图，配置检查点为memory, 旧的实现方式是这里配置中断点（但仍然可用的），新方式是直接在节点中配置中断点即可。
# graph = builder.compile(checkpointer=memory, interrupt_before=['tools'],)
graph = builder.compile(checkpointer=memory)
draw_graph(graph, '../graph_chat/graph2.png')

session_id = str(uuid.uuid4())
update_dates()  # 每次测试的时候：保证数据库是全新的，保证，时间也是最近的时间
config = {
    "configurable": {
        # passenger_id 乘客ID，用于我们的航班工具，以获取用户的航班信息
        "passenger_id": "3442 587242",
        # 检查点由session_id访问
        "thread_id": session_id,
    }
}

# 执行工作流
_printed = set()  # set集合，避免重复打印
while True:
    question = input('用户：')
    if question.lower() in ['q', 'exit', 'quit']:
        print('对话结束，拜拜！')
        break
    else:
        # 使用新的流式处理方式
        for event in graph.stream(
            {'messages': [HumanMessage(content=question)]}, 
            config, 
            stream_mode="values"
        ):
            _print_event(event, _printed)

        # 用于获取当前图（工作流）的最新状态。并判断是否需要中断处理
        current_state = graph.get_state(config)

        # current_state.next 是状态对象的一个属性，它包含了图中下一个应该执行的节点列表。包含以下关键属性：
        # - values：当前状态通道的值
        # - next：图中接下来要执行的节点名称的元组
        # - config：与此检查点关联的配置
        # - metadata：与此检查点关联的元数据
        # - tasks：包含有关要执行的下一个任务信息的 PregelTask 对象的元组
        if hasattr(current_state.values.get("messages", []), '__iter__') and current_state.values.get("messages"):
            last_message = current_state.values["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print("检测到需要工具调用，正在等待用户确认...")
                user_input = input("您是否批准上述操作？输入'y'继续；否则，请说明您请求的更改。\n")
                
                if user_input.strip().lower() == "y":
                    # 用户批准，继续执行
                    for event in graph.stream(
                        Command(resume="approved"),
                        config,
                        stream_mode="values"
                    ):
                        _print_event(event, _printed)
                else:
                    # 用户拒绝，传递拒绝原因
                    for event in graph.stream(
                        Command(resume=f"Tool的调用被用户拒绝。原因：'{user_input}'。"),
                        config,
                        stream_mode="values"
                    ):
                        _print_event(event, _printed)

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing import Literal, TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
import operator




# 修复路由函数
def route_primary_assistant(state: State):
    """
    根据当前状态判断路由到哪个节点
    """
    # 检查是否有工具调用
    if state["messages"] and state["messages"][-1].tool_calls:
        return "approval_handler"

    # 检查是否被拒绝
    if state.get("status") == "rejected":
        return "primary_assistant"

    # 默认结束
    return END




# 条件边：从主助理路由
builder.add_conditional_edges(
    'primary_assistant',
    route_primary_assistant,
    ["approval_handler", END]
)

# 条件边：从批准处理器路由
builder.add_conditional_edges(
    'approval_handler',
    after_approval,
    ["primary_tools", "primary_assistant"]
)

# 从工具节点回到主助理
builder.add_edge("primary_tools", "primary_assistant")

# 检查中断
current_state = graph.get_state(config)
if current_state.next:
    next_node = current_state.next[0]
    if next_node in interrupt_before:
        print("检测到需要工具调用，正在等待用户确认...")
        user_input = input(
            "您是否批准上述操作？输入'y'继续；否则，请说明您请求的更改。\n"
        )
        if user_input.strip().lower() == "y":
            resume_value = {"status": "approved"}
        else:
            resume_value = {"status": user_input}

        # 继续执行
        events = graph.stream(
            Command(resume=resume_value),
            config,
            stream_mode="values"
        )