import os
from typing import Annotated, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver  # 更企业级的检查点存储

# --- 状态定义 ---
class ExpenseState(TypedDict):
    """定义费用审批流程的状态。"""
    messages: Annotated[list, add_messages]  # 消息历史
    expense_report: Optional[dict]  # 存储费用报告数据
    status: Optional[Literal["pending", "approved", "rejected", "requires_revision"]]  # 审批状态
    applicant_email: Optional[str]  # 申请人邮箱
    amount: Optional[float]  # 费用金额
    currency: Optional[str]  # 货币类型


# --- 工具函数 ---
# 模拟一个内部策略检查工具
def check_against_policy(amount: float, category: str) -> dict:
    """根据公司政策检查费用。返回是否需要人工审批。"""
    POLICY_LIMITS = {"travel": 1000.0, "entertainment": 500.0, "equipment": 2000.0}
    requires_approval = amount > POLICY_LIMITS.get(category, float('inf'))
    return {"requires_approval": requires_approval, "checked": True}


# 模拟发送邮件的工具
def send_email(to: str, subject: str, body: str) -> str:
    """模拟发送邮件。企业中可集成SendGrid、Mailgun等API。"""
    print(f"\n--- 邮件开始 ---\n发送至: {to}\n主题: {subject}\n内容: {body}\n--- 邮件结束 ---")
    return f"邮件已发送至 {to}"


# --- 节点函数 ---
def ai_initial_review(state: ExpenseState):
    """AI节点：初步审核费用报告。"""
    # 这里简化处理。实际企业中，可能从state中解析用户消息来获取费用数据。
    expense_data = state.get("expense_report", {})
    amount = expense_data.get("amount", 0)
    category = expense_data.get("category", "other")

    # 调用策略检查工具
    policy_check = check_against_policy(amount, category)

    if policy_check["requires_approval"]:
        # 需要人工审批，准备详细信息并中断
        approval_query = {
            "expense_id": expense_data.get("id", "N/A"),
            "employee": expense_data.get("employee_name", "Unknown"),
            "amount": f"{amount} {expense_data.get('currency', 'USD')}",
            "category": category,
            "description": expense_data.get("description", ""),
            "message": f"费用金额 {amount} {expense_data.get('currency', 'USD')} 超过 {category} 类别的自动审批限额，请经理审批。"
        }
        # 中断执行，等待人工处理。此信息会被传递到外部系统。
        interrupt(approval_query)  # 传递一个复杂的字典对象
        # 中断后，代码会在此暂停，直到收到恢复的Command。

    # 如果不需要人工审批，或人工审批后恢复执行，继续更新状态
    # 注意：人工审批的结果（批准/拒绝）是通过Command在恢复时传入的，会体现在新的状态中。
    return {"status": "approved", "messages": [{"role": "assistant", "content": "AI初审完成/人工审批已完成。"}]}


def human_approval_node(state: ExpenseState):
    """
    人工审批节点（模拟）。
    注意：在实际LangGraph中断流程中，此节点可能不会以标准节点形式存在。
    因为中断发生在`ai_initial_review`节点内部，恢复后的逻辑通常也在该节点或通过条件边处理。
    这里仅为演示概念。更常见的做法是在中断恢复后，根据Command携带的数据直接更新状态。
    """
    # 在实际应用中，人工审批的“动作”是在图外通过另一个系统/界面完成的。
    # 恢复图时带来的Command已经包含了审批结果。
    # 这个节点可能用于记录日志或执行后续操作，但审批决策本身来自interrupt之后的Command。
    print("人工审批意见已接收并处理。")
    return state


def notify_applicant(state: ExpenseState):
    """通知申请人最终结果。"""
    status = state.get("status")
    email = state.get("applicant_email")
    expense_id = state.get("expense_report", {}).get("id", "N/A")

    if not email:
        return {"messages": [{"role": "assistant", "content": "无申请人邮箱，无法通知。"}]}

    subject_map = {
        "approved": f"您的费用报销单 #{expense_id} 已批准",
        "rejected": f"您的费用报销单 #{expense_id} 未被批准",
        "requires_revision": f"您的费用报销单 #{expense_id} 需要修改"
    }
    subject = subject_map.get(status, "关于您的费用报销单更新")

    body_map = {
        "approved": f"您好！很高兴通知您，费用报销单 #{expense_id} 已获批准。款项将按公司流程支付。",
        "rejected": f"您好！很遗憾，您的费用报销单 #{expense_id} 未被批准。原因：不符合公司相关政策。请联系您的经理了解更多细节。",
        "requires_revision": f"您好！您的费用报销单 #{expense_id} 需要一些修改才能继续处理。请登录系统查看具体反馈意见。"
    }
    body = body_map.get(status, f"您的费用报销单 #{expense_id} 状态已更新为：{status}。")

    # 调用发送邮件工具
    send_email_result = send_email(to=email, subject=subject, body=body)
    return {"messages": [{"role": "assistant", "content": f已尝试通知申请人。{send_email_result}"}]}


# --- 构建图 ---
builder = StateGraph(State=ExpenseState)
builder.add_node("ai_review", ai_initial_review)
builder.add_node("human_approval", human_approval_node)  # 概念性节点
builder.add_node("notify", notify_applicant)

builder.add_edge(START, "ai_review")

# 条件路由：根据状态决定下一步
def route_after_ai_review(state: ExpenseState):
    """AI审核后，根据状态决定下一步"""
    status = state.get("status")
    if status == "approved":
        return "notify"  # 直接去通知
    elif status in ["rejected", "requires_revision"]:
        return "notify"  # 也去通知（拒绝或需修改）
    else:
        # 其他情况，或者需要人工干预（interrupt）的情况，可能会在恢复后继续从此处判断
        return "notify"  # 简化处理，实际逻辑可能更复杂


builder.add_conditional_edges(
    "ai_review",
    route_after_ai_review,
    {
        "notify": "notify",
        # ... 其他可能的目标节点
    }
)
builder.add_edge("notify", END)

# 使用SQLite存储检查点，更符合企业持久化需求
memory = SqliteSaver.from_conn_string(":memory:")  # 示例用内存数据库，企业应用请换为实际DB路径

# 编译图
# 在需要人工审批的节点【之后】或工具调用【之前】预设中断点，可以使用`interrupt_before`。
# 本例是在ai_review节点内部主动调用interrupt()。
graph = builder.compile(checkpointer=memory)

# --- 模拟执行流程 ---

if __name__ == "__main__":
    # 模拟一份高额费用报告，会触发人工审批
    sample_expense = {
        "id": "EXP-2025-09-15-001",
        "employee_name": "张三",
        "amount": 1500.0,
        "currency": "CNY",
        "category": "travel",
        "description": "客户洽谈差旅费"
    }

    initial_state = {
        "messages": [{"role": "user", "content": "提交费用报告"}],
        "expense_report": sample_expense,
        "applicant_email": "zhangsan@example.com",
        "status": "pending",
        "amount": sample_expense["amount"],
        "currency": sample_expense["currency"]
    }

    config = {"configurable": {"thread_id": "expense_thread_zhangsan_001"}}  # 线程ID，唯一标识此次审批流程

    print("=== 第1步：员工提交费用报告，AI初步审核 ===")
    # 首次流式执行
    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            print(f"当前状态: {event.get('status')}")
    except Exception as e:
        # 预期中，AI审核节点会触发中断
        print(f"\n流程已中断，等待人工审批。中断信息: {e}")
        # 此处模拟一个外部审批系统捕获到了中断及其传递的approval_query信息

    print("\n=== 第2步：模拟经理在外部系统进行审批操作 ===")
    # 模拟经理批准了费用
    # 构建恢复执行的Command，携带审批结果
    manager_decision = "approved"  # 或 "rejected", "requires_revision"
    resume_data = {
        "status": manager_decision,
        "manager_comment": "情况特殊，批准执行。",
        "approved_by": "李经理(2025-09-15 10:30)"
    }

    # 关键：使用Command恢复执行，并将人工审批结果带入图中
    resume_command = Command(resume=resume_data)

    print("=== 第3步：携带经理审批结果恢复图执行 ===")
    # 恢复执行
    for event in graph.stream(resume_command, config, stream_mode="values"):
        print(f"处理事件: {event}")
        if "messages" in event:
            last_message = event["messages"][-1]
            print(f"最新消息: {last_message['content']}")

    print("\n流程执行完毕。")
    # 可以查询最终状态
    final_state = graph.get_state(config)
    print(f"最终审批状态: {final_state.values.get('status')}")