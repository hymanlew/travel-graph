from pydantic import BaseModel, Field, validator

# 所有继承自BaseModel的类都可以作为工具使用。这是因为LangChain会自动将这些模型转换为工具定义（llm.bind_tools）：
# - 类名成为工具名
# - 类的文档字符串成为工具描述
# - 字段定义成为工具参数
# 定义数据模型类，只要继承了BaseModel，就可以被当做工具使用（信号型，通知要调用助手节点），即调用实际上是在 LLM 的响应中返回一个工具调用的指示，而不是直接执行代码。
# LLM会根据对话上下文和字段描述来提取或生成相应的值（类中字段的值），并匹配到对应的类工具。
# Config类中的json_schema_extra提供了一个示例，这有助于LLM理解如何填充这个工具的参数。

# 实际执行时，LangGraph框架会根据这个响应（信号）将控制流转到对应的节点（需要在图中定义好节点与边），在那里才会真正执行相关的操作。
# 因此，这个类工具不仅是一个函数调用，还是一个工作流转交的指令，它触发了图中不同部分（子图）的激活。

# 它与 tool 装饰的工具是两类工具。
# - 后者称为功能型工具，一般是调用某个接口来得到结果，有明确的输入与输出。
# - 而继承自 BaseModel 类的工具称为信号型工具，其没有输入、输出，而是负责存储一些属性。
# 在帮助模型明确了属性的作用后（通过类名、类注释、示例等），模型会在认为需要调用工具或属性需要修改时，自动调用工具进行相应的操作。
class CompleteOrEscalate(BaseModel):
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


class ToFlightBookingAssistant(BaseModel):
    """
    将工作转交给专门处理航班查询，更新和取消的助理。
    """

    request: str = Field(
        description="更新航班，助理在继续之前需要澄清的任何后续问题。"
    )


class ToBookCarRental(BaseModel):
    """
    将工作转交给专门处理租车预订的助理。
    """

    location: str = Field(
        description="用户想要租车的位置。"
    )
    start_date: str = Field(description="租车开始日期。")
    end_date: str = Field(description="租车结束日期。")
    request: str = Field(
        description="用户关于租车的任何额外信息或请求。"
    )

    class Config:
        json_schema_extra = {
            "示例": {
                "location": "巴塞尔",
                "start_date": "2023-07-01",
                "end_date": "2023-07-05",
                "request": "我需要一辆带自动变速器的小型车。",
            }
        }


class ToHotelBookingAssistant(BaseModel):
    """
    将工作转交给专门处理酒店预订的助理。
    """

    location: str = Field(
        description="用户想要预订酒店的位置。"
    )
    checkin_date: str = Field(description="酒店入住日期。")
    checkout_date: str = Field(description="酒店退房日期。")
    request: str = Field(
        description="用户关于酒店预订的任何额外信息或请求。"
    )

    # 验证器确保结束日期不早于开始日期
    @validator('checkout_date')
    def end_date_after_start_date(cls, v, values):
        if 'checkin_date' in values and v < values['checkin_date']:
            raise ValueError('结束日期必须晚于开始日期')
        return v

    class Config:
        json_schema_extra = {
            "示例": {
                "location": "苏黎世",
                "checkin_date": "2023-08-15",
                "checkout_date": "2023-08-20",
                "request": "我偏好靠近市中心且房间有景观的酒店。",
            }
        }


class ToBookExcursion(BaseModel):
    """
    将工作转交给专门处理旅行推荐及其他游览预订的助理。
    """

    location: str = Field(
        description="用户想要预订推荐旅行的位置。"
    )
    request: str = Field(
        description="用户关于旅行推荐的任何额外信息或请求。"
    )

    class Config:
        json_schema_extra = {
            "示例": {
                "location": "卢塞恩",
                "request": "用户对户外活动和风景名胜感兴趣。",
            }
        }
