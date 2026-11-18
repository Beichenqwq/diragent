# 使用type关键字定义类型别名
# 定义一些agent层底层共享的类型别名
# 工具参数和与provider交互的数据都得是json类型的，由于api使用结构化的json来交换数据。

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue] # 递归类型
type JSONObject = dict[str, JSONValue]

