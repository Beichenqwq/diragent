"""Markdown 资源的简单 frontmatter 解析。"""


class ResourceError(ValueError):
    """资源格式无效或无法使用。"""


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    """解析开头的单行 key: value 元信息，原样返回结束标记后的正文。

    frontmatter 必须用两行独立的 --- 包围。支持空行、整行注释和
    值两端配对的单引号或双引号；不实现完整 YAML 语法。
    必填字段（例如技能的 description）由调用方校验。
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ResourceError("Missing opening frontmatter delimiter (---) on line 1")

    metadata: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        stripped = line.strip()
        if stripped == "---": # 遇到结束处的---，返回上面已经解析的元信息
            return metadata, "".join(lines[line_number:])
        if not stripped or stripped.startswith("#"):
            continue

        key, separator, value = stripped.partition(":")
        key = key.strip()
        if not separator or not key:
            raise ResourceError(f"Invalid frontmatter key: value pair on line {line_number}")
        if key in metadata:
            raise ResourceError(f"Duplicate frontmatter key {key!r} on line {line_number}")

        value = value.strip()
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        metadata[key] = value

    raise ResourceError("Missing closing frontmatter delimiter (---)")
