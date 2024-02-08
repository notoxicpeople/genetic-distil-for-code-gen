import re


def simple_extract_python_code(text):
    # def で始まる行を抽出
    start = text.find("def ")

    # def が見つからない場合は、元のテキストを返す
    if start == -1:
        return text

    # def で始まる行の次の行を抽出
    end = text.find("\n", start)
    while end != -1:
        # 次の行が空行の場合は、次の行を抽出
        while end + 1 < len(text) and text[end + 1] == "\n":
            end = text.find("\n", end + 1)

        # レンジ外の場合またはスペースでない場合は、抽出を終了
        if end + 1 >= len(text) or text[end + 1] != " ":
            break
        end = text.find("\n", end + 1)

    if end == -1:
        return text

    extracted_code = text[:end]

    return extracted_code


# 文字列からコードを抽出するためのメソッド
def extract_python_code(text):
    # コードブロックの抽出
    code_pattern = re.compile(r"(def.*?)(\n\n)", re.DOTALL)
    # インポート文の抽出
    import_pattern = re.compile(r"(\n|^)(import.*?)(\n\n)", re.DOTALL)

    # コードブロックを取得
    defs = code_pattern.finditer(text)
    for match in defs:
        # 最初にコードブロックの正規表現とマッチしたものを返す
        code_block = match.group(1)
        if "return" in code_block.strip().split("\n")[-1]:
            # return文が\nの前に含まれている場合は、インポート文を取得
            imports = import_pattern.findall(text[: match.start()])
            # インポート文を結合
            import_block = "\n".join([imp[1] for imp in imports])
            # インポート文とコードブロックを結合して返す
            return import_block + "\n" + code_block

    return text
