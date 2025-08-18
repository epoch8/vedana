"""
conveniently borrowed from https://github.com/yym68686/md2tgmd
"""

import re


def find_all_index(str, pattern):
    index_list = [0]
    for match in re.finditer(pattern, str, re.MULTILINE):
        if match.group(1) is not None:
            start = match.start(1)
            end = match.end(1)
            index_list += [start, end]
    index_list.append(len(str))
    return index_list


def replace_all(text, pattern, function):
    poslist = [0]
    strlist = []
    originstr = []
    poslist = find_all_index(text, pattern)
    for i in range(1, len(poslist[:-1]), 2):
        start, end = poslist[i : i + 2]
        strlist.append(function(text[start:end]))
    for i in range(0, len(poslist), 2):
        j, k = poslist[i : i + 2]
        originstr.append(text[j:k])
    if len(strlist) < len(originstr):
        strlist.append("")
    else:
        originstr.append("")
    new_list = [item for pair in zip(originstr, strlist) for item in pair]
    return "".join(new_list)


def escapeshape(text):
    return "▎*" + " ".join(text.split()[1:]) + "*\n\n"


def escapeminus(text):
    return "\\" + text


def escapeminus2(text):
    return r"@+>@"


def escapebackquote(text):
    return r"\`\`"


def escapebackquoteincode(text):
    return r"@->@"


def escapeplus(text):
    return "\\" + text


def escape_all_backquote(text):
    return "\\" + text


def dedent_space(text):
    import textwrap

    return "\n\n" + textwrap.dedent(text).strip() + "\n\n"


def split_code(text):
    split_list = []
    if len(text) > 2300:
        split_str_list = text.split("\n\n")

        conversation_len = len(split_str_list)
        message_index = 1
        while message_index < conversation_len:
            if split_str_list[message_index].startswith("    "):
                split_str_list[message_index - 1] += "\n\n" + split_str_list[message_index]
                split_str_list.pop(message_index)
                conversation_len = conversation_len - 1
            else:
                message_index = message_index + 1

        split_index = 0
        for index, _ in enumerate(split_str_list):
            if len("".join(split_str_list[:index])) < len(text) // 2:
                split_index += 1
                continue
            else:
                break
        str1 = "\n\n".join(split_str_list[:split_index])
        if not str1.strip().endswith("```"):
            str1 = str1 + "\n```"
        split_list.append(str1)
        code_type = text.split("\n")[0]
        str2 = "\n\n".join(split_str_list[split_index:])
        str2 = code_type + "\n" + str2
        if not str2.strip().endswith("```"):
            str2 = str2 + "\n```"
        split_list.append(str2)
    else:
        split_list.append(text)

    if len(split_list) > 1:
        split_list = "\n@|@|@|@\n\n".join(split_list)
    else:
        split_list = split_list[0]
    return split_list


def find_lines_with_char(s, char, min_count):
    """
    Find lines containing a specific character at least min_count times.

    Args:
        s (str): String to process
        char (str): Character to count
        min_count (int): Minimum occurrence count

    Returns:
        str: String with escaped characters
    """
    lines = s.split("\n")  # Split string by lines

    for index, line in enumerate(lines):
        if re.sub(r"```", "", line).count(char) % 2 != 0 or (
            not line.strip().startswith("```") and line.count(char) % 2 != 0
        ):
            lines[index] = replace_all(lines[index], r"\\`|(`)", escape_all_backquote)

    return "\n".join(lines)


def escape(text, flag=0, italic=True):
    """
    Convert Markdown to Telegram Markdown format.

    Args:
        text (str): Input Markdown text
        flag (int): Processing flag
        italic (bool): Whether to process italic formatting

    Returns:
        str: Telegram-formatted markdown
    """
    # In all other places characters
    # _ * [ ] ( ) ~ ` > # + - = | { } . !
    # must be escaped with the preceding character '\'.
    text = re.sub(r"\\\[", "@->@", text)
    text = re.sub(r"\\]", "@<-@", text)
    text = re.sub(r"\\\(", "@-->@", text)
    text = re.sub(r"\\\)", "@<--@", text)
    if flag:
        text = re.sub(r"\\\\", "@@@", text)
    text = re.sub(r"\\`", "@<@", text)
    text = re.sub(r"\\", r"\\\\", text)
    if flag:
        text = re.sub(r"@{3}", r"\\\\", text)
    # _italic_
    if italic:
        text = re.sub(r"_(.*?)_", "@@@\\1@@@", text)
        text = re.sub(r"_", r"\_", text)
        text = re.sub(r"@{3}(.*?)@{3}", "_\\1_", text)
    else:
        text = re.sub(r"_", r"\_", text)

    text = re.sub(r"\*{2}(.*?)\*{2}", "@@@\\1@@@", text)
    text = re.sub(r"\n{1,2}\*\s", "\n\n• ", text)
    text = re.sub(r"\*", r"\*", text)
    text = re.sub(r"@{3}(.*?)@{3}", "*\\1*", text)
    text = re.sub(r"!?\[(.*?)]\((.*?)\)", "@@@\\1@@@^^^\\2^^^", text)
    text = re.sub(r"\[", r"\[", text)
    text = re.sub(r"]", r"\]", text)
    text = re.sub(r"\(", r"\(", text)
    text = re.sub(r"\)", r"\)", text)
    text = re.sub(r"@->@", r"\[", text)
    text = re.sub(r"@<-@", r"\]", text)
    text = re.sub(r"@-->@", r"\(", text)
    text = re.sub(r"@<--@", r"\)", text)
    text = re.sub(r"@{3}(.*?)@{3}\^{3}(.*?)\^{3}", "[\\1](\\2)", text)

    # ~strikethrough~
    text = re.sub(r"~{2}(.*?)~{2}", "@@@\\1@@@", text)
    text = re.sub(r"~", r"\~", text)
    text = re.sub(r"@{3}(.*?)@{3}", "~\\1~", text)

    text = re.sub(r"\n>\s", "\n@@@ ", text)
    text = re.sub(r">", r"\>", text)
    text = re.sub(r"@{3}", ">", text)

    text = replace_all(text, r"(^#+\s.+?\n+)|```[\D\d\s]+?```", escapeshape)
    text = re.sub(r"#", r"\#", text)
    text = replace_all(text, r"(\+)|\n[\s]*-\s|```[\D\d\s]+?```|`[\D\d\s]*?`", escapeplus)

    # Numbered lists
    text = re.sub(r"\n{1,2}(\s*\d{1,2}\.\s)", "\n\n\\1", text)

    # Replace - outside code blocks
    text = replace_all(text, r"```[\D\d\s]+?```|(-)", escapeminus2)
    text = re.sub(r"-", "@<+@", text)
    text = re.sub(r"@\+>@", "-", text)

    text = re.sub(r"\n{1,2}(\s*)-\s", "\n\n\\1• ", text)
    text = re.sub(r"@<+@", r"\-", text)
    text = replace_all(text, r"(-)|\n[\s]*-\s|```[\D\d\s]+?```|`[\D\d\s]*?`", escapeminus)
    text = re.sub(r"```([\D\d\s]+?)```", "@@@\\1@@@", text)
    # Replace backticks in code blocks
    text = replace_all(text, r"\@\@\@[\s\d\D]+?\@\@\@|(`)", escapebackquoteincode)
    text = re.sub(r"`", r"\`", text)
    text = re.sub(r"@<@", r"\`", text)
    text = re.sub(r"@->@", "`", text)
    text = re.sub(r"\s`\\`\s", r" `\\\\\` ", text)

    text = replace_all(text, r"(``)", escapebackquote)
    text = re.sub(r"@{3}([\D\d\s]+?)@{3}", "```\\1```", text)
    text = re.sub(r"=", r"\=", text)
    text = re.sub(r"\|", r"\|", text)
    # text = re.sub(r"\@\!\@", '||', text)
    text = re.sub(r"{", r"\{", text)
    text = re.sub(r"}", r"\}", text)
    text = re.sub(r"\.", r"\.", text)
    text = re.sub(r"!", r"\!", text)
    text = find_lines_with_char(text, "`", 5)
    text = replace_all(text, r"(\n+\x20*```[\D\d\s]+?```\n+)", dedent_space)
    return text


test_text = r"""
# title

### `probs.scatter_(1, ind`

**bold**
```
# comment
print(qwer) # ferfe
ni1
```
# bn


# b

# Header
## Subheader

[1.0.0](http://version.com)
![1.0.0](http://version.com)

- item 1 -
    - item 1 -
    - item 1 -
* item 2 #
* item 3 ~

1. item 1
2. item 2

1. item 1
```python

# comment
print("1.1\n")_
\subsubsection{1.1}
- item 1 -
```
2. item 2

sudo apt install package # Install command

\subsubsection{1.1}

And simple text `with-dashes` `with+plus` + some - **symbols**.

    ```
    print("Hello, World!") -
    app.listen(PORT, () => {
        console.log(`Server is running on http://localhost:${PORT}`);
    });
    ```

Cxy = abs (Pxy)**2/ (Pxx*Pyy)

`a`a-b-c`n`
\[ E[X^4] = \int_{-\infty}^{\infty} x^4 f(x) dx \]

`-a----++++`++a-b-c`-n-`
`[^``]*`a``b-c``d``
# pattern = r"`[^`]*`-([^`-]*)"``
w`-a----`ccccc`-n-`bbbb``a

1. Open VSCode terminal: Go to `View` > `Terminal` or use `Ctrl+``

How to write: `line.strip().startswith("```")`?

`View` > `Terminal`

Escape example: `\``

- `Path.open()` method opens the `README.md` file with UTF-8 encoding.

3. `(`

3. Parentheses example: `(`

According to Euler's totient function, for \( n = p_1^{k_1} \times p_2^{k_2} \times \cdots \times p_r^{k_r} \) (where \( p_1, p_2, \ldots, p_r \) are distinct primes):

\[ \varphi(n) = n \left(1 - \frac{1}{p_1}\right) \left(1 - \frac{1}{p_2}\right) \cdots \left(1 - \frac{1}{p_r}\right) \]

Therefore:

\[ \varphi(35) = 35 \left(1 - \frac{1}{5}\right) \left(1 - \frac{1}{7}\right) \]

Calculating step by step:

\[ \varphi(35) = 35 \left(\frac{4}{5}\right) \left(\frac{6}{7}\right) \]

\[ \varphi(35) = 35 \times \frac{24}{35} \]

\[ \varphi(35) = 24 \]

To calculate acceleration \( a \), use the formula:

\[ a = \frac{\Delta v}{\Delta t} \]

Where:
- \(\Delta v\) is the change in velocity
- \(\Delta t\) is the change in time

Given:
- Initial velocity \( v_0 = 0 \) m/s
- Final velocity \( v = 27.8 \) m/s
- Time \( \Delta t = 3.85 \) s

Substituting values:

\[ a = \frac{27.8 \, \text{m/s} - 0 \, \text{m/s}}{3.85 \, \text{s}} \]

Calculating:

\[ a = \frac{27.8}{3.85} \approx 7.22 \, \text{m/s}^2 \]

Therefore, the car's acceleration is approximately 7.22 m/s².

Thus, Euler's totient function \( \varphi(35) \) equals 24.
"""

if __name__ == "__main__":
    test_text_escaped = escape(test_text)
    print(test_text_escaped)
