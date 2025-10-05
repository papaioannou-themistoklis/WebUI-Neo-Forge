import re

from modules import shared


def strip_comments(text):
    if shared.opts.enable_prompt_comments:
        text = re.sub("(^|\n)#[^\n]*(\n|$)", "\n", text)  # whole line comment
        text = re.sub("#[^\n]*(\n|$)", "\n", text)  # in the middle of the line comment

    return text


shared.options_templates.update(
    shared.options_section(
        ("ui_alternatives", "UI Alternatives", "ui"),
        {"enable_prompt_comments": shared.OptionInfo(True, "Enable Comments").info("Ignore the texts between # and the end of the line from the prompts")},
    )
)
