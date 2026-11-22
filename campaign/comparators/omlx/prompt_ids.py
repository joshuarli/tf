import json
import sys
from pathlib import Path

from jinja2 import Environment, nodes
from jinja2.ext import Extension
from tokenizers import Tokenizer


class Generation(Extension):
    tags = {'generation'}

    def parse(self, parser):
        line = next(parser.stream).lineno
        body = parser.parse_statements(('name:endgeneration',), drop_needle=True)
        return nodes.CallBlock(self.call_method('render'), [], [], body).set_lineno(line)

    def render(self, caller):
        return caller()


def prompt_ids(model_dir, text):
    tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    template = Environment(extensions=[Generation]).from_string((model_dir / 'chat_template.jinja').read_text())
    rendered = template.render(messages=[{'role': 'user', 'content': text}], add_generation_prompt=True)
    return tokenizer.encode(rendered).ids


def main():
    model_dir, source, size, output = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    ids = prompt_ids(model_dir, source.read_text())
    if size == 16384:
        if len(ids) < 16384:
            raise ValueError('source prompt is shorter than 16K tokens')
        ids = ids[:16380] + ids[-4:]
    if len(ids) + 512 > 32768:
        raise ValueError('prompt plus completion exceeds the 32K cap')
    output.write_text(json.dumps(ids))
    print(len(ids))


if __name__ == '__main__':
    main()
