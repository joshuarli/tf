import unittest

try:
    from jinja2 import Environment
    from prompt_ids import Generation
except ModuleNotFoundError:
    Environment = None


@unittest.skipIf(Environment is None, 'prompt tokenization dependencies unavailable')
class PromptTemplateTests(unittest.TestCase):
    def test_generation_block_renders_its_content(self):
        template = Environment(extensions=[Generation]).from_string('before{% generation %}{{ value }}{% endgeneration %}after')
        self.assertEqual(template.render(value='answer'), 'beforeanswerafter')
