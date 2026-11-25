| case | base | candidate | reply bytes |
| --- | --- | --- | --- |
| code-c01 | pass | pass | differ |
| code-c02 | fail | pass | differ |
| code-c03 | pass | pass | differ |
| structured-s01 | fail | fail | equal |
| structured-s02 | fail | fail | equal |
| structured-s09 | fail | fail | equal |
| reasoning-gsm8k-167 | pass | pass | equal |
| reasoning-gsm8k-83 | pass | pass | equal |
| reasoning-mmlu-163 | pass | pass | equal |
| retrieval-t01 | pass | pass | differ |
| retrieval-t02 | pass | pass | differ |

6/11 cases byte-equal; base {'cases': 11, 'failed': 4, 'kinds': {'code': {'cases': 3, 'failed': 1, 'passed': 2}, 'reasoning': {'cases': 3, 'failed': 0, 'passed': 3}, 'retrieval': {'cases': 2, 'failed': 0, 'passed': 2}, 'structured': {'cases': 3, 'failed': 3, 'passed': 0}}, 'passed': 7}, candidate {'cases': 11, 'failed': 3, 'kinds': {'code': {'cases': 3, 'failed': 0, 'passed': 3}, 'reasoning': {'cases': 3, 'failed': 0, 'passed': 3}, 'retrieval': {'cases': 2, 'failed': 0, 'passed': 2}, 'structured': {'cases': 3, 'failed': 3, 'passed': 0}}, 'passed': 8}
