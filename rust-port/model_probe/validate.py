#!/usr/bin/env python3
# Copyright 2026 rt-rs contributors. SPDX-License-Identifier: Apache-2.0
"""Fail closed on an unobserved or misconfigured native diagnostic run."""
import json
import sys

def validate(report):
    if not report['status'] == 'OK':
        raise ValueError(report['status'])
    if not report['kind'] == 'instrumented_native_diagnostic_not_benchmark':
        raise ValueError("Invalid model diagnostic: report['kind'] == 'instrumented_native_diagnostic_not_benchmark'")
    if not report['factory_calls'] == len(report['factories']) > 0:
        raise ValueError("Invalid model diagnostic: report['factory_calls'] == len(report['factories']) > 0")
    if not 0 < report['sample_calls'] == len(report['samples']) <= 16:
        raise ValueError("Invalid model diagnostic: 0 < report['sample_calls'] == len(report['samples']) <= 16")
    if not report['resolved_threads'] == 2:
        raise ValueError("Invalid model diagnostic: report['resolved_threads'] == 2")
    if not report['resolved_max_output_tokens'] == 16:
        raise ValueError("Invalid model diagnostic: report['resolved_max_output_tokens'] == 16")
    if not report['resolved_engine_settings']:
        raise ValueError("Invalid model diagnostic: report['resolved_engine_settings']")
    if not report['resolved_conversation_session_config']:
        raise ValueError("Invalid model diagnostic: report['resolved_conversation_session_config']")
    if not report['resolved_prompt_template']:
        raise ValueError("Invalid model diagnostic: report['resolved_prompt_template']")
    if not report['response']:
        raise ValueError('Conversation returned no response')
    expected = dict(type=2, k=1, p=1, temperature=1, seed=0, has_seed=True)
    for factory in report['factories']:
        if not factory['status'] == 'OK':
            raise ValueError("Invalid model diagnostic: factory['status'] == 'OK'")
        if not factory['backend'] == 3:
            raise ValueError('Expected LiteRT CPU backend enum')
        if not factory['batch_size'] == 1:
            raise ValueError("Invalid model diagnostic: factory['batch_size'] == 1")
    ids = []
    dtypes = set()
    for sample in report['samples']:
        if not sample['status'] == 'OK':
            raise ValueError(sample)
        if not sample['sampling'] == expected:
            raise ValueError(sample)
        if not sample['dtype'] in ('float32', 'float16'):
            raise ValueError(sample)
        shape = sample['shape']
        if not (len(shape) == 3 and shape[0:2] == [1, 1] and (shape[2] > 0)):
            raise ValueError(shape)
        if not len(sample['token_ids']) == 1:
            raise ValueError(sample)
        if not all((isinstance(token, int) and 0 <= token < shape[2] for token in sample['token_ids'])):
            raise ValueError(sample)
        ids.extend(sample['token_ids'])
        dtypes.add(sample['dtype'])
    return {'observed_dtypes': sorted(dtypes), 'sampled_token_ids': ids, 'sampling_calls': len(report['samples']), 'interpretation': 'Real native sampling observed; instrumentation is not a performance benchmark.'}
if __name__ == '__main__':
    with open(sys.argv[1], encoding='utf-8') as source:
        print(json.dumps(validate(json.load(source)), indent=2))
