from phenometrix_prosody.evaluate_curriculum import challenge_gate, original_case_gates


def records():
    result = []
    for index in range(32):
        voiced = index < 28
        result.append({'id': str(index), 'metrics': {'augmented': {
            'reference_voiced_frames': 100 if voiced else 0,
            'reference_unvoiced_frames': 0 if voiced else 100,
            'reference_voiced_success_rate': .95 if voiced else None,
            'voicing_false_positive_count': 0, 'voicing_false_positive_rate': 0.}}})
    return result


def test_challenge_gate_cannot_hide_a_bad_case_behind_macro_average():
    values = records()
    assert challenge_gate(values, 'augmented')['challenge_component_pass']
    values[0]['metrics']['augmented']['reference_voiced_success_rate'] = .79
    result = challenge_gate(values, 'augmented')
    assert result['equal_scenario_reference_voiced_success'] > .9
    assert result['voiced_scenarios_below_80percent'] == ['0']
    assert not result['challenge_component_pass'] and not result['final_keep']


def test_unvoiced_false_positives_and_missing_scenarios_cannot_disappear():
    values = records()
    values[-1]['metrics']['augmented']['voicing_false_positive_count'] = 6
    values[-1]['metrics']['augmented']['voicing_false_positive_rate'] = .06
    assert not challenge_gate(values, 'augmented')['challenge_component_pass']
    assert not challenge_gate(values[:-1], 'augmented')['challenge_component_pass']


def test_original_case_gates_require_every_original_case():
    cases = []
    for kind in ['constant', 'observation-change']:
        for hz in [100, 200, 300]:
            cases.append({'scenario': {'id':f'{kind}-{hz}', 'kind':kind}, 'methods':{'mlp':{'regions':{
                'all':{'success_within_50c_all_frames':1.}, 'after-observation-change':{'success_within_50c_all_frames':1.}}}}})
    assert original_case_gates({'scenarios':cases})['stationary_component_pass']
    cases[2]['methods']['mlp']['regions']['all']['success_within_50c_all_frames'] = .94
    assert not original_case_gates({'scenarios':cases})['stationary_component_pass']
