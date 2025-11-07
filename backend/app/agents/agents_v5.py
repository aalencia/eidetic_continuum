from ..core.durable_rules_v5 import HADSDurableRulesEngineV5
from typing import Any, Dict, List
# hads_agents_v5.py
"""
HADS Agents v5 - Rule-Aware AI Components
"""

class HADSEnhancedLLMV5:
    """v5: LLM with built-in rule awareness"""
    
    def __init__(self, rule_engine: HADSDurableRulesEngineV5):
        self.rule_engine = rule_engine
        self.rule_constraints = self._load_rule_constraints()
    
    def generate_with_constraints(self, input_text: str, constraints: Dict, trace_id: str) -> Dict[str, Any]:
        """v5: Rule-constrained generation"""
        # Pre-validate against critical rules
        critical_violations = self._check_critical_rules(input_text, trace_id)
        if critical_violations:
            return self._build_constrained_response(input_text, critical_violations)
        
        # Generate with rule awareness
        prompt = self._build_rule_aware_prompt(input_text, constraints)
        response = self.generate(prompt)
        
        return self._enrich_with_rule_context(response, constraints)
    
    def _check_critical_rules(self, input_text: str, trace_id: str) -> List[str]:
        """v5: Early critical rule checking"""
        test_facts = {'input_text': input_text, 'quick_check': True}
        actions, _ = self.rule_engine.execute_rules(test_facts, trace_id)
        return [action for action in actions if 'REJECT' in action]

class HADSTestHarnessV5:
    """v5: Enhanced testing with real rule engine"""
    
    @staticmethod
    def run_compliance_suite_v5():
        """v5: Test with real Durable Rules engine"""
        engine = HADSECEEngineV5.create_production_instance()
        
        test_cases = [
            {
                "input": "Process $75,000 international wire transfer",
                "expected": "requires_manual_review",
                "rules_fired": ["compliance_high_value"]
            }
        ]
        
        for test in test_cases:
            result = engine.haze_loop_enhanced_v5(test["input"])
            
            # v5: Validate rule engine behavior
            assert result.status == test["expected"]
            assert any(rule in str(result.rule_validations) for rule in test["rules_fired"])
            
            print(f"✓ Test passed: {test['input']}")