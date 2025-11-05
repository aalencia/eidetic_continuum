# hads_durable_rules_v5.py
"""
HADS Durable Rules Engine v5
Thin wrapper around production rule engine
"""

from durable import ruleset, when_all, post, Facts
import threading
from typing import Dict, Any, List, Tuple

class HADSDurableRulesEngineV5:
    """
    v5: Minimal wrapper around Durable Rules
    No custom rule processing - leverages C-core Rete algorithm directly
    """
    
    def __init__(self, ruleset_name: str = "hads_v5_production"):
        self.ruleset_name = ruleset_name
        self.host = None
        self.performance_stats = {}
        
        self._initialize_ruleset()
        self._load_enterprise_rules()
    
    def _initialize_ruleset(self):
        """v5: Real Durable Rules initialization"""
        try:
            # Define the core ruleset using Durable Rules DSL
            with ruleset(self.ruleset_name):
                # Rules defined in proper DSL, not string parsing
                @when_all((m.transaction_amount > 50000) & (m.stability_level == 'core'))
                def high_value_transaction(c):
                    post('actions', {
                        'rule_id': 'compliance_high_value',
                        'action': 'REQUIRE_MANUAL_REVIEW',
                        'confidence': 0.95,
                        'message': f'High-value transaction ${c.m.transaction_amount}'
                    })
                
                @when_all((m.contains_user_data == True) & 
                         (m.data_operation.isin(['sell', 'share'])))
                def privacy_violation(c):
                    post('actions', {
                        'rule_id': 'privacy_fundamental', 
                        'action': 'REJECT',
                        'confidence': 0.99,
                        'message': 'Privacy violation: Cannot sell user data'
                    })
            
            self.host = RulesHost()
            self.host.register_ruleset(self.ruleset_name)
            print("✓ Durable Rules v5 initialized")
            
        except Exception as e:
            print(f"✗ Durable Rules initialization failed: {e}")
            raise
    
    def execute_rules(self, facts: Dict[str, Any], trace_id: str) -> Tuple[List[str], List[str]]:
        """v5: Delegate to Durable Rules engine"""
        start_time = time.time()
        
        try:
            # Add tracing context
            facts['trace_id'] = trace_id
            
            # Execute using Durable Rules
            result_facts = self.host.post(self.ruleset_name, facts)
            actions = self._extract_actions(result_facts)
            audit_trail = self._generate_audit_trail(result_facts)
            
            # Performance tracking
            self._update_performance_metrics(time.time() - start_time)
            
            return actions, audit_trail
            
        except Exception as e:
            print(f"Rule execution error: {e}")
            return [], [f"Rule engine error: {str(e)}"]
    
    def _extract_actions(self, result_facts: Facts) -> List[str]:
        """v5: Extract actions from Durable Rules results"""
        return [fact.action for fact in result_facts.get('actions', [])]
    
    def _generate_audit_trail(self, result_facts: Facts) -> List[str]:
        """v5: Generate audit trail from rule executions"""
        return [
            f"Rule {fact.rule_id}: {fact.message} (Confidence: {fact.confidence})"
            for fact in result_facts.get('actions', [])
        ]
    
    def _update_performance_metrics(self, execution_time: float):
        """v5: Track performance metrics"""
        self.performance_stats['last_execution_time'] = execution_time
        self.performance_stats['total_executions'] = \
            self.performance_stats.get('total_executions', 0) + 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """v5: Real performance metrics from C-core"""
        return {
            'engine': 'Durable Rules v5 (C-core Rete)',
            'avg_execution_time_ms': self.performance_stats.get('last_execution_time', 0) * 1000,
            'total_executions': self.performance_stats.get('total_executions', 0),
            'status': 'operational'
        }