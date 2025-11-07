# config_v5.py
"""
v5: Simplified configuration leveraging production defaults
"""

@dataclass
class HADSConfigV5:
    """v5: Streamlined configuration"""
    # Rule Engine
    durable_rules_timeout: int = 1000  # ms
    max_rule_complexity: int = 1000
    
    # Event Bus
    kafka_brokers: List[str] = None
    event_timeout: int = 5000
    
    # LLM
    max_tokens: int = 4000
    temperature: float = 0.1
    
    # ECE Refinement
    max_refine_iters: int = 3
    tau_refine: float = 0.35
    
    def __post_init__(self):
        if self.kafka_brokers is None:
            self.kafka_brokers = ["localhost:9092"]