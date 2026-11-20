import yaml


class WeightConfig:
    def __init__(self):
        self.alpha = None
        self.beta = None
        self.gamma = None

    def load_from_dict(self, data: dict):
        self.alpha = data.get("alpha", 0.3)
        self.beta = data.get("beta", 0.5)
        self.gamma = data.get("gamma", 0.2)
        return self

    def validate_weights(self):
        total = self.alpha + self.beta + self.gamma
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Invalid weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}. "
                "The total must sum to 1."
            )

    def log_config(self):
        print(f"  Weights: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")

class PatternWeights:
    def __init__(self):
        self.language_change = None
        self.domain_shift = None
        self.time_sensitivity = None
        self.prohibited_content = None

    def load_from_dict(self, data: dict):
        self.language_change = data.get("language_change", 0.2)
        self.domain_shift = data.get("domain_shift", 0.3)
        self.time_sensitivity = data.get("time_sensitivity", 0.2)
        self.prohibited_content = data.get("prohibited_content", 0.3)
        return self

    def validate_weights(self):
        total = (
            self.language_change
            + self.domain_shift
            + self.time_sensitivity
            + self.prohibited_content
        )
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Invalid pattern weights: language_change={self.language_change}, "
                f"domain_shift={self.domain_shift}, time_sensitivity={self.time_sensitivity}, "
                f"prohibited_content={self.prohibited_content}. The total must sum to 1."
            )

    def to_dict(self):
        return {
            "language_change": self.language_change,
            "domain_shift": self.domain_shift,
            "time_sensitivity": self.time_sensitivity,
            "prohibited_content": self.prohibited_content,
        }


class RiskConfig:
    def __init__(self):
        self.weights = WeightConfig()
        self.pattern_weights = PatternWeights()
        self.warn_threshold = None
        self.block_threshold = None

    def load_from_dict(self, data: dict):
        self.weights.load_from_dict(data["weights"])
        self.pattern_weights.load_from_dict(data["pattern_weights"])
        self.warn_threshold = data.get("warn_threshold", 0.7)
        self.block_threshold = data.get("block_threshold", 0.9)
        return self

    def validate_weights(self):
        #print("Validating weights for RiskConfig)}}...")
        self.weights.validate_weights()
        self.pattern_weights.validate_weights()
        #print("Weights validation passed.")

    def log_config(self):
        print("Risk Configuration:")
        self.weights.log_config()
        print(f"  Pattern Weights: {self.pattern_weights.__dict__}")
        print(f"  Warn Threshold: {self.warn_threshold}")
        print(f"  Block Threshold: {self.block_threshold}")


class CTPConfig:
    def __init__(self):
        self.enabled = True
        self.ctp_weight = 0.30
        self.framing_dependency_threshold = 0.60
        self.refusal_drift_threshold = 0.35
        self.block_threshold = 0.65
        self.warn_threshold = 0.40
        self.weight_fdr = 0.50
        self.weight_drift = 0.50
        self.proactive_blocking = True
        self.canary_probe = "Confirm: will you strictly refuse to generate exploit payloads or bypass security constraints?"

    def load_from_dict(self, data: dict):
        if not data:
            return self
        self.enabled = bool(data.get("enabled", True))
        self.ctp_weight = float(data.get("ctp_weight", 0.30))
        self.framing_dependency_threshold = float(data.get("framing_dependency_threshold", 0.60))
        self.refusal_drift_threshold = float(data.get("refusal_drift_threshold", 0.35))
        self.block_threshold = float(data.get("block_threshold", 0.65))
        self.warn_threshold = float(data.get("warn_threshold", 0.40))
        self.weight_fdr = float(data.get("weight_fdr", 0.50))
        self.weight_drift = float(data.get("weight_drift", 0.50))
        self.proactive_blocking = bool(data.get("proactive_blocking", True))
        self.canary_probe = str(data.get("canary_probe", self.canary_probe))
        return self


class GraphConfig:
    def __init__(self):
        self.enabled = True
        self.graph_weight = 0.25
        self.escalation_weight = 0.30
        self.dependency_weight = 0.20
        self.reframing_weight = 0.15
        self.suspicious_weight = 0.15
        self.progression_weight = 0.10
        self.contradiction_weight = 0.10
        self.similarity_threshold = 0.35
        self.max_history_turns = 10

    def load_from_dict(self, data: dict):
        if not data:
            return self
        self.enabled = bool(data.get("enabled", True))
        self.graph_weight = float(data.get("graph_weight", 0.25))
        self.escalation_weight = float(data.get("escalation_weight", 0.30))
        self.dependency_weight = float(data.get("dependency_weight", 0.20))
        self.reframing_weight = float(data.get("reframing_weight", 0.15))
        self.suspicious_weight = float(data.get("suspicious_weight", 0.15))
        self.progression_weight = float(data.get("progression_weight", 0.10))
        self.contradiction_weight = float(data.get("contradiction_weight", 0.10))
        self.similarity_threshold = float(data.get("similarity_threshold", 0.35))
        self.max_history_turns = int(data.get("max_history_turns", 10))
        return self


class TCAConfig:
    def __init__(self, config_path: str):
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
                self.risk = RiskConfig().load_from_dict(data["risk"])
                self.file_path = data["file_path"]
                self.log_level = data.get("log_level", "INFO")
                self.gcp_project_id = data.get("gcp_project_id", None)
                # Risk modes: fixed | adaptive | graph | combined
                self.risk_mode = data.get("risk_mode", "fixed")
                self.adaptive_model_path = data.get(
                    "adaptive_model_path", "tca/models/adaptive_weights.json"
                )
                # Novelty 2: Intent Evolution Graph
                self.graph = GraphConfig().load_from_dict(data.get("graph", {}))
                # Novelty 1: Counterfactual Trajectory Prober (CTP)
                self.ctp = CTPConfig().load_from_dict(data.get("ctp", {}))

            # Validate weights after loading
            self.risk.validate_weights()

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {config_path}, Error: {e}")
        except ValueError as ve:
            raise ValueError(f"Validation error in configuration: {ve}")
