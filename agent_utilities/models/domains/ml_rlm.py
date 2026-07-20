from ...models.knowledge_graph import RegistryNode, RegistryNodeType


class RLMActorNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.RLM_ACTOR
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    discount_factor: float = 0.99


class OptimizationGoalNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.OPTIMIZATION_GOAL
    target_metric: str = "reward"
    threshold: float = 0.9
    is_active: bool = True


class AgentSwarmNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.AGENT_SWARM
    swarm_size: int = 5
    coordination_strategy: str = "decentralized"


class NeuralNetworkModelNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.NEURAL_NETWORK_MODEL
    architecture: str = "transformer"
    parameters_count: int = 0
    framework: str = "pytorch"


class LSTMNetworkNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.LSTM_NETWORK
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class KronosModelNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.KRONOS_MODEL
    time_horizon: int = 30
    granularity: str = "1d"
    confidence_interval: float = 0.95


class ParetoFrontierEntryNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PARETO_FRONTIER_ENTRY
    objective_1_score: float = 0.0
    objective_2_score: float = 0.0
    is_dominated: bool = False


class StochasticProcessNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.STOCHASTIC_PROCESS
    process_type: str = "markov"
    state_space_size: int = 0


class TransitionMatrixNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.TRANSITION_MATRIX
    dimension: int = 0
    is_sparse: bool = False
