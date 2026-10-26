"""AdaRoute Selection Domain Package (Week-4 Revamp)."""

from .candidates import CandidateDeployment, CandidateResolver
from .catalog import (
    CapabilityProfile,
    DeploymentDefinition,
    ModelDefinition,
    ProviderDefinition,
    ReplicaDefinition,
    RuntimeSnapshot,
    SystemCatalog,
)
from .constraints import EligibilityDecision, HardConstraintEngine
from .estimator import DeploymentOutputEstimator, OutputEstimate
from .explanations import ExplanationBuilder, RoutingExplanation
from .features import CandidateFeatures, FeatureSnapshotBuilder, ProvenanceValue
from .normalization import NormalizationDomain, NormalizationEngine
from .objectives import (
    CandidateScoreRecord,
    ConfigurableObjectiveEngine,
    ObjectiveDefinition,
)
from .pareto import ParetoFrontierEngine
from .selector import RoutingDecision, UnifiedModelSelector
from .session import SessionState, SessionStore
from .signals import SemanticSignals, SignalExtractor

__all__ = [
    "CandidateDeployment",
    "CandidateFeatures",
    "CandidateResolver",
    "CandidateScoreRecord",
    "CapabilityProfile",
    "ConfigurableObjectiveEngine",
    "DeploymentDefinition",
    "DeploymentOutputEstimator",
    "EligibilityDecision",
    "ExplanationBuilder",
    "FeatureSnapshotBuilder",
    "HardConstraintEngine",
    "ModelDefinition",
    "NormalizationDomain",
    "NormalizationEngine",
    "ObjectiveDefinition",
    "OutputEstimate",
    "ParetoFrontierEngine",
    "ProvenanceValue",
    "ProviderDefinition",
    "ReplicaDefinition",
    "RoutingDecision",
    "RoutingExplanation",
    "RuntimeSnapshot",
    "SemanticSignals",
    "SessionState",
    "SessionStore",
    "SignalExtractor",
    "SystemCatalog",
    "UnifiedModelSelector",
]
