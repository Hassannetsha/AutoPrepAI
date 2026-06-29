"""
Update all import paths to match the new layered folder structure.
Run from project root after moving files.
"""
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordered replacements: (old_pattern, new_pattern)
# Applied in order, most specific first
REPLACEMENTS = [
    # backend.* → new locations (most specific first)
    ("from backend.Routes", "from presentation.api.routes"),
    ("from backend.database", "from data_access.database.connection"),
    ("from backend.models", "from data_access.database.models"),
    ("from backend.settings", "from config.settings"),
    ("from backend.b2_service", "from data_access.storage.b2_service"),
    ("from backend.schemas", "from presentation.api.schemas"),
    ("from backend.ml_service", "from business_logic.cleaning_coordinator.ml_service"),
    ("from backend.", "from data_access."),
    ("import backend", "import data_access"),

    # auth.* → business_logic.auth.*
    ("from auth.", "from business_logic.auth."),
    ("from auth import", "from business_logic.auth import"),
    ("import auth", "import business_logic.auth"),

    # services.* → ml_layer.*
    ("from services.feature_engineering_service", "from ml_layer.features.engineering_service"),
    ("from services.feature_selection_service", "from ml_layer.features.selection_service"),
    ("from services.nlp_service", "from ml_layer.nlp.nlp_service"),
    ("from services.encoding_service", "from ml_layer.encoding.encoding_service"),
    ("from services.scaling_service", "from ml_layer.scaling.scaling_service"),
    ("from services.outliers_service", "from ml_layer.outliers.outliers_service"),
    ("from services.", "from ml_layer."),

    # agents.* → ml_layer.agents.*
    ("from agents.", "from ml_layer.agents."),

    # Root-level pipeline modules → business_logic.cleaning_coordinator
    ("from pipeline_builder", "from business_logic.cleaning_coordinator.pipeline_builder"),
    ("import pipeline_builder", "from business_logic.cleaning_coordinator import pipeline_builder"),
    ("from pipeline_node", "from business_logic.cleaning_coordinator.pipeline_node"),
    ("import pipeline_node", "from business_logic.cleaning_coordinator import pipeline_node"),
    ("from execution_condition", "from business_logic.cleaning_coordinator.execution_condition"),
    ("import execution_condition", "from business_logic.cleaning_coordinator import execution_condition"),
    ("from parameter_resolver", "from business_logic.cleaning_coordinator.parameter_resolver"),
    ("import parameter_resolver", "from business_logic.cleaning_coordinator import parameter_resolver"),
    ("from agent_params", "from business_logic.cleaning_coordinator.agent_params"),
    ("import agent_params", "from business_logic.cleaning_coordinator import agent_params"),
    ("from data_context", "from business_logic.cleaning_coordinator.data_context"),
    ("import data_context", "from business_logic.cleaning_coordinator import data_context"),
    ("from intent", "from business_logic.cleaning_coordinator.intent"),
    ("import intent", "from business_logic.cleaning_coordinator import intent"),

    # pipeline import (must be after more specific pipeline_* rules)
    ("from pipeline import", "from business_logic.cleaning_coordinator.pipeline import"),
    ("import pipeline", "from business_logic.cleaning_coordinator import pipeline"),

    # api_key_manager
    ("from api_key_manager", "from business_logic.services.api_key_manager"),
    ("import api_key_manager", "from business_logic.services import api_key_manager"),

    # utils.*
    ("import utils.utilities as utilities", "from business_logic.services import session_store as utilities"),
    ("from utils.utilities import", "from business_logic.services.session_store import"),
    ("from utils.column_detector", "from ml_layer.utils.column_detector"),
    ("from utils.retry_handler", "from business_logic.services.retry_handler"),
]

def update_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def main():
    changed = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip hidden dirs, node_modules, __pycache__, .git, etc.
        skip_dirs = {"__pycache__", ".git", "node_modules", ".github", ".pytest_cache"}
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            if update_file(fpath):
                changed.append(os.path.relpath(fpath, PROJECT_ROOT))

    print(f"Updated {len(changed)} files:")
    for f in sorted(changed):
        print(f"  {f}")

if __name__ == "__main__":
    main()
