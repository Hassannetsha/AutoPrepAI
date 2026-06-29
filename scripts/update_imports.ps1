# PowerShell script to update all import paths for the new layered structure

$root = "D:\cs\GP\AutoPrepAI\AutoPrepAI"

$replacements = @(
    # backend.* → new locations (most specific first)
    @{old="from backend.Routes"; new="from presentation.api.routes"}
    @{old="from backend.database"; new="from data_access.database.connection"}
    @{old="from backend.models"; new="from data_access.database.models"}
    @{old="from backend.settings"; new="from config.settings"}
    @{old="from backend.b2_service"; new="from data_access.storage.b2_service"}
    @{old="from backend.schemas"; new="from presentation.api.schemas"}
    @{old="from backend.ml_service"; new="from business_logic.cleaning_coordinator.ml_service"}
    @{old="from backend."; new="from data_access."}
    @{old="import backend"; new="import data_access"}

    # auth.* → business_logic.auth.*
    @{old="from auth."; new="from business_logic.auth."}
    @{old="from auth import"; new="from business_logic.auth import"}
    @{old="import auth"; new="import business_logic.auth"}

    # services.* → ml_layer.*
    @{old="from services.feature_engineering_service"; new="from ml_layer.features.engineering_service"}
    @{old="from services.feature_selection_service"; new="from ml_layer.features.selection_service"}
    @{old="from services.nlp_service"; new="from ml_layer.nlp.nlp_service"}
    @{old="from services.encoding_service"; new="from ml_layer.encoding.encoding_service"}
    @{old="from services.scaling_service"; new="from ml_layer.scaling.scaling_service"}
    @{old="from services.outliers_service"; new="from ml_layer.outliers.outliers_service"}
    @{old="from services."; new="from ml_layer."}

    # agents.* → ml_layer.agents.*
    @{old="from agents."; new="from ml_layer.agents."}

    # Root-level pipeline modules → business_logic.cleaning_coordinator
    @{old="from pipeline_builder"; new="from business_logic.cleaning_coordinator.pipeline_builder"}
    @{old="import pipeline_builder"; new="from business_logic.cleaning_coordinator import pipeline_builder"}
    @{old="from pipeline_node"; new="from business_logic.cleaning_coordinator.pipeline_node"}
    @{old="import pipeline_node"; new="from business_logic.cleaning_coordinator import pipeline_node"}
    @{old="from execution_condition"; new="from business_logic.cleaning_coordinator.execution_condition"}
    @{old="import execution_condition"; new="from business_logic.cleaning_coordinator import execution_condition"}
    @{old="from parameter_resolver"; new="from business_logic.cleaning_coordinator.parameter_resolver"}
    @{old="import parameter_resolver"; new="from business_logic.cleaning_coordinator import parameter_resolver"}
    @{old="from agent_params"; new="from business_logic.cleaning_coordinator.agent_params"}
    @{old="import agent_params"; new="from business_logic.cleaning_coordinator import agent_params"}
    @{old="from data_context"; new="from business_logic.cleaning_coordinator.data_context"}
    @{old="import data_context"; new="from business_logic.cleaning_coordinator import data_context"}
    @{old="from intent"; new="from business_logic.cleaning_coordinator.intent"}
    @{old="import intent"; new="from business_logic.cleaning_coordinator import intent"}

    # pipeline import (must be after more specific pipeline_* rules)
    @{old="from pipeline import"; new="from business_logic.cleaning_coordinator.pipeline import"}
    @{old="import pipeline"; new="from business_logic.cleaning_coordinator import pipeline"}

    # api_key_manager
    @{old="from api_key_manager"; new="from business_logic.services.api_key_manager"}
    @{old="import api_key_manager"; new="from business_logic.services import api_key_manager"}

    # utils.*
    @{old="import utils.utilities as utilities"; new="from business_logic.services import session_store as utilities"}
    @{old="from utils.utilities import"; new="from business_logic.services.session_store import"}
    @{old="from utils.column_detector"; new="from ml_layer.utils.column_detector"}
    @{old="from utils.retry_handler"; new="from business_logic.services.retry_handler"}

    # Flattened nested module dirs
    @{old="from data_standardization."; new="from ml_layer.data_standardization."}
    @{old="from duplicates."; new="from ml_layer.duplicates."}
    @{old="from Encoders."; new="from ml_layer.encoding.encoders."}
    @{old="from data_type_inconsistency_detector"; new="from ml_layer.data_type_inconsistency.detector"}
    @{old="from data_type_inconsistency_resolver"; new="from ml_layer.data_type_inconsistency.resolver"}
    @{old="from outliers."; new="from ml_layer.outliers."}
)

$changedFiles = @()
$totalReplacements = 0

# Collect all Python files (skip __pycache__, .git, node_modules, etc.)
$files = Get-ChildItem -Path $root -Recurse -Filter "*.py" | Where-Object {
    $_.FullName -notmatch "__pycache__" -and
    $_.FullName -notmatch "\\.git" -and
    $_.FullName -notmatch "node_modules" -and
    $_.FullName -notmatch "\\.github"
}

foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    $original = $content
    $fileChanged = $false

    foreach ($repl in $replacements) {
        $old = $repl.old
        $new = $repl.new
        if ($content -match [regex]::Escape($old)) {
            $content = $content -creplace [regex]::Escape($old), $new
            $fileChanged = $true
            $totalReplacements++
        }
    }

    if ($fileChanged) {
        # Use UTF8 without BOM to keep consistent
        [System.IO.File]::WriteAllText($file.FullName, $content, [System.Text.UTF8Encoding]::new($false))
        $relPath = $file.FullName.Substring($root.Length).TrimStart('\')
        $changedFiles += $relPath
    }
}

Write-Output "Updated $($changedFiles.Count) files with $totalReplacements total replacements:"
$changedFiles | Sort-Object | ForEach-Object { Write-Output "  $_" }
