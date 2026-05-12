param(
    [ValidateSet("all", "pretrain", "adapt")]
    [string]$Stage = "all",
    [string[]]$FoldNames = @("fold1", "fold2", "fold3", "fold4"),
    [int[]]$Shots = @(1, 5),
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$folds = @(
    @{
        Name = "fold1"
        Config = "configs/curvature_fold1_pretrain.yaml"
        TestDatasets = @("Airport", "Actor")
    },
    @{
        Name = "fold2"
        Config = "configs/curvature_fold2_pretrain.yaml"
        TestDatasets = @("Cora", "Disease")
    },
    @{
        Name = "fold3"
        Config = "configs/curvature_fold3_pretrain.yaml"
        TestDatasets = @("CiteSeer")
    },
    @{
        Name = "fold4"
        Config = "configs/curvature_fold4_pretrain.yaml"
        TestDatasets = @("PubMed", "Cornell")
    }
)

function Get-PretrainDatasetNames {
    param(
        [string]$ConfigPath
    )

    $lines = Get-Content $ConfigPath
    $datasets = @()
    $capture = $false

    foreach ($line in $lines) {
        if ($line -match "^pretrain_single_graph_data:") {
            $capture = $true
            continue
        }

        if ($capture) {
            if ($line -match "^\s*-\s*(.+?)\s*$") {
                $datasets += $matches[1]
                continue
            }

            if ($line -notmatch "^\s+") {
                break
            }
        }
    }

    return $datasets
}

function New-AdaptConfigFile {
    param(
        [string]$FoldName,
        [string]$BaseConfigPath,
        [string]$CheckpointPath,
        [string]$TestDataset,
        [int]$Shot
    )

    $lines = Get-Content $BaseConfigPath
    $tempDir = "configs/generated"
    if (-not (Test-Path $tempDir)) {
        New-Item -ItemType Directory -Path $tempDir | Out-Null
    }

    $tempPath = Join-Path $tempDir "${FoldName}_${TestDataset}_${Shot}shot_adapt.yaml"
    $newLines = @()
    $inMultiBlock = $false

    foreach ($line in $lines) {
        if ($line -match "^pretrain_multi_graph_data:") {
            $newLines += "pretrain_multi_graph_data: []"
            $inMultiBlock = $true
            continue
        }

        if ($inMultiBlock) {
            if ($line -match "^\s*-\s+") {
                continue
            }
            if ($line -match "^\s+") {
                continue
            }
            $inMultiBlock = $false
        }

        $newLines += $line
    }

    $newLines += "pretrained_checkpoint: $CheckpointPath"
    $newLines += "data_name: $TestDataset"
    $newLines += "task_type: node_cls"
    $newLines += "metric: acc"
    $newLines += "k_shot: $Shot"

    Set-Content -Path $tempPath -Value $newLines -Encoding UTF8
    return $tempPath
}

foreach ($fold in $folds) {
    $foldName = $fold.Name
    if ($FoldNames -notcontains $foldName) {
        continue
    }

    $configPath = $fold.Config
    $testDatasets = $fold.TestDatasets

    if ($Stage -eq "all" -or $Stage -eq "pretrain") {
        Write-Host "==================== $foldName: pretrain ====================" -ForegroundColor Cyan
        & $PythonExe main.py --run_type pretrain --config_load_path $configPath
        if ($LASTEXITCODE -ne 0) {
            throw "Pretraining failed for $foldName"
        }
    }

    $pretrainDatasets = Get-PretrainDatasetNames -ConfigPath $configPath
    $dirName = [string]::Join("_", $pretrainDatasets)
    $checkpoint = "checkpoints/pretrain/$dirName/pretrain_final_model.pth"

    if ($Stage -eq "all" -or $Stage -eq "adapt") {
        if (-not (Test-Path $checkpoint)) {
            throw "Checkpoint not found for $foldName: $checkpoint"
        }

        foreach ($testDataset in $testDatasets) {
            foreach ($shot in $Shots) {
                $adaptConfig = New-AdaptConfigFile `
                    -FoldName $foldName `
                    -BaseConfigPath $configPath `
                    -CheckpointPath $checkpoint `
                    -TestDataset $testDataset `
                    -Shot $shot

                Write-Host "==================== $foldName: $testDataset ${shot}-shot ====================" -ForegroundColor Yellow
                & $PythonExe main.py `
                    --run_type adapt `
                    --config_load_path $adaptConfig

                if ($LASTEXITCODE -ne 0) {
                    throw "Adaptation failed for $foldName / $testDataset / ${shot}-shot"
                }
            }
        }
    }
}

Write-Host "All curvature experiments finished." -ForegroundColor Green
