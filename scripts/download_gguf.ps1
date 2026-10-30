$modelsDir = "$PSScriptRoot\..\data\models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

$files = @{
    "qwen2.5-0.5b-instruct-q4_k_m.gguf" = 491400032
    "qwen2.5-0.5b-instruct-q8_0.gguf"   = 675710816
    "qwen2.5-0.5b-instruct-fp16.gguf"   = 1266425696
}

foreach ($item in $files.GetEnumerator()) {
    $file = $item.Key
    $expectedSize = $item.Value
    $targetPath = "$modelsDir\$file"
    
    if (Test-Path $targetPath) {
        $actualSize = (Get-Item $targetPath).Length
        if ($actualSize -eq $expectedSize) {
            Write-Host "File $file already fully downloaded ($actualSize bytes)."
            continue
        } else {
            Write-Host "Incomplete file $file ($actualSize != $expectedSize). Resuming..."
        }
    }

    $url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$file"
    Write-Host "Downloading $file..."
    # -C - resumes partially downloaded files
    curl.exe -L -C - -o $targetPath $url
    Write-Host "Finished downloading $file"
}
