$ErrorActionPreference = "Stop"

$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutName = "Qwen " + [string]([char]0x672C) + [string]([char]0x5730) + [string]([char]0x591A) + [string]([char]0x6A21) + [string]([char]0x6001) + ".lnk"
$shortcutPath = Join-Path $desktop $shortcutName
$launcherPath = Join-Path $PSScriptRoot "start_qwen_voice.ps1"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcherPath`""
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.IconLocation = "$env:WINDIR\System32\shell32.dll,14"
$shortcut.WindowStyle = 7
$shortcut.Description = "Start the Qwen local multimodal project and open its web page"
$shortcut.Save()

Write-Output $shortcutPath
