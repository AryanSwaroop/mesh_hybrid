
if (Test-Path -Path "__pycache__") {
    Remove-Item -Path "__pycache__" -Recurse -Force
}
if (Test-Path -Path "src/mesh_rl/__pycache__") {
    Remove-Item -Path "src/mesh_rl/__pycache__" -Recurse -Force
}
