@echo off
setlocal

set "ROOT=%~dp0"
set "PS1=%ROOT%tools\orchestrate_inference_server.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%

