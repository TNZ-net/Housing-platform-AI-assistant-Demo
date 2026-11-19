$script  = Join-Path $PSScriptRoot "run_ingest.bat"
$action  = New-ScheduledTaskAction -Execute $script
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00am
Register-ScheduledTask -TaskName "hp-ingest" -Action $action -Trigger $trigger -Force