@echo off
schtasks /delete /tn Faceable /f
schtasks /delete /tn "Faceable Runner" /f
schtasks /delete /tn "Faceable Killer" /f