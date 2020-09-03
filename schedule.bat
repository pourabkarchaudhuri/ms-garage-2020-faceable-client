@echo off
schtasks /Create /XML Faceable.xml /tn Faceable
schtasks /Create /XML "Faceable Killer.xml" /tn "Faceable Killer"
schtasks /Create /XML "Faceable Runner.xml" /tn "Faceable Runner"
