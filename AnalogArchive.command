#!/usr/bin/env bash
# Open the site in your default browser, then run the server
cd "/Users/alvarotrabanco/documents_mis_proyectos_small/Mis_Proyectos/Fotografia/AnalogProcess/Archive/xmp_archive_harvester_v2" || exit 1
open "http://localhost:8000"
exec ./serve.sh
