# Description: Remove entries from a CSV file whose names are not contained in a directory
# Load CSV file into PowerShell
$csv = Import-Csv .\metadata.csv -Delimiter "|" -Header Filename, Transcription
# Filter out entries whose names are not contained in a directory
# Get all the .wav files in the trimmed directory
$wav_files = Get-ChildItem .\wavs\trimmed | Where-Object { $_.Extension -eq ".wav" } | ForEach-Object { $_.Name }
# Filter the metadata to include only the wav files
$filtered_metadata = $csv | Where-Object { $_.Filename -in $wav_files }
# Export the filtered metadata
$filtered_metadata | Export-Csv .\metadata_filtered.csv -NoTypeInformation -UseQuotes Never -Delimiter "|"
