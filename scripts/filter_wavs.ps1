# Description: Filter out WAV files whose names are contained in a CSV file, and trim silence from the start and end of the WAV files
# Make a new directory to store trimmed WAV files
New-Item -ItemType Directory -Path .\wavs\trimmed
# Filter out .wav files whose names are contained in a CSV file
# Load CSV file into PowerShell
$wavs_csv = Import-Csv .\metadata.csv -Delimiter "|" -Header Filename, Transcription
Get-ChildItem .\wavs | Where-Object { $_.Extension -eq ".wav" } | ForEach-Object {
    # Check if the WAV file name is contained in the CSV file
    if ($wavs_csv.Filename -contains $_.Name) {
        # Trim silence from the start and end of the WAV file
        $new_filename = ".\wavs\trimmed\$($_.Name)"
        ffmpeg -i $_.Name -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse" -y $new_filename
        # Get the duration of the trimmed WAV file
        $duration = [float](ffprobe -i $new_filename -show_entries format=duration -v quiet -of csv="p=0")
        # Delete the trimmed file if its duration is not between 1 and 10 seconds
        if ($duration -lt 1 -or $duration -gt 10) {
            Remove-Item $new_filename
        }
    }
}
