# Create a new directory to store the trimmed WAV files
New-Item -ItemType Directory -Path .\trimmed

# Loop through each WAV file in the current directory
Get-ChildItem | Where-Object { $_.Extension -eq ".wav" } | ForEach-Object {
    # Trim silence from the start and end of the WAV file
    ffmpeg -i $_.Name -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:detection=peak,aformat=dblp,areverse" -y ".\trimmed\$($_.Name)"
}