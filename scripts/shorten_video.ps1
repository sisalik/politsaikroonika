# Description: Adjust video playback speed to ensure duration is less than 60 seconds
# Get the duration of the video
$duration = (ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $args[0])
# If the duration is less than 60 seconds, exit
if ($duration -lt 60) {
    exit
}
# Calculate the speedup factor
$speedup = 59.9 / $duration
$speedup_inverse = $duration / 59.9
# Confirm speed up factor and overwriting of original file with user
Write-Host "Speeding up video by $speedup_inverse times and overwriting original file"
$confirmation = Read-Host "Are you sure you want to continue? (y/n)"
if ($confirmation -ne "y") {
    exit
}

# Speed up the video and audio tracks; write to a temporary file
ffmpeg -i $args[0] -filter_complex "[0:v]setpts=$speedup*PTS[v];[0:a]atempo=$speedup_inverse[a]" -map "[v]" -map "[a]" -y temp.mp4
# Verify the duration of the temporary file is less than 60 seconds
$duration = (ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 temp.mp4)
if ($duration -gt 60) {
    Write-Host "Error: duration of temporary file is $duration seconds"
    exit
}
# Overwrite the original file with the temporary file
Move-Item -Force temp.mp4 $args[0]
