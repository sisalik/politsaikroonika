# Description: Measure the maximum audio volume and normalise the audio if it is too quiet. Also downsample the audio to 16-bit, 16kHz, mono.
$volume_target = -3
Get-ChildItem .\wav | ForEach-Object {
    # Get the filename without the extension
    $filename = $_.BaseName
    # Get the full path to the audio file
    $audio_path = $_.FullName
    # Measure the maximum audio volume
    $max_volume = [float]((ffmpeg -i $audio_path -af "volumedetect" -f null - 2>&1) | Select-String -Pattern "max_volume" | ForEach-Object { $_.Line } | ForEach-Object { $_.Split(":")[1].Trim().Split(" ")[0] })
    # Write-Host "${filename}: $max_volume dB"
    # Normalise the audio if it is too quiet
    $audio_path_normalised = ".\wav_normalised\${filename}.wav"
    if ($max_volume -lt $volume_target) {
        $volume_adjustment = $volume_target - $max_volume
        Write-Host "Increasing volume of ${filename} by ${volume_adjustment} dB..."
        ffmpeg -i $audio_path -af "volume=${volume_adjustment}dB" -acodec pcm_s16le -ac 1 -ar 16000 $audio_path_normalised -hide_banner -loglevel error
    } else {
        Write-Host "Downsampling ${filename}..."
        ffmpeg -i $audio_path -acodec pcm_s16le -ac 1 -ar 16000 $audio_path_normalised -hide_banner -loglevel error
    }
}