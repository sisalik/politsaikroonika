# Description: Extract audio from video files
Get-ChildItem .\video_cut | ForEach-Object {
    # Get the filename without the extension
    $filename = $_.BaseName
    # Get the full path to the video file
    $video_path = $_.FullName
    # Extract the audio from the video (WAV, 16-bit, 44.1kHz, stereo)
    $audio_path = ".\wav_full\${filename}.wav"
    if (!(Test-Path $audio_path)) { ffmpeg -i $video_path -acodec pcm_s16le -ac 2 -ar 44100 $audio_path }
    # Extract MP3 audio from the video
    $audio_path = ".\mp3_full\${filename}.mp3"
    if (!(Test-Path $audio_path)) { ffmpeg -i $video_path -vn -acodec libmp3lame -ac 2 -ab 192k -ar 44100 $audio_path }
}