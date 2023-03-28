# Description: Download videos from YouTube using yt-dlp
$videoUrls = Get-Content .\urls_politseikroonika.txt
ForEach ($url in $videoUrls) { yt-dlp --paths .\video $url }