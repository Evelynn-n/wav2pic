from pydub import AudioSegment
from ffmpy3 import FFmpeg
song = AudioSegment.from_mp3('8K HDR Colorful Dolby Vision.mp3')
print(song.channels)
print('')