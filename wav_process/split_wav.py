from pydub import AudioSegment
test = AudioSegment.from_wav('test.wav')
# ten_seconds = 1 * 1000
# total_num = int(test.duration_seconds)*5
total_num = int(len(test)/200)
for i in range(total_num):
    if i == 0:
        current_wav = AudioSegment.silent(duration=800) + test[0:(i+1)*200]
    elif i == 1:
        current_wav = AudioSegment.silent(duration=600) + test[0:(i+1)*200]
    elif i == 2:
        current_wav = AudioSegment.silent(duration=400) + test[0:(i+1)*200]
    elif i == 3:
        current_wav = AudioSegment.silent(duration=200) + test[0:(i+1)*200]
    elif i == 4:
        current_wav = AudioSegment.silent(duration=0) + test[0:(i+1)*200]
    else:
        current_wav = test[(i-4)*200:(i+1)*200]
    current_wav.export('wav/'+str(i)+'_.wav',format='wav')

