#!/usr/bin/env python3
import speech_recognition as sr
from os import path
import pyaudio
import wave
import cv2




frame = cv2.imread("background.png")

while True:
    background = frame.copy()
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "output_russian_command.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "output_russian_command.wav")
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        recognized_command = r.recognize_google(audio, language="ru-RU")
        print("Google Speech Recognition thinks you said " + recognized_command)

        color = (255, 255, 255)
        center = (background.shape[1] // 2, background.shape[0] // 2)
        if ("Жёлтый" in recognized_command) or ("жёлтый" in recognized_command):
            color = (0, 255, 255)
        if ("Красный" in recognized_command) or ("красный" in recognized_command):
            color = (0, 0, 255)
        if ("Синий" in recognized_command) or ("синий" in recognized_command):
            color = (255, 0, 0)
        if ("Зелёный" in recognized_command) or ("зелёный" in recognized_command):
            color = (0, 255, 0)

        if ("влево" in recognized_command) or ("Влево" in recognized_command):
            center = (background.shape[1] // 2 - 200, background.shape[0] // 2)

        if ("вправо" in recognized_command) or ("Вправо" in recognized_command):
            center = (background.shape[1] // 2 + 200, background.shape[0] // 2)

        if ("верх" in recognized_command) or ("Верх" in recognized_command):
            center = (background.shape[1] // 2, background.shape[0] // 2 - 200)

        if ("вниз" in recognized_command) or ("Вниз" in recognized_command):
            center = (background.shape[1] // 2, background.shape[0] // 2 + 200)


        cv2.circle(background, center, 8, color, 3)


    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    cv2.imshow("wWin", background)
    cv2.waitKey()

