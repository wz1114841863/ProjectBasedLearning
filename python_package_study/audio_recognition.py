import speech_recognition as sr


def recognize_speech():
    """ 识别麦克风输入内容 """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("请说出您的问题...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='zh-CN')
            print("您说的是: " + text)
            return text
        except sr.UnknownValueError:
            print("抱歉,我无法理解您说的话.")
            return None


def recognize_audio_file(audio_file_path, offset=None, duration=None, language="en-US"):
    """ 识别音频文件内容

    audio_file_path:
        音频文件路径, 需要满足speech_recognition可以处理的格式要求

    offset:
        起始偏移

    duration:
        持续时间, 调用多次时, 会持续后移.

    lagguage:
        识别的语言
    """
    recognizer = sr.Recognizer()
    test_audio = sr.AudioFile(audio_file_path)
    with test_audio as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source, offset=offset, duration=duration)
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"识别结果: {text}")
        except sr.UnknownValueError:
            print("抱歉, 无法进行识别.")
            return None


def get_chatgpt_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].text.strip()


if __name__ == "__main__":
    OPENAI_API_KEY = "INSERT OPENAI API KEY HERE"
    while True:
        text = recognize_speech()
        answer = get_chatgpt_response(text)
        print("ChatGPT的回答: " + answer)
    # audio_path = "./data/jackhammer.wav"
    # recognize_audio_file(audio_path)
