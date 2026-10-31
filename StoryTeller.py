from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
def llm():
    model = ChatGroq(api_key="API_KEY",
                     model="openai/gpt-oss-20b")
    return model



def story_chain():
    model = llm()
    prompt_template = PromptTemplate(input_variables=['topic'],
    template = f"""Write an engaging and educational story about {topic} for beginners. 
                Use simple and clear language to explain basic concepts. 
                Include interesting facts and keep it friendly and encouraging. 
                The story should be around 200-300 words and end with a brief summary of what we learned. 
                Make it perfect for someone just starting to learn about this topic.""")
    response = prompt_template | model | StrOutputParser()
    return response

def generate_story(topic):
    response = story_chain()
    return response.invoke({"topic":topic})





from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import io

def text_to_speech(story):
    audio_buffer = io.BytesIO()

    # Generate MP3 directly in memory
    gTTS(text=story, lang="en").write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    # Decode MP3 into audio samples
    audio, sample_rate = sf.read(audio_buffer, dtype="float32")

    # Play directly through your speakers
    sd.play(audio, sample_rate)
    sd.wait()


topic = "League Of Legends"
story = generate_story(topic)
print("Generated Story:\n", story)
text_to_speech(story)