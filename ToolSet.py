import os
from dotenv import load_dotenv
import pandas as pd
import whisper
from pydantic import BaseModel, Field
from langchain_experimental.utilities import PythonREPL
import cv2
from pathlib import Path
from yt_dlp import YoutubeDL
from ultralytics import YOLO, settings
from typing import List, Dict
from typing import TypedDict, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain.tools import Tool, tool

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY is None:
    raise RuntimeError("API not set in environment")


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Adds two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a + b


@tool
def subtract(a: float, b: float) -> int:
    """Subtracts two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a - b

@tool
def divide(a: float, b: float) -> float:
    """Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    Args:
        a (int): the first number
        b (int): the second number
    """
    return a % b


@tool
def power(a: float, b: float) -> float:
    """Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    return a**b

@tool
def excel_reader(path: str):
    """
    Reads the specified Excel file into a pandas DataFrame,
    converts it to CSV-style text, and asks the LLM to answer the question
    based on that data.
    Args:
        path: path indicating the excel file
    """
    print("reading_excel_file")
    df = pd.read_excel(path)
    data_context = df.to_csv(df)

    return data_context


@tool
def get_web_search_result(query: str) -> str:
    """Fetches information from the internet (web) based on given query.
    
    Args:
        query: The search query.
        
    Returns:
        The search results.
    """
    print("get_web_search_result")
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)      
    return{"web_search_results": search_docs}


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 5 results. Use this tool only if the query specifies Wiki or Wikipedia.
    Args:
        query: The search query.
    Returns:
        An array documents.
    """
    print("wiki_search")
    search_docs = WikipediaLoader(query=query, load_max_docs=5).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query.
    Returns:
        An array of documents.
    """
    print("arxiv_search")
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arxiv_results": formatted_search_docs}

@tool
def reverse_text(prompt: str) -> str:
    """
    Returns the reversed version of a given reversed text so that the text makes sense.
    Args:
        prompt: The prompt which contains word and sentence in a reverse order.
    Returns:
        A reversed version of  the reversed sentence which is human readable and understandable.
    """

    print("restoring_text")
    return prompt[::-1]


@tool
def transcribe_audio(file_path: str):
    """
    Transcribes an audio file to text using local Whisper model.
    Then uses the transcription to answer question from the given prompt.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        A dictionary containing the transcription and metadata
    """
    try:
        print(f"Transcribing audio file: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Load a Whisper model - we'll use the small model for better performance
        # Options include: tiny, base, small, medium, large
        model = whisper.load_model("small")
        
        # Transcribe the audio
        result = model.transcribe(file_path)
        print({
            "status": "success",
            "transcription": result["text"],
            "language": result.get("language", "unknown"),
            "file_path": file_path
        })
        
        # Return the transcription and metadata
        return {
            "status": "success",
            "transcription": result["text"],
            "language": result.get("language", "unknown"),
            "file_path": file_path
        }
        
    except Exception as e:
        print({
            "status": "error",
            "message": f"Error transcribing audio: {str(e)}"
        })
        return {
            "status": "error",
            "message": f"Error transcribing audio: {str(e)}"
        }


class PythonREPLInput(BaseModel):
    code: str = Field(description="The Python code string to execute.")

python_repl = PythonREPL()

python_repl_tool = Tool(
    name="python_repl",
    description="""A Python REPL shell (Read-Eval-Print Loop).
Use this to execute single or multi-line python commands.
Input should be syntactically valid Python code.
Always end your code with `print(...)` to see the output.
Do NOT execute code that could be harmful to the host system.
You are allowed to download files from URLs.
Do not use this tool as a web search.
Do NOT send commands that block indefinitely (e.g., `input()`).""",
    func=python_repl.run,
    args_schema=PythonREPLInput
)


class YouTubeFrameExtractor:
    def __init__(self, model_path: str = 'yolov8n.pt', frame_rate: int = 1):
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.frame_rate = frame_rate  # frames per second to sample

    def download_video(self, url: str) -> str:
        ydl_opts = {
            "cookiefile": "cookies.txt",
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': '%(id)s.%(ext)s',
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    def extract_counts_per_frame(self, url: str) -> List[Dict[str, int]]:
        video_path = self.download_video(url)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = max(1, int(round(fps / self.frame_rate)))

        frame_counts: List[Dict[str, int]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                counts: Dict[str, int] = {}
                results = self.model(frame)
                for det in results:
                    for *box, conf, cls in det.boxes.data.tolist():
                        name = self.model.names[int(cls)]
                        counts[name] = counts.get(name, 0) + 1
                frame_counts.append(counts)
            frame_idx += 1

        cap.release()
        os.remove(video_path)
        return frame_counts

def max_object_counter_tool() -> Tool:
    extractor = YouTubeFrameExtractor()

    def _max_object(input_str: str) -> str:
        # Expect input: '<video_url> <object_name>'
        parts = input_str.strip().split()
        if len(parts) < 2:
            return "Usage: <YouTube_URL> <object_name>"
        url, obj_name = parts[0], parts[1]
        frames = extractor.extract_counts_per_frame(url)
        if not frames:
            return "No frames processed or unable to download video."
        # Compute max occurrences across frames
        max_count = max(frame.get(obj_name, 0) for frame in frames)
        return f"Maximum count of '{obj_name}' in any sampled frame: {max_count}"

    return Tool(
        name="youtube_max_object_counter",
        func=_max_object,
        description=(
            "Downloads a YouTube video, samples frames at a given rate, runs YOLO detection, "
            "and returns the maximum count of the specified object across all sampled frames."
        )
    )


class YouTubeTranscriber:
    def __init__(self, model_size: str = "small"):
        # Load Whisper model (tiny/base/small/medium/large/turbo)
        self.model = whisper.load_model(model_size)

    def download_audio(self, url: str) -> str:
        """
        Download only the audio from a YouTube URL and return the local filename.
        """
        ydl_opts = {
            "format": "bestaudio/best",               # best available audio :contentReference[oaicite:3]{index=3}
            "postprocessors": [{
                "key": "FFmpegExtractAudio",           # extract with FFmpeg :contentReference[oaicite:4]{index=4}
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": "%(id)s.%(ext)s",              # name file as "<video_id>.mp3"
            "quiet": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return f"{info['id']}.mp3"

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Run Whisper on the given audio file and return the transcript.
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            without_timestamps=True
        )
        # os.remove(audio_path)
        return result["text"]


def transcription_generation_tool() -> Tool:
    """
    Returns a LangChain Tool that takes a YouTube URL and optional language code,
    then returns the transcription text.
    """
    transcriber = YouTubeTranscriber(model_size="small")

    def _transcribe_tool(input_str: str) -> str:
        # Expect: "<YouTube_URL> [language_code] "Question Text""
        parts = input_str.strip().split()
        url = parts[0]
        lang = parts[1] if len(parts) > 2 and not input_str.split('"')[1] else "en"
        # Extract question between quotes
        question = input_str.split('"')[1]
        try:
            audio_file = transcriber.download_audio(url)
            transcript = transcriber.transcribe(audio_file, language=lang)
            os.remove(audio_file)
            return transcript
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        name="youtube_transcriber",
        func=_transcribe_tool,
        description=(
            "Downloads audio from YouTube, transcribes it, and answers a question based on the transcript. "
            "Usage: <YouTube_URL> [language_code] \"Question text\""
        )
    )
        
toolset = [
    get_web_search_result,
    wiki_search,
    arxiv_search,
    reverse_text,
    transcribe_audio,
    python_repl_tool,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    excel_reader,
    max_object_counter_tool(),
    transcription_generation_tool()
]