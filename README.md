# Edge AI Medical Voice Assistant 🩺🎙️

## 📌 About the Project

The Edge AI Medical Voice Assistant is an offline, CPU-based voice assistant designed to respond to health-related questions using locally running AI models.
The project integrates speech recognition, a local large language model (LLM), and text-to-speech technology to provide a voice-based interaction experience. It explores the potential of Edge AI for privacy-conscious and accessible healthcare information systems.

## ✨ Features

- 🎙️ Voice input through a browser-based interface
- 📝 Speech-to-text conversion using Vosk
- 🧠 Locally generated responses using Ollama and Qwen 2.5
- 🔊 Text-to-speech conversion using Piper
- 🔒 Designed for local processing without cloud-based AI inference
- 💻 Built to run on a CPU without requiring a dedicated GPU

## 🛠️ Technologies Used

## 🛠️ Technologies Used

- Python: Core application logic
- Flask:Web server and backend
- Vosk: Speech recognition
- Ollama:Local language model runtime
- Qwen 2.5:AI response generation
- Piper:Text-to-speech
- FFmpeg:Audio format conversion
- HTML, CSS, JavaScript:Web interface

## ⚙️ How It Works

1. The user speaks through the browser microphone.
2. The recorded audio is sent to the local Flask server.
3. FFmpeg converts the audio into a suitable format.
4. Vosk converts speech into text.
5. Ollama runs the Qwen language model to generate a response.
6. Piper converts the generated response into speech.
7. The response is displayed on the webpage and played as audio.

## 🚀 Project Setup

### Prerequisites

- Windows operating system
- Python
- Git
- Ollama
- FFmpeg
- Vosk speech recognition model
- Piper text-to-speech model

### Run the Application

1. Clone this repository.
2. Install the required Python dependencies.
3. Download and configure the required local speech recognition and text-to-speech models.
4. Install Ollama and download the configured Qwen model.
5. Ensure FFmpeg is installed and accessible to the application.
6. Start the Flask server using the project's configured Python environment.
7. Open `https://localhost:5000` in your browser.

**Note:** Additional setup may be required because model files and local dependencies are not included in this repository.

## ⚠️ Current Limitations

- Speech recognition may struggle with medical terminology and unclear speech.
- Small language models can generate incomplete or inaccurate answers.
- Response quality and processing speed depend on the local hardware and models.
- The project has not been validated for clinical use.

## 🔮 Future Improvements

- Improve recognition of medical terminology.
- Integrate a curated medical knowledge base using retrieval-augmented generation (RAG).
- Add audio playback controls.
- Evaluate speech recognition accuracy and response latency.
- Improve medical response safety and reliability.
- Optimize local model performance.

## 🎯 Objective

To explore how locally running AI models can combine speech recognition, language processing, and speech synthesis to create an accessible, privacy-conscious voice assistant.

## ⚕️ Disclaimer

This is an educational and experimental project. It is not intended to provide medical diagnoses, replace healthcare professionals, or guide emergency medical decisions.
