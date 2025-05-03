import os
import torch
import numpy as np
import soundfile as sf
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
    MarianMTModel,
    MarianTokenizer
)
import fasttext
import librosa
import warnings
warnings.filterwarnings("ignore")

class VoiceEmotionPipeline:
    """
    End-to-end pipeline for voice recording, language detection, 
    transcription, and emotion analysis.
    """
    
    def __init__(self, use_indic=True, load_translation=True):
        """
        Initialize models for the pipeline
        
        Args:
            use_indic (bool): Whether to use IndicWav2Vec for Indian languages
            load_translation (bool): Whether to load translation models
        """
        print("Loading models... this may take a minute.")
        
        # 1. Load Whisper model for speech recognition
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        
        # 2. For Indian languages (optional)
        self.use_indic = use_indic
        if use_indic:
            self.indic_processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
            self.indic_model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")
        
        # 3. Language detection model
        # Download model if not present
        if not os.path.exists("lid.176.bin"):
            os.system("wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        self.lang_model = fasttext.load_model("lid.176.bin")
        
        # 4. Emotion analysis models
        # General English emotion model
        self.emotion_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
        
        # 5. Translation models for non-English languages
        self.translation_models = {}
        self.load_translation = load_translation
        
        if load_translation:
            # Load common language translation models
            # We'll load these lazily to save memory
            self.translation_language_map = {
                # Indian languages
                "hi": "Helsinki-NLP/opus-mt-hi-en",  # Hindi
                "bn": "Helsinki-NLP/opus-mt-bn-en",  # Bengali
                "ta": "Helsinki-NLP/opus-mt-ta-en",  # Tamil
                "te": "Helsinki-NLP/opus-mt-mul-en", # Telugu (via multilingual)
                "mr": "Helsinki-NLP/opus-mt-mr-en",  # Marathi
                
                # Other major languages
                "es": "Helsinki-NLP/opus-mt-es-en",  # Spanish
                "fr": "Helsinki-NLP/opus-mt-fr-en",  # French
                "de": "Helsinki-NLP/opus-mt-de-en",  # German
                "ru": "Helsinki-NLP/opus-mt-ru-en",  # Russian
                "zh": "Helsinki-NLP/opus-mt-zh-en",  # Chinese
                "ar": "Helsinki-NLP/opus-mt-ar-en",  # Arabic
                "ja": "Helsinki-NLP/opus-mt-ja-en",  # Japanese
                
                # Fallback for other languages
                "mul": "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English
            }
            print("Translation models will be loaded on demand")
        
        # 6. Acoustic emotion features (complementary approach)
        self.acoustic_emotion = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        
        print("All models loaded successfully!")

    def record_audio(self, duration=5, sample_rate=16000, filename="recorded_audio.wav"):
        """
        Record audio from microphone
        
        Args:
            duration (int): Duration to record in seconds 
            sample_rate (int): Audio sample rate
            filename (str): Where to save the audio file
            
        Returns:
            str: Path to saved audio file
        """
        try:
            import sounddevice as sd
            
            print(f"Recording for {duration} seconds...")
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            sf.write(filename, audio, sample_rate)
            print(f"Audio saved to {filename}")
            return filename
        except Exception as e:
            print(f"Error recording audio: {e}")
            print("Please ensure you have installed sounddevice: pip install sounddevice")
            return None

    def process_audio_file(self, file_path):
        """
        Process an existing audio file through the pipeline
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            dict: Results containing language, transcript, and emotions
        """
        # Load and preprocess audio
        audio, sample_rate = librosa.load(file_path, sr=16000)
        
        # Run full pipeline
        results = {
            "file_path": file_path,
            "audio_duration": len(audio) / sample_rate
        }
        
        # 1. Transcribe with Whisper
        transcript, language = self._transcribe_audio(audio)
        results["transcript"] = transcript
        results["detected_language_code"] = language
        
        # 2. Additional language detection from text
        lang_results = self._detect_language_from_text(transcript)
        results["text_language"] = lang_results
        
        # 3. Use IndicWav2Vec if appropriate
        if self.use_indic and lang_results["language"] in ["hi", "bn", "ta", "te", "mr", "ur", "gu", "kn", "ml", "pa"]:
            indic_transcript = self._process_with_indicwav2vec(audio)
            results["indic_transcript"] = indic_transcript
            # Use this transcript for further processing if it's better
            if len(indic_transcript) > len(transcript) * 0.8:  # Simple heuristic
                transcript = indic_transcript
                
        # 4. Translate non-English text to English for better emotion analysis
        if lang_results["language"] != "en" and self.load_translation:
            translated_text = self._translate_to_english(transcript, lang_results["language"])
            results["translated_text"] = translated_text
            # Use translated text for emotion analysis
            text_for_emotion = translated_text
        else:
            text_for_emotion = transcript
        
        # 5. Emotion analysis from text (using English text when available)
        text_emotions = self._analyze_text_emotion(text_for_emotion, "en" if lang_results["language"] != "en" else lang_results["language"])
        results["text_emotion"] = text_emotions
        
        # 6. Acoustic emotion analysis
        acoustic_emotions = self._analyze_acoustic_emotion(audio, sample_rate)
        results["acoustic_emotion"] = acoustic_emotions
        
        # 7. Combined results
        results["final_emotion"] = self._combine_emotion_results(text_emotions, acoustic_emotions)
        
        return results

    def _transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        input_features = self.whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        
        # Generate tokens with Whisper
        predicted_ids = self.whisper_model.generate(input_features)
        
        # Decode the tokens to text
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Get language from generation config
        detected_language = self.whisper_model.config.forced_decoder_ids[0][1] if hasattr(self.whisper_model.config, 'forced_decoder_ids') else "unknown"
        
        return transcription, detected_language

    def _process_with_indicwav2vec(self, audio):
        """Process Indian language audio with IndicWav2Vec"""
        if not self.use_indic:
            return ""
            
        inputs = self.indic_processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.indic_model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.indic_processor.batch_decode(predicted_ids)[0]
        
        return transcription

    def _detect_language_from_text(self, text):
        """Detect language from text using FastText"""
        if not text.strip():
            return {"language": "unknown", "confidence": 0.0}
            
        prediction = self.lang_model.predict(text.replace("\n", " "))
        language = prediction[0][0].replace("__label__", "")
        confidence = prediction[1][0]
        
        language_names = {
            "en": "English",
            "hi": "Hindi",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "ur": "Urdu",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "pa": "Punjabi",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "zh": "Chinese",
            "ar": "Arabic"
        }
        
        return {
            "language": language,
            "language_name": language_names.get(language, "Unknown"),
            "confidence": float(confidence)
        }

    def _translate_to_english(self, text, source_language):
        """
        Translate text to English using appropriate translation model
        
        Args:
            text (str): Text to translate
            source_language (str): Source language code
            
        Returns:
            str: Translated text in English
        """
        if not self.load_translation or not text.strip():
            return text
            
        try:
            # Get appropriate model for the language
            model_name = self.translation_language_map.get(
                source_language, 
                self.translation_language_map.get("mul")  # Fallback to multilingual
            )
            
            # Load model if not already loaded
            if model_name not in self.translation_models:
                print(f"Loading translation model for {source_language}...")
                self.translation_models[model_name] = {
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "model": MarianMTModel.from_pretrained(model_name)
                }
            
            # Get the model components
            tokenizer = self.translation_models[model_name]["tokenizer"]
            model = self.translation_models[model_name]["model"]
            
            # Translate
            batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                generated_ids = model.generate(**batch)
            
            translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return translated_text
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def _analyze_text_emotion(self, text, language_code):
        """
        Analyze emotion from text
        Different models used based on language
        """
        if not text.strip():
            return {"emotion": "unknown", "scores": {}}
            
        # Note: We now expect English text since translation happens before this
        # But we keep the language check as a fallback
        if language_code == "en":
            # Use RoBERTa for English
            inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
            
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
            emotions = ["anger", "joy", "optimism", "sadness"]
            
            emotion_scores = {emotion: score for emotion, score in zip(emotions, scores)}
            top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            return {
                "emotion": top_emotion[0],
                "confidence": top_emotion[1],
                "scores": emotion_scores
            }
        else:
            # Should rarely happen now due to translation step
            return {"emotion": "unknown", "scores": {}, "note": f"No emotion model available for {language_code}"}

    def _analyze_acoustic_emotion(self, audio, sample_rate):
        """Analyze emotion from acoustic features"""
        try:
            # Ensure audio matches expected format
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Save temporary file for pipeline processing
            temp_file = "temp_audio_chunk.wav"
            sf.write(temp_file, audio, 16000)
            
            # Run acoustic emotion recognition
            result = self.acoustic_emotion(temp_file)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Process results
            top_emotion = result[0]
            emotion_dict = {item["label"]: item["score"] for item in result}
            
            return {
                "emotion": top_emotion["label"],
                "confidence": top_emotion["score"],
                "scores": emotion_dict
            }
        except Exception as e:
            print(f"Error in acoustic emotion analysis: {e}")
            return {"emotion": "unknown", "scores": {}}

    def _combine_emotion_results(self, text_emotion, acoustic_emotion):
        """
        Combine text and acoustic emotion results
        Using a weighted approach
        """
        if text_emotion["emotion"] == "unknown" and acoustic_emotion["emotion"] == "unknown":
            return {"emotion": "unknown", "confidence": 0.0}
            
        if text_emotion["emotion"] == "unknown":
            return {"emotion": acoustic_emotion["emotion"], "confidence": acoustic_emotion["confidence"]}
            
        if acoustic_emotion["emotion"] == "unknown":
            return {"emotion": text_emotion["emotion"], "confidence": text_emotion["confidence"]}
        
        # Map emotions to common space
        # This is a simplified mapping - a real implementation would be more sophisticated
        emotion_mapping = {
            # Text emotions (RoBERTa)
            "anger": "angry",
            "joy": "happy",
            "optimism": "happy",
            "sadness": "sad",
            
            # Acoustic emotions (Wav2Vec2)
            "angry": "angry",
            "happy": "happy",
            "sad": "sad",
            "neutral": "neutral",
            "fearful": "fearful",
            "disgust": "disgust"
        }
        
        text_mapped = emotion_mapping.get(text_emotion["emotion"], text_emotion["emotion"])
        acoustic_mapped = emotion_mapping.get(acoustic_emotion["emotion"], acoustic_emotion["emotion"])
        
        # Simple weighted combination (text 60%, acoustic 40%)
        # If emotions match, higher confidence
        if text_mapped == acoustic_mapped:
            return {
                "emotion": text_mapped,
                "confidence": 0.6 * text_emotion["confidence"] + 0.4 * acoustic_emotion["confidence"],
                "text_contribution": text_emotion["emotion"],
                "acoustic_contribution": acoustic_emotion["emotion"]
            }
        
        # If emotions don't match, return the one with higher confidence
        if text_emotion["confidence"] > acoustic_emotion["confidence"]:
            return {
                "emotion": text_mapped,
                "confidence": text_emotion["confidence"],
                "source": "text",
                "text_contribution": text_emotion["emotion"],
                "acoustic_contribution": acoustic_emotion["emotion"]
            }
        else:
            return {
                "emotion": acoustic_mapped,
                "confidence": acoustic_emotion["confidence"],
                "source": "acoustic",
                "text_contribution": text_emotion["emotion"],
                "acoustic_contribution": acoustic_emotion["emotion"]
            }

    def process_live(self, duration=5):
        """
        Record and process live audio
        
        Args:
            duration (int): Duration to record in seconds
            
        Returns:
            dict: Analysis results
        """
        audio_file = self.record_audio(duration=duration)
        if audio_file:
            return self.process_audio_file(audio_file)
        else:
            return {"error": "Failed to record audio"}

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = VoiceEmotionPipeline(use_indic=True)
    
    # Option 1: Process existing file
    results = pipeline.process_audio_file("sample_audio.wav")
    
    # Option 2: Record and process live
    # results = pipeline.process_live(duration=5)
    
    # Print results
    print("\n=== Voice Analysis Results ===")
    print(f"Transcript: {results['transcript']}")
    print(f"Detected Language: {results.get('text_language', {}).get('language_name', 'Unknown')}")
    
    # Print translation if available
    if 'translated_text' in results:
        print(f"English Translation: {results['translated_text']}")
    
    print(f"Emotion (Text): {results.get('text_emotion', {}).get('emotion', 'Unknown')}")
    print(f"Emotion (Acoustic): {results.get('acoustic_emotion', {}).get('emotion', 'Unknown')}")
    print(f"Final Emotion: {results.get('final_emotion', {}).get('emotion', 'Unknown')}")
    print(f"Confidence: {results.get('final_emotion', {}).get('confidence', 0):.2f}")