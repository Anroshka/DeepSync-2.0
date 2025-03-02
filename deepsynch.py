import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from auth import AuthManager, LoginDialog
import faster_whisper
import torch

# Настройка DirectML для PyTorch
try:
    import torch_directml
    dml = torch_directml.device()
    # Проверяем поддержку DirectML
    test_tensor = torch.randn(1, 1).to(dml)
    del test_tensor  # Очищаем тестовый тензор
    
    # Если тест прошел успешно, настраиваем DirectML
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = lambda: dml
    torch.cuda.device = lambda device: dml
    torch.cuda.device_count = lambda: 1
    torch.cuda.set_device = lambda device: None
    torch.cuda.get_device_name = lambda device: "DirectML Device"
    print("DirectML успешно инициализирован для ускорения вычислений")
except (ImportError, RuntimeError) as e:
    print(f"DirectML не удалось инициализировать: {str(e)}")
    print("Используется CPU")
    dml = None

from deep_translator import GoogleTranslator
from TTS.api import TTS
import ffmpeg
import logging
import librosa
import soundfile as sf
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.working_dir = "temp"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Инициализируем Demucs
        self.separator = get_model('htdemucs')
        self.separator.eval()
        
        # Определяем устройство для вычислений
        try:
            if 'dml' in globals() and dml is not None:
                self.device = dml
                print("Используется DirectML для ускорения")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("Используется CUDA")
            else:
                self.device = "cpu"
                print("Используется CPU")
        except Exception as e:
            print(f"Ошибка при инициализации устройства: {str(e)}")
            self.device = "cpu"
            print("Используется CPU из-за ошибки")
            
        self.separator.to(self.device)
        
    def extract_audio(self):
        """Извлечение аудио из видео"""
        self.status.emit("Извлечение аудио...")
        output_audio = os.path.join(self.working_dir, "audio.wav")
        try:
            stream = ffmpeg.input(self.video_path)
            stream = ffmpeg.output(stream, output_audio, acodec='pcm_s16le', ac=2, ar='44100')
            ffmpeg.run(stream, overwrite_output=True)
            return output_audio
        except ffmpeg.Error as e:
            logger.error(f"Ошибка при извлечении аудио: {str(e)}")
            raise

    def separate_audio(self, audio_path):
        """Разделение аудио на голос и фон"""
        self.status.emit("Разделение аудио на голос и фон...")
        
        try:
            # Загружаем аудио
            audio_file = AudioFile(audio_path).read(streams=0, samplerate=self.separator.samplerate, channels=self.separator.audio_channels)
            ref = audio_file.mean(0)
            audio_file = (audio_file - ref.mean()) / ref.std()
            
            # Применяем модель
            audio_file = audio_file.unsqueeze(0)
            
            # Для операций с комплексными числами временно используем CPU
            self.separator.cpu()
            audio_file = audio_file.cpu()
            
            with torch.no_grad():
                sources = apply_model(self.separator, audio_file, split=True, device="cpu")
                sources = sources * ref.std() + ref.mean()
            
            # Возвращаем модель на DirectML/GPU
            self.separator.to(self.device)
            
            # Получаем отдельные дорожки
            sources = sources.numpy()
            vocals = sources[0, self.separator.sources.index('vocals')]
            no_vocals = np.zeros_like(vocals)
            
            # Суммируем все, кроме вокала
            for source in ['drums', 'bass', 'other']:
                idx = self.separator.sources.index(source)
                no_vocals += sources[0, idx]
            
            # Сохраняем результаты
            vocals_path = os.path.join(self.working_dir, "vocals.wav")
            background_path = os.path.join(self.working_dir, "background.wav")
            
            sf.write(vocals_path, vocals.T, self.separator.samplerate)
            sf.write(background_path, no_vocals.T, self.separator.samplerate)
            
            return vocals_path, background_path
            
        except Exception as e:
            logger.error(f"Ошибка при разделении аудио: {str(e)}")
            # В случае ошибки возвращаем модель на исходное устройство
            self.separator.to(self.device)
            raise

    def get_reference_audio(self, vocals_path):
        """Получение референсного аудио из выделенного голоса"""
        self.status.emit("Подготовка референсного голоса...")
        # Загружаем аудио
        y, sr = librosa.load(vocals_path, sr=24000)
        
        # Берем 10 секунд из середины или всё аудио если оно короче
        duration = len(y) / sr
        if duration > 10:
            start = int((duration/2 - 5) * sr)
            end = int((duration/2 + 5) * sr)
            y = y[start:end]
        
        # Сохраняем референсный файл
        reference_path = os.path.join(self.working_dir, "reference.wav")
        sf.write(reference_path, y, sr)
        return reference_path
            
    def transcribe_audio(self, vocals_path):
        """Распознавание речи с помощью Faster Whisper из выделенного голоса"""
        self.status.emit("Распознавание речи...")
        model_size = "base"
        
        # Настраиваем вычислительное устройство для Faster Whisper
        # Faster Whisper поддерживает только 'cuda' или 'cpu' как строки
        if torch.cuda.is_available() and str(self.device) == 'cuda':
            device = "cuda"
            compute_type = "float16"
            print("Используется CUDA для Whisper")
        else:
            device = "cpu"
            compute_type = "int8"
            print("Используется CPU для Whisper")
            
        model = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(vocals_path, beam_size=5)
        
        # Собираем текст из всех сегментов
        result_text = ""
        for segment in segments:
            result_text += segment.text + " "
            
        return result_text.strip()
        
    def translate_text(self, text):
        """Перевод текста на русский"""
        self.status.emit("Перевод текста...")
        translator = GoogleTranslator(source='auto', target='ru')
        return translator.translate(text)
        
    def generate_speech(self, text, reference_audio):
        """Генерация речи на русском"""
        self.status.emit("Генерация речи...")
        output_speech = os.path.join(self.working_dir, "generated_speech.wav")
        
        # TTS пока не полностью совместим с DirectML, используем CUDA или CPU
        if torch.cuda.is_available() and str(self.device) == 'cuda':
            device = "cuda"
            print("Используется CUDA для TTS")
        else:
            device = "cpu"
            print("Используется CPU для TTS")
            
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        
        tts.tts_to_file(text=text, 
                        file_path=output_speech,
                        language="ru",
                        speaker_wav=reference_audio)
        
        return output_speech

    def mix_audio(self, speech_path, background_path):
        """Смешивание сгенерированной речи с фоновой музыкой"""
        self.status.emit("Микширование аудио...")
        
        # Загружаем аудио файлы
        speech, sr = librosa.load(speech_path, sr=44100)
        background, _ = librosa.load(background_path, sr=44100)
        
        # Обеспечиваем одинаковую длину
        if len(speech) > len(background):
            background = np.pad(background, (0, len(speech) - len(background)))
        else:
            speech = np.pad(speech, (0, len(background) - len(speech)))
        
        # Микшируем с разными весами
        mixed = speech * 0.7 + background * 0.3
        
        # Сохраняем результат
        output_path = os.path.join(self.working_dir, "final_audio.wav")
        sf.write(output_path, mixed, sr)
        return output_path
        
    def merge_audio_video(self, audio_path):
        """Объединение видео с новым аудио"""
        self.status.emit("Сборка финального видео...")
        output_video = "output_video.mp4"
        try:
            input_video = ffmpeg.input(self.video_path)
            input_audio = ffmpeg.input(audio_path)
            
            # Добавляем параметры для корректной работы с AAC
            stream = ffmpeg.output(input_video.video, 
                                 input_audio.audio,
                                 output_video,
                                 acodec='aac',
                                 vcodec='copy',
                                 strict='experimental')
            
            ffmpeg.run(stream, overwrite_output=True)
            return output_video
        except ffmpeg.Error as e:
            logger.error(f"Ошибка при сборке видео: {str(e)}")
            raise
            
    def run(self):
        try:
            # Извлечение аудио
            self.progress.emit(10)
            audio_path = self.extract_audio()
            
            # Разделение на голос и фон
            self.progress.emit(20)
            vocals_path, background_path = self.separate_audio(audio_path)
            
            # Получение референсного голоса из выделенного голоса
            self.progress.emit(30)
            reference_audio = self.get_reference_audio(vocals_path)
            
            # Распознавание речи из выделенного голоса
            self.progress.emit(40)
            text = self.transcribe_audio(vocals_path)
            
            # Перевод
            self.progress.emit(50)
            translated_text = self.translate_text(text)
            
            # Генерация речи
            self.progress.emit(60)
            new_speech = self.generate_speech(translated_text, reference_audio)
            
            # Микширование с фоном
            self.progress.emit(70)
            final_audio = self.mix_audio(new_speech, background_path)
            
            # Финальная сборка
            self.progress.emit(90)
            output_video = self.merge_audio_video(final_audio)
            
            self.progress.emit(100)
            self.status.emit("Готово!")
            
        except Exception as e:
            logger.error(f"Ошибка в процессе обработки: {str(e)}")
            self.status.emit(f"Ошибка: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSynch")
        self.setGeometry(100, 100, 400, 200)
        
        # Инициализируем менеджер авторизации
        self.auth_manager = AuthManager()
        
        # Создаем центральный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Метка для отображения информации о пользователе
        self.user_label = QLabel("Пользователь: не авторизован")
        layout.addWidget(self.user_label)
        
        # Кнопка авторизации/выхода
        self.auth_button = QPushButton("Войти")
        self.auth_button.clicked.connect(self.toggle_auth)
        layout.addWidget(self.auth_button)
        
        # Кнопка выбора файла
        self.select_button = QPushButton("Выбрать видео")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)
        
        # Метка для отображения выбранного файла
        self.file_label = QLabel("Файл не выбран")
        layout.addWidget(self.file_label)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Метка статуса
        self.status_label = QLabel("Готов к работе")
        layout.addWidget(self.status_label)
        
        self.video_path = None
        
        # Проверяем, требуется ли авторизация для работы
        self.check_auth()
        
    def check_auth(self):
        """Проверка авторизации и обновление интерфейса"""
        user_info = self.auth_manager.get_current_user_info()
        if user_info:
            self.user_label.setText(f"Пользователь: {user_info['name']}")
            self.auth_button.setText("Выйти")
            self.select_button.setEnabled(True)
        else:
            self.user_label.setText("Пользователь: не авторизован")
            self.auth_button.setText("Войти")
            self.select_button.setEnabled(False)
            
    def toggle_auth(self):
        """Переключение между авторизацией и выходом"""
        if self.auth_manager.current_user:
            # Выход из системы
            self.auth_manager.logout()
            self.check_auth()
        else:
            # Авторизация
            login_dialog = LoginDialog(self.auth_manager, self)
            if login_dialog.exec_():
                self.check_auth()
        
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            "",
            "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_path:
            self.video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.start_processing()
            
    def start_processing(self):
        self.processor = VideoProcessor(self.video_path)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 