import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
                           QStyle, QStyleFactory, QHBoxLayout, QFrame, QGraphicsOpacityEffect)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation, QEasingCurve, QTime
from PyQt5.QtGui import QPalette, QColor, QIcon
from qt_material import apply_stylesheet, list_themes
from auth import AuthManager, LoginDialog
import faster_whisper
import torch

# Диагностика CUDA
print("CUDA доступность:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Количество GPU:", torch.cuda.device_count())
    print("Текущее устройство:", torch.cuda.current_device())
    print("Имя устройства:", torch.cuda.get_device_name(0))

# Приоритет CUDA над DirectML
if torch.cuda.is_available():
    print("Используется CUDA для ускорения")
    dml = None
else:
    # Пробуем DirectML только если нет CUDA
    try:
        import torch_directml
        dml = torch_directml.device()
        # Проверяем поддержку DirectML
        test_tensor = torch.randn(1, 1).to(dml)
        del test_tensor  # Очищаем тестовый тензор
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
    finished = pyqtSignal()
    acceleration_changed = pyqtSignal(str)  # Новый сигнал
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.working_dir = "temp"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Инициализируем Demucs
        self.separator = get_model('htdemucs')
        self.separator.eval()
        
        # Определяем устройство для вычислений
        if torch.cuda.is_available():
            self.device = "cuda"
            self.acceleration_changed.emit("CUDA")
            print("Используется CUDA для вычислений")
            # Явно переводим модель на GPU
            self.separator.cuda()
        else:
            if 'dml' in globals() and dml is not None:
                self.device = dml
                self.acceleration_changed.emit("DirectML")
                print("Используется DirectML для вычислений")
            else:
                self.device = "cpu"
                self.acceleration_changed.emit("CPU")
                print("Используется CPU для вычислений")
            
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
        if torch.cuda.is_available() and str(self.device) == 'cuda':
            device = "cuda"
            compute_type = "float16"
            self.acceleration_changed.emit("CUDA")
            print("Используется CUDA для Whisper")
        else:
            device = "cpu"
            compute_type = "int8"
            if 'dml' in globals() and dml is not None:
                self.acceleration_changed.emit("DirectML")
            else:
                self.acceleration_changed.emit("CPU")
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
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"Ошибка в процессе обработки: {str(e)}")
            self.status.emit(f"Ошибка: {str(e)}")

class StyledFrame(QFrame):
    """Стилизованная карточка для группировки элементов"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("styledFrame")
        self.update_style(True)  # По умолчанию тёмная тема
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        
    def update_style(self, is_dark):
        if is_dark:
            self.setStyleSheet("""
                QFrame#styledFrame {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#styledFrame {
                    background-color: rgba(33, 150, 243, 0.05);
                    border-radius: 10px;
                    border: 1px solid rgba(33, 150, 243, 0.2);
                }
            """)

class AnimatedProgressBar(QProgressBar):
    """Прогресс-бар с анимацией"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_style(True)  # По умолчанию тёмная тема
        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.setDuration(500)

    def update_style(self, is_dark):
        if is_dark:
            self.setStyleSheet("""
                QProgressBar {
                    border: none;
                    border-radius: 10px;
                    text-align: center;
                    background-color: rgba(255, 255, 255, 0.1);
                    height: 20px;
                    color: white;
                }
                QProgressBar::chunk {
                    border-radius: 10px;
                    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                        stop:0 #2196F3, stop:1 #00BCD4);
                }
            """)
        else:
            self.setStyleSheet("""
                QProgressBar {
                    border: none;
                    border-radius: 10px;
                    text-align: center;
                    background-color: rgba(33, 150, 243, 0.1);
                    height: 20px;
                    color: #2196F3;
                }
                QProgressBar::chunk {
                    border-radius: 10px;
                    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                        stop:0 #1976D2, stop:1 #2196F3);
                }
            """)
            
    def setValue(self, value):
        self._animation.stop()
        self._animation.setStartValue(self.value())
        self._animation.setEndValue(value)
        self._animation.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSynch")
        self.setGeometry(100, 100, 500, 600)
        
        # Определяем цветовые схемы
        self.color_schemes = {
            'dark': {
                'primary': '#2196F3',      # Синий
                'secondary': '#00BCD4',    # Голубой
                'background': '#121212',   # Тёмный фон
                'surface': '#1E1E1E',     # Поверхность
                'text': '#FFFFFF',        # Белый текст
                'text_secondary': 'rgba(255, 255, 255, 0.7)',  # Полупрозрачный белый
                'button': '#2196F3',      # Синие кнопки
                'button_hover': '#1976D2', # Тёмно-синий при наведении
                'error': '#CF6679'        # Красный для ошибок
            },
            'light': {
                'primary': '#2196F3',      # Синий
                'secondary': '#0D47A1',    # Тёмно-синий
                'background': '#FFFFFF',   # Белый фон
                'surface': '#F5F5F5',     # Светло-серый
                'text': '#000000',        # Чёрный текст
                'text_secondary': 'rgba(0, 0, 0, 0.6)',  # Полупрозрачный чёрный
                'button': '#2196F3',      # Синие кнопки
                'button_hover': '#1976D2', # Тёмно-синий при наведении
                'error': '#B00020'        # Красный для ошибок
            }
        }
        
        # Определяем тип ускорения
        if torch.cuda.is_available():
            self.acceleration_type = "CUDA"
        elif 'dml' in globals() and dml is not None:
            self.acceleration_type = "DirectML"
        else:
            self.acceleration_type = "CPU"
        
        # Инициализируем менеджер авторизации
        self.auth_manager = AuthManager()
        
        # Инициализация темы
        self.is_dark_theme = True
        
        # Создаем центральный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Верхняя панель
        top_frame = StyledFrame()
        top_layout = QVBoxLayout()  # Изменено на вертикальный layout
        
        # Верхняя строка с информацией
        info_layout = QHBoxLayout()
        
        # Информация о пользователе
        self.user_label = QLabel("Пользователь: не авторизован")
        self.user_label.setStyleSheet("font-size: 14px;")
        info_layout.addWidget(self.user_label)
        
        # Добавляем растягивающийся элемент
        info_layout.addStretch()
        
        # Информация об ускорении
        self.acceleration_label = QLabel(f"Ускорение: {self.acceleration_type}")
        self.acceleration_label.setStyleSheet("""
            font-size: 14px;
            color: #2196F3;
            font-weight: bold;
        """)
        info_layout.addWidget(self.acceleration_label)
        
        top_layout.addLayout(info_layout)
        
        # Кнопки управления
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Кнопка авторизации
        self.auth_button = QPushButton("Войти")
        self.auth_button.setIcon(QIcon.fromTheme("system-users"))
        self.auth_button.clicked.connect(self.toggle_auth)
        buttons_layout.addWidget(self.auth_button)
        
        # Кнопка темы
        self.theme_button = QPushButton()
        self.theme_button.setToolTip("Сменить тему оформления")
        self.theme_button.setFixedSize(120, 35)
        self.theme_button.clicked.connect(self.toggle_theme)
        self.update_theme_button()
        buttons_layout.addWidget(self.theme_button)
        
        # Выравнивание кнопок по правому краю
        buttons_layout.addStretch()
        
        top_layout.addLayout(buttons_layout)
        top_frame.layout.addLayout(top_layout)
        layout.addWidget(top_frame)
        
        # Основная панель
        main_frame = StyledFrame()
        
        # Кнопка выбора файла
        self.select_button = QPushButton("Выбрать видео")
        self.select_button.setIcon(QIcon.fromTheme("video-x-generic"))
        self.select_button.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 14px;
            }
        """)
        self.select_button.clicked.connect(self.select_file)
        main_frame.layout.addWidget(self.select_button)
        
        # Метка файла
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        main_frame.layout.addWidget(self.file_label)
        
        layout.addWidget(main_frame)
        
        # Панель прогресса
        progress_frame = StyledFrame()
        
        # Прогресс бар
        self.progress_bar = AnimatedProgressBar()
        progress_frame.layout.addWidget(self.progress_bar)
        
        # Время и отмена
        status_layout = QHBoxLayout()
        
        # Метка времени
        self.time_label = QLabel("Оставшееся время: --:--")
        self.time_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        status_layout.addWidget(self.time_label)
        
        # Кнопка отмены
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.setIcon(QIcon.fromTheme("process-stop"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        status_layout.addWidget(self.cancel_button)
        
        progress_frame.layout.addLayout(status_layout)
        
        # Метка статуса
        self.status_label = QLabel("Готов к работе")
        self.status_label.setStyleSheet("""
            padding: 5px;
            color: rgba(255, 255, 255, 0.7);
            font-style: italic;
        """)
        progress_frame.layout.addWidget(self.status_label)
        
        layout.addWidget(progress_frame)
        
        # Инициализация
        self.video_path = None
        self.processing = False
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_estimate)
        
        # Проверяем авторизацию
        self.check_auth()
        
        # Применяем тему Material Design
        self.apply_theme()
        
    def apply_theme(self):
        """Применение выбранной темы"""
        theme = 'dark_blue.xml' if self.is_dark_theme else 'light_blue.xml'
        apply_stylesheet(self, theme=theme, invert_secondary=True)
        
        # Получаем текущую цветовую схему
        colors = self.color_schemes['dark' if self.is_dark_theme else 'light']
        
        # Обновляем стили для кастомных элементов
        for frame in self.findChildren(StyledFrame):
            frame.update_style(self.is_dark_theme)
            
        # Обновляем прогресс-бар
        self.progress_bar.update_style(self.is_dark_theme)
        
        # Обновляем стили кнопок
        button_style = f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {'white' if self.is_dark_theme else 'white'};
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {colors['button_hover']};
            }}
            QPushButton:disabled {{
                background-color: {'rgba(255, 255, 255, 0.12)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.12)'};
                color: {'rgba(255, 255, 255, 0.3)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.3)'};
            }}
        """
        
        self.select_button.setStyleSheet(button_style)
        self.auth_button.setStyleSheet(button_style)
        self.cancel_button.setStyleSheet(button_style)
        
        # Специальный стиль для кнопки темы
        theme_button_style = f"""
            QPushButton {{
                background-color: {'rgba(255, 255, 255, 0.1)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.1)'};
                color: {colors['text']};
                border: 1px solid {'rgba(255, 255, 255, 0.2)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.2)'};
                border-radius: 5px;
                padding: 5px 10px;
                font-weight: bold;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {'rgba(255, 255, 255, 0.15)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.15)'};
            }}
        """
        self.theme_button.setStyleSheet(theme_button_style)
        
        # Обновляем стили текстовых меток
        text_style = f"""
            color: {colors['text_secondary']};
            font-size: 14px;
        """
        
        self.file_label.setStyleSheet(text_style)
        self.time_label.setStyleSheet(text_style)
        self.status_label.setStyleSheet(f"""
            {text_style}
            font-style: italic;
            padding: 5px;
        """)
        
        # Особый стиль для метки пользователя
        self.user_label.setStyleSheet(f"""
            color: {colors['text']};
            font-size: 14px;
            font-weight: bold;
        """)
        
        # Стиль для метки ускорения
        self.acceleration_label.setStyleSheet(f"""
            color: {colors['primary']};
            font-size: 14px;
            font-weight: bold;
        """)
        
    def show_with_animation(self):
        """Анимированное появление окна"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)
        self.fade_in.setDuration(500)
        self.fade_in.setEasingCurve(QEasingCurve.OutCubic)
        
        self.show()
        self.fade_in.start()
        
    def toggle_theme(self):
        """Переключение между темной и светлой темой"""
        self.is_dark_theme = not self.is_dark_theme
        self.update_theme_button()
        self.apply_theme()
        
    def update_theme_button(self):
        """Обновление внешнего вида кнопки темы"""
        if self.is_dark_theme:
            self.theme_button.setText(" Светлая тема")
            self.theme_button.setIcon(QIcon.fromTheme("weather-clear"))
        else:
            self.theme_button.setText(" Тёмная тема")
            self.theme_button.setIcon(QIcon.fromTheme("weather-clear-night"))
        
    def update_time_estimate(self):
        """Обновление оценки оставшегося времени"""
        if self.processing and hasattr(self, 'processor'):
            progress = self.progress_bar.value()
            if progress > 0:
                elapsed = (QTime.currentTime().msecsSinceStartOfDay() - self.start_time) / 1000
                total_estimated = (elapsed * 100) / progress
                remaining = total_estimated - elapsed
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                self.time_label.setText(f"Оставшееся время: {minutes:02d}:{seconds:02d}")
                
    def cancel_processing(self):
        """Отмена обработки видео"""
        if hasattr(self, 'processor') and self.processing:
            self.processor.terminate()
            self.processing = False
            self.timer.stop()
            self.progress_bar.setValue(0)
            self.status_label.setText("Обработка отменена")
            self.cancel_button.setEnabled(False)
            self.time_label.setText("Оставшееся время: --:--")
            self.select_button.setEnabled(True)
            
    def start_processing(self):
        self.processor = VideoProcessor(self.video_path)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.status.connect(self.status_label.setText)
        self.processor.finished.connect(self.processing_finished)
        self.processor.acceleration_changed.connect(self.update_acceleration_label)
        self.processing = True
        self.start_time = QTime.currentTime().msecsSinceStartOfDay()
        self.timer.start(1000)
        self.cancel_button.setEnabled(True)
        self.select_button.setEnabled(False)
        self.processor.start()
        
    def processing_finished(self):
        """Обработчик завершения обработки видео"""
        self.processing = False
        self.timer.stop()
        self.cancel_button.setEnabled(False)
        self.time_label.setText("Оставшееся время: --:--")
        self.select_button.setEnabled(True)
        
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

    def update_acceleration_label(self, acceleration_type):
        """Обновление метки с информацией об ускорении"""
        self.acceleration_type = acceleration_type
        self.acceleration_label.setText(f"Ускорение: {acceleration_type}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show_with_animation()
    sys.exit(app.exec_()) 