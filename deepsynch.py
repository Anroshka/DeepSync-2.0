import sys
import math # <--- Добавлен импорт
import os
import shutil # <--- Добавлен для очистки папки
import subprocess # <--- Добавлен для запуска Wav2Lip
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
                           QStyle, QStyleFactory, QHBoxLayout, QFrame, QGraphicsOpacityEffect,
                           QComboBox, QCheckBox, QSlider)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation, QEasingCurve, QTime
from PyQt5.QtGui import QPalette, QColor, QIcon
from qt_material import apply_stylesheet, list_themes
# Импортируем только WhisperX
try:
    import whisperx
    import torch
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
import torch

# Диагностика CUDA
print("CUDA доступность:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Количество GPU:", torch.cuda.device_count())
    print("Текущее устройство:", torch.cuda.current_device())
    print("Имя устройства:", torch.cuda.get_device_name(0))

# Используем только CUDA или CPU
if torch.cuda.is_available():
    print("Используется CUDA для ускорения")
else:
    print("Используется CPU")

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
from scipy import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Настройки Wav2Lip ---
WAV2LIP_PYTHON_PATH = "python"  # Путь к Python с установленным Wav2Lip
WAV2LIP_INFERENCE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wav2Lip", "inference.py")
WAV2LIP_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wav2Lip", "checkpoints", "wav2lip_gan.pth")

# Проверка доступности Wav2Lip
WAV2LIP_AVAILABLE = os.path.exists(WAV2LIP_INFERENCE_SCRIPT) and os.path.exists(WAV2LIP_CHECKPOINT)
if not WAV2LIP_AVAILABLE:
    logger.warning(f"Wav2Lip недоступен. Проверьте наличие файлов: {WAV2LIP_INFERENCE_SCRIPT}, {WAV2LIP_CHECKPOINT}")
else:
    logger.info(f"Wav2Lip доступен и готов к использованию")

# --- Класс VideoProcessor ---
class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str) # Изменен, чтобы передавать путь к итоговому файлу
    error = pyqtSignal(str)   # Сигнал для передачи ошибок в GUI
    acceleration_changed = pyqtSignal(str)

    def __init__(self, video_path, target_language='ru', use_wav2lip=False, resize_factor=4):
        super().__init__()
        self.video_path = video_path
        self.target_language = target_language
        self.use_wav2lip = use_wav2lip and WAV2LIP_AVAILABLE  # Используем Wav2Lip только если он доступен
        self.resize_factor = resize_factor  # Фактор масштабирования для Wav2Lip
        # Параметры громкости для микширования
        self.speech_gain = 0.8  # Громкость речи (0.0 - 1.0)
        self.background_gain = 0.2  # Громкость фона (0.0 - 1.0)
        # Создаем уникальное имя папки на случай параллельной обработки
        base_working_dir = "temp_processing"
        # Можно добавить timestamp или uuid для уникальности, если нужно
        self.working_dir = os.path.join(base_working_dir, os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(self.working_dir, exist_ok=True)
        logger.info(f"Временная папка: {self.working_dir}")
        
        # Директория для кеширования результатов
        self.cache_dir = os.path.join("temp_processing", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Уникальный ключ кеша для текущей обработки
        self.cache_key = f"{os.path.splitext(os.path.basename(video_path))[0]}_{target_language}"
        if use_wav2lip:
            self.cache_key += "_wav2lip"
        
        if self.use_wav2lip:
            logger.info("Активирована синхронизация губ с Wav2Lip")
        else:
            if use_wav2lip and not WAV2LIP_AVAILABLE:
                logger.warning("Синхронизация губ с Wav2Lip запрошена, но недоступна. Функция отключена.")
        
        self.supported_languages = {
            'ar': 'Arabic', 'cs': 'Czech', 'de': 'German', 'en': 'English',
            'es': 'Spanish', 'fr': 'French', 'hi': 'Hindi', 'hu': 'Hungarian',
            'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'nl': 'Dutch',
            'pl': 'Polish', 'pt': 'Portuguese', 'ru': 'Russian', 'tr': 'Turkish',
            'zh-cn': 'Chinese'
        }

        self._is_cancelled = False # Флаг для отмены

        # Инициализация моделей и устройства (Demucs)
        try:
             # Используем htdemucs_ft для лучшего качества разделения
            self.separator = get_model('htdemucs_ft') # Обновлено
            self.separator.eval()
        except Exception as e:
            logger.error(f"Не удалось загрузить модель Demucs: {e}")
            # Возможно, стоит сразу сигнализировать об ошибке
            # self.error.emit(f"Ошибка загрузки Demucs: {e}")
            raise # Прерываем инициализацию

        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type_whisper = "float16"
            self.tts_device = "cuda"
            logger.info("Используется CUDA для Demucs, Whisper, TTS")
            self.separator.cuda()
            self.acceleration_changed.emit("CUDA")
        else:
            self.device = "cpu"
            self.compute_type_whisper = "int8" # int8 для CPU Whisper
            self.tts_device = "cpu"
            logger.info("Используется CPU для Demucs, Whisper, TTS")
            self.acceleration_changed.emit("CPU")
            self.separator.to(self.device) # Перемещаем Demucs на CPU

        # Инициализация TTS вынесена в run для экономии памяти, если не используется
        self.tts_model = None
        # Инициализация Whisper вынесена в transcribe_audio
        self.whisper_model = None

    def request_cancellation(self):
        """Установка флага отмены потока"""
        logger.info("Получен запрос на отмену обработки.")
        self._is_cancelled = True

    def check_cancel(self):
        """Проверка флага отмены и выброс исключения если нужно"""
        if self._is_cancelled:
            raise InterruptedError("Обработка отменена пользователем.")

    def get_audio_duration(self, audio_path):
        """Получает длительность аудиофайла в секундах"""
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.warning(f"Soundfile не смог получить длительность {audio_path}: {e}. Пробуем librosa.")
            try:
                # Librosa может быть медленнее, т.к. декодирует больше данных
                return librosa.get_duration(filename=audio_path)
            except Exception as e2:
                logger.error(f"Librosa тоже не смог получить длительность {audio_path}: {e2}")
                return 0

    def extract_audio(self):
        self.check_cancel()
        self.status.emit("Извлечение аудио...")
        output_audio = os.path.join(self.working_dir, "original_audio.wav")
        logger.info(f"Извлечение аудио из {self.video_path} в {output_audio}")
        try:
            (
                ffmpeg
                .input(self.video_path)
                .output(output_audio, acodec='pcm_s16le', ac=1, ar='44100') # Моно для вокала лучше
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info("Аудио успешно извлечено.")
            return output_audio
        except ffmpeg.Error as e:
            logger.error(f"Ошибка ffmpeg при извлечении аудио: {e.stderr.decode()}")
            raise

    def separate_audio(self, audio_path):
        self.check_cancel()
        self.status.emit("Разделение аудио (Demucs)...")
        vocals_path = os.path.join(self.working_dir, "vocals.wav")
        background_path = os.path.join(self.working_dir, "background.wav")
        logger.info("Начало разделения аудио с помощью Demucs...")

        try:
            # Загружаем аудиофайл с помощью утилиты Demucs
            wav = AudioFile(audio_path).read(streams=0, samplerate=self.separator.samplerate, channels=self.separator.audio_channels)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            # Перемещаем тензор на нужное устройство (CUDA/CPU)
            wav = wav.to(self.device)

            # Применяем модель Demucs
            with torch.no_grad():
                # Передаем устройство явно
                sources = apply_model(self.separator, wav.unsqueeze(0), device=self.device, split=True, overlap=0.25)[0]
            sources = sources * ref.std().to(self.device) + ref.mean().to(self.device)

            # Сохраняем вокал и фон
            # Индексы могут зависеть от модели, проверяем
            if 'vocals' in self.separator.sources:
                 vocal_idx = self.separator.sources.index('vocals')
                 # Перемещаем тензор на CPU перед сохранением и конвертацией в numpy
                 save_audio(sources[vocal_idx].cpu(), vocals_path, samplerate=self.separator.samplerate)
                 logger.info(f"Вокал сохранен: {vocals_path}")
            else:
                 logger.error("Источник 'vocals' не найден в модели Demucs!")
                 raise RuntimeError("Источник 'vocals' не найден в модели Demucs!")

            # Собираем фон из остальных источников
            other_sources = []
            for i, source_name in enumerate(self.separator.sources):
                 if source_name != 'vocals':
                     other_sources.append(sources[i])

            if other_sources:
                 background_tensor = torch.stack(other_sources).sum(dim=0)
                 save_audio(background_tensor.cpu(), background_path, samplerate=self.separator.samplerate)
                 logger.info(f"Фон сохранен: {background_path}")
            else:
                 # Если только вокал, создаем тишину в качестве фона
                 logger.warning("Не найдено других источников кроме вокала. Фон будет тишиной.")
                 # Длительность фона берем как у вокала
                 duration_samples = sources[vocal_idx].shape[-1]
                 background_tensor = torch.zeros((self.separator.audio_channels, duration_samples))
                 save_audio(background_tensor.cpu(), background_path, samplerate=self.separator.samplerate)


            return vocals_path, background_path

        except Exception as e:
            logger.exception("Ошибка при разделении аудио:") # Используем exception для stack trace
            raise
        finally:
             # Очистка CUDA кэша, если используется GPU
             if self.device == "cuda":
                 torch.cuda.empty_cache()


    def get_reference_audio(self, vocals_path):
        self.check_cancel()
        self.status.emit("Подготовка референсного голоса...")
        reference_path = os.path.join(self.working_dir, "reference_voice.wav")
        target_sr = 24000 # XTTS требует 24kHz
        logger.info(f"Создание референса ({reference_path}) из {vocals_path}")
        try:
            # Загружаем вокальную дорожку, убедившись, что это именно результат демукса
            if not os.path.exists(vocals_path) or not os.path.basename(vocals_path) == "vocals.wav":
                logger.warning(f"Неверный путь к вокальному файлу: {vocals_path}. Убедитесь, что используется файл vocals.wav.")
                raise ValueError("Необходим файл vocals.wav для создания качественного референса")
                
            y, sr = librosa.load(vocals_path, sr=target_sr) # Сразу загружаем с нужной SR

            # Проверка на тишину и уровень шума
            max_amplitude = np.max(np.abs(y))
            if max_amplitude < 0.01: # Порог тишины (можно настроить)
                logger.warning("Обнаружена тишина или очень тихий звук в вокальной дорожке. TTS может работать некорректно.")
                raise ValueError("Вокальная дорожка слишком тихая для создания качественного референса")
            
            # Дополнительный анализ для выбора наиболее чистого фрагмента голоса
            # 1. Вычисляем огибающую сигнала для анализа речевой активности
            envelope = np.abs(y)
            frame_length = int(0.025 * target_sr)  # 25 мс фреймы
            hop_length = int(0.010 * target_sr)    # 10 мс перекрытие
            
            # 2. Находим энергию в каждом фрейме
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 3. Нормализуем энергию
            energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy
            
            # 4. Определяем пороговое значение для выделения речи (адаптивно)
            threshold = 0.15  # Начальный порог
            if np.median(energy_norm) > 0.05:
                # Если медианная энергия высокая, повышаем порог для избежания шума
                threshold = np.median(energy_norm) * 1.5
            
            # 5. Определяем "речевые" фреймы
            speech_frames = np.where(energy_norm > threshold)[0]
            
            # 6. Если нашли речевые фреймы, выбираем из них самый длинный непрерывный фрагмент
            if len(speech_frames) > 0:
                # Ищем самый длинный непрерывный фрагмент речи
                gaps = np.diff(speech_frames)
                gap_idx = np.where(gaps > 1)[0]
                
                if len(gap_idx) > 0:
                    segments = np.split(speech_frames, gap_idx + 1)
                    # Берем самый длинный сегмент
                    best_segment = max(segments, key=len)
                else:
                    best_segment = speech_frames
                
                # Преобразуем индексы фреймов в сэмплы
                start_sample = max(0, best_segment[0] * hop_length - frame_length)
                end_sample = min(len(y), (best_segment[-1] + 1) * hop_length + frame_length)
                
                # Если фрагмент слишком короткий (менее 5 сек), расширяем его
                min_duration_samples = 5 * target_sr
                if (end_sample - start_sample) < min_duration_samples:
                    additional_samples = min_duration_samples - (end_sample - start_sample)
                    # Пытаемся добавить равномерно с обеих сторон
                    start_sample = max(0, start_sample - additional_samples // 2)
                    end_sample = min(len(y), end_sample + additional_samples // 2)
                
                # Увеличиваем длительность до 15 секунд если возможно
                desired_duration_samples = 15 * target_sr
                if (end_sample - start_sample) < desired_duration_samples and len(y) > desired_duration_samples:
                    # Находим окно с максимальной суммарной энергией
                    speech_energy = energy_norm.copy()
                    window_size = desired_duration_samples // hop_length
                    
                    if len(speech_energy) > window_size:
                        # Находим окно с максимальной суммарной энергией
                        energy_sum = np.convolve(speech_energy, np.ones(window_size), mode='valid')
                        best_start_frame = np.argmax(energy_sum)
                        
                        # Преобразуем индексы в сэмплы
                        start_sample = best_start_frame * hop_length
                        end_sample = start_sample + desired_duration_samples
                    else:
                        # Файл слишком короткий, берем все
                        start_sample = 0
                        end_sample = len(y)
                
                # Вырезаем лучший сегмент
                y_ref = y[start_sample:end_sample]
                logger.info(f"Найден качественный сегмент голоса: {start_sample/target_sr:.2f}s - {end_sample/target_sr:.2f}s (длительность {(end_sample-start_sample)/target_sr:.2f}s)")
            else:
                # Если речь не обнаружена, используем стандартную стратегию
                logger.warning("Не обнаружены четкие речевые фрагменты, используем стандартную стратегию выбора")
                # Выбор фрагмента для референса (15 секунд если возможно)
                min_dur = 3
                max_dur = 15  # Увеличено до 15 секунд
                if len(y) / sr > max_dur + min_dur:
                    start_sample = min_dur * sr
                    end_sample = start_sample + max_dur * sr
                    y_ref = y[start_sample:end_sample]
                elif len(y) / sr > min_dur: # Если короче max_dur+min_dur, но длиннее min_dur
                     start_sample = int(sr * 1) # Берем с 1 секунды
                     y_ref = y[start_sample:]
                else: # Если совсем коротко, берем все
                     y_ref = y
            
            # Дополнительная обработка: нормализация громкости референса
            if np.max(np.abs(y_ref)) > 0:
                y_ref = y_ref / np.max(np.abs(y_ref)) * 0.95  # Нормализуем до 95% максимальной амплитуды
            
            # Сохраняем референсный аудио
            sf.write(reference_path, y_ref, target_sr, subtype='PCM_16')
            logger.info(f"Референсный голос сохранен: {reference_path} (длительность {len(y_ref)/target_sr:.2f} сек)")
            return reference_path
        except Exception as e:
            logger.error(f"Ошибка при создании референсного голоса: {str(e)}")
            raise

    def transcribe_audio(self, vocals_path):
        self.check_cancel()
        self.status.emit("Распознавание речи (WhisperX)...")
        
        # Проверяем, доступен ли WhisperX
        if WHISPERX_AVAILABLE:
            logger.info(f"Запуск WhisperX (модель turbo) для {vocals_path}")
            return self._transcribe_with_whisperx(vocals_path)
        else:
            logger.error("WhisperX не найден. Установите whisperx для продолжения.")
            raise ImportError("Для работы программы требуется установленный WhisperX")
            
    def _transcribe_with_whisperx(self, vocals_path):
        """
        Транскрибирует аудио с помощью WhisperX
        """
        try:
            self.check_cancel()
            logger.info(f"Запуск WhisperX (модель turbo) для {vocals_path}")

            compute_device = "cuda" if self.device == "cuda" else "cpu"
            
            # 1. Загружаем модель и выполняем начальное распознавание
            model = whisperx.load_model("turbo", compute_device, compute_type="float16" if self.device == "cuda" else "int8")
            audio = whisperx.load_audio(vocals_path)
            result = model.transcribe(audio, batch_size=16)
            
            # 2. Выравниваем речь на уровне слов
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=compute_device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, compute_device, return_char_alignments=False)

            segments = []
            for segment in result["segments"]:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            
            logger.info(f"WhisperX успешно транскрибировал {len(segments)} сегментов")
            return segments
            
        except Exception as e:
            logger.exception(f"Ошибка при транскрибировании с WhisperX: {e}")
            raise

    def translate_text(self, segments):
        self.check_cancel()
        self.status.emit(f"Перевод на {self.supported_languages.get(self.target_language, self.target_language)}...")
        logger.info(f"Начало перевода {len(segments)} сегментов на язык '{self.target_language}'")

        # Создаем один экземпляр переводчика
        try:
            translator = GoogleTranslator(source='auto', target=self.target_language)
        except Exception as e:
             logger.error(f"Не удалось инициализировать GoogleTranslator: {e}")
             raise

        translated_segments = []
        total_segments = len(segments)
        for i, segment in enumerate(segments):
            self.check_cancel()
            original_text = segment['text']
            start_time = segment['start']
            end_time = segment['end']

            if not original_text:
                 logger.info(f"Пропуск перевода для пустого сегмента {i}")
                 translated_segments.append({
                     "start": start_time, "end": end_time, "text": "", "original_text": ""
                 })
                 continue

            try:
                translated_text = translator.translate(original_text)
                if not translated_text:
                    logger.warning(f"Переводчик вернул пустой результат для сегмента {i}: '{original_text}'")
                    translated_text = "" # Используем пустую строку вместо None

                translated_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": translated_text.strip(), # Убираем лишние пробелы и в переводе
                    "original_text": original_text
                })
                logger.info(f"Перевод {i+1}/{total_segments}: [{start_time:.2f}s] '{original_text}' -> '{translated_text}'")
                self.progress.emit(50 + int(10 * (i + 1) / total_segments)) # Прогресс 50-60%

            except Exception as e:
                 logger.error(f"Ошибка перевода сегмента {i} ('{original_text}'): {str(e)}")
                 # Добавляем сегмент с пустым текстом в случае ошибки
                 translated_segments.append({
                     "start": start_time, "end": end_time, "text": "", "original_text": original_text, "error": str(e)
                 })

        return translated_segments

    def generate_speech(self, translated_segments, reference_audio):
        self.check_cancel()
        self.status.emit("Генерация речи (TTS)...")
        tts_segment_dir = os.path.join(self.working_dir, "tts_segments")
        os.makedirs(tts_segment_dir, exist_ok=True)
        logger.info(f"Начало генерации TTS для {len(translated_segments)} сегментов в папку {tts_segment_dir}")

        # Проверка референсного аудио перед запуском TTS
        if not os.path.exists(reference_audio) or os.path.getsize(reference_audio) == 0:
            logger.error(f"Референсный аудиофайл не найден или пуст: {reference_audio}")
            raise FileNotFoundError(f"Референсный аудиофайл не найден или пуст: {reference_audio}")
            
        # Проверяем качество референсного аудио
        try:
            ref_audio, ref_sr = librosa.load(reference_audio, sr=None)
            ref_max_amp = np.max(np.abs(ref_audio))
            
            if ref_max_amp < 0.01:
                logger.warning(f"Референсный аудиофайл имеет очень низкую громкость: {ref_max_amp}. Это может привести к шуму в TTS.")
                # Попробуем нормализовать громкость
                ref_audio = ref_audio / ref_max_amp * 0.95 if ref_max_amp > 0 else ref_audio
                
                # Сохраним нормализованный файл
                normalized_ref_path = os.path.join(self.working_dir, "reference_voice_normalized.wav")
                sf.write(normalized_ref_path, ref_audio, ref_sr, subtype='PCM_16')
                logger.info(f"Создан нормализованный референсный файл: {normalized_ref_path}")
                reference_audio = normalized_ref_path
        except Exception as e:
            logger.warning(f"Не удалось проверить/нормализовать референсный аудиофайл: {e}")
            # Продолжаем с исходным файлом

        # Ленивая инициализация TTS
        if self.tts_model is None:
             try:
                  logger.info(f"Загрузка модели TTS 'tts_models/multilingual/multi-dataset/xtts_v2' на устройство '{self.tts_device}'")
                  # Указываем папку для кеша моделей TTS
                  tts_cache_dir = os.path.join(os.getcwd(), "tts_models_cache")
                  os.environ["TTS_HOME"] = tts_cache_dir # Установка переменной окружения для TTS
                  os.makedirs(tts_cache_dir, exist_ok=True)

                  # Добавляем обработку прерывания во время загрузки модели
                  self.check_cancel()
                  
                  # Уменьшаем используемую память для модели TTS
                  try:
                      # Освобождаем память перед загрузкой большой модели
                      if self.device == "cuda":
                          torch.cuda.empty_cache()
                      
                      # Пытаемся загрузить модель с экономией памяти
                      self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", 
                                           progress_bar=False).to(self.tts_device)
                      logger.info("Модель TTS успешно загружена.")
                  except Exception as e:
                      # Если не удалось, попробуем использовать CPU независимо от настроек
                      logger.warning(f"Не удалось загрузить TTS на выбранное устройство: {e}. Пробуем CPU.")
                      self.tts_device = "cpu"
                      self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                                           progress_bar=False).to("cpu")
                      logger.info("Модель TTS загружена на CPU.")
                      
             except Exception as e:
                  logger.error(f"Не удалось загрузить модель TTS: {e}")
                  # Очистка переменной окружения в случае ошибки
                  if "TTS_HOME" in os.environ: del os.environ["TTS_HOME"]
                  # Освобождаем память в случае ошибки
                  if self.device == "cuda":
                      torch.cuda.empty_cache()
                  if hasattr(self, 'tts_model') and self.tts_model is not None:
                      del self.tts_model
                      self.tts_model = None
                  raise

        generated_files_info = []
        total_segments = len(translated_segments)

        for i, segment in enumerate(translated_segments):
            self.check_cancel()
            text_to_speak = segment['text']
            start_time = segment['start']

            if not text_to_speak:
                logger.info(f"Пропуск генерации TTS для пустого сегмента {i}")
                generated_files_info.append({"start": start_time, "filepath": None})
                continue

            segment_filename = os.path.join(tts_segment_dir, f"segment_{i:04d}.wav") # Добавляем нули для сортировки

            try:
                # Генерация TTS с расширенными параметрами
                # Основная цель - уменьшить шум в генерируемом голосе
                self.tts_model.tts_to_file(
                    text=text_to_speak,
                    file_path=segment_filename,
                    speaker_wav=reference_audio,
                    language=self.target_language,
                    # Дополнительные параметры для минимизации шума
                    speed=1.0  # Нормальная скорость для лучшего качества
                )

                # Проверка, что файл создан и не пустой
                if os.path.exists(segment_filename) and os.path.getsize(segment_filename) > 0:
                    # Проверяем уровень шума в сгенерированном аудио
                    try:
                        tts_audio, tts_sr = librosa.load(segment_filename, sr=None)
                        tts_max_amp = np.max(np.abs(tts_audio))
                        
                        # Если амплитуда слишком низкая, нормализуем
                        if tts_max_amp < 0.01 and tts_max_amp > 0:
                            tts_audio = tts_audio / tts_max_amp * 0.95
                            sf.write(segment_filename, tts_audio, tts_sr, subtype='PCM_16')
                            logger.info(f"Нормализован уровень громкости для сегмента {i}")
                    except Exception as e:
                        logger.warning(f"Не удалось проверить/нормализовать сегмент {i}: {e}")
                    
                    generated_files_info.append({"start": start_time, "filepath": segment_filename})
                    tts_duration = self.get_audio_duration(segment_filename)
                    logger.info(f"TTS {i+1}/{total_segments}: [{start_time:.2f}s] Сгенерирован '{segment_filename}' (длит. {tts_duration:.2f}s)")
                else:
                    logger.error(f"TTS для сегмента {i} не сгенерировал файл или файл пуст: {segment_filename}")
                    generated_files_info.append({"start": start_time, "filepath": None, "error": "TTS failed to generate file"})

            except Exception as e:
                logger.exception(f"Ошибка генерации TTS для сегмента {i} ('{text_to_speak}'):")
                generated_files_info.append({"start": start_time, "filepath": None, "error": str(e)})

            self.progress.emit(60 + int(20 * (i + 1) / total_segments)) # Прогресс 60-80%

        return generated_files_info

    def assemble_audio(self, tts_files_info, total_duration_sec, sample_rate=44100):
        self.check_cancel()
        self.status.emit("Сборка аудиодорожки (синхронизация)...")
        output_path = os.path.join(self.working_dir, "assembled_speech.wav")
        logger.info("Начало сборки финальной аудиодорожки с умным стретчем.")

        # Переменная для отслеживания статуса pyrubberband
        rubberband_available = None
        
        # Флаг для отслеживания проблем с librosa time_stretch
        librosa_timestretch_works = None

        # Добавляем простую функцию изменения темпа через ресемплинг (запасной вариант)
        def simple_resample_timestretch(audio_data, sample_rate, ratio):
            """Простой time stretch через ресемплинг без сохранения высоты звука"""
            # Исправленная логика: при ratio < 1 (ускорение) нам нужно уменьшить количество сэмплов
            # при ratio > 1 (замедление) нам нужно увеличить количество сэмплов
            target_samples = int(len(audio_data) * ratio)
            
            try:
                # Используем scipy.signal.resample вместо librosa.resample для более надежного результата
                from scipy import signal
                resampled = signal.resample(audio_data, target_samples)
                return resampled
            except Exception as e:
                logger.warning(f"Ошибка при простом ресемплинге (scipy): {e}")
                try:
                    # Запасной вариант - librosa
                    resampled = librosa.util.fix_length(audio_data, size=target_samples)
                    return resampled
                except Exception as e2:
                    logger.warning(f"Ошибка при простом ресемплинге (librosa): {e2}")
                    # В случае ошибки возвращаем исходные данные
                    return audio_data

        # Добавляем WSOLA алгоритм для изменения скорости без изменения тона
        def wsola_timestretch(audio_data, sample_rate, ratio):
            """Реализация алгоритма WSOLA (Waveform Similarity Overlap-Add) для изменения скорости без изменения тона"""
            try:
                # Попытка использовать модифицированный phase vocoder из librosa
                hop_length = 512
                n_fft = 2048
                
                # Преобразование в частотную область с помощью STFT
                D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
                
                # Изменяем фазу для изменения скорости без изменения тона
                time_stretched = librosa.phase_vocoder(D, rate=1.0/ratio, hop_length=hop_length)
                
                # Преобразование обратно во временную область
                y_stretched = librosa.istft(time_stretched, hop_length=hop_length)
                
                return y_stretched
            except Exception as e:
                logger.warning(f"Ошибка в WSOLA алгоритме: {e}")
                # Если ошибка, пробуем другой подход через PyTorch
                try:
                    import torch
                    import torch.nn.functional as F
                    
                    # Конвертируем в тензор
                    x = torch.FloatTensor(audio_data)
                    
                    # Используем PyTorch interpolate для изменения скорости
                    # Преобразуем 1D массив в 2D тензор нужной формы [1, 1, length]
                    x_reshaped = x.view(1, 1, -1)
                    
                    # Применяем интерполяцию с линейным режимом
                    resampled = F.interpolate(
                        x_reshaped, 
                        scale_factor=ratio,
                        mode='linear', 
                        align_corners=False
                    )
                    
                    # Получаем обратно 1D массив
                    return resampled.view(-1).numpy()
                    
                except Exception as e2:
                    logger.warning(f"Ошибка в PyTorch stretch: {e2}")
                    # В крайнем случае возвращаемся к простому ресемплингу
                    return simple_resample_timestretch(audio_data, sample_rate, ratio)
        
        # Реализация Pitch-preserving Time-Stretch с использованием фазовой вокодации
        def pitch_preserving_timestretch(audio_data, sample_rate, ratio):
            """Изменение скорости без изменения тона с использованием нескольких методов с автоматическим выбором лучшего"""
            if abs(ratio - 1.0) < 0.05:  # Если изменение меньше 5%, не применяем стретч
                return audio_data
                
            # Пробуем все доступные методы
            methods = []
            
            # 1. Пробуем rubberband через pyrubberband (если доступен)
            try:
                import pyrubberband as pyrb
                # Инвертируем коэффициент для pyrubberband
                stretch_rate = 1.0 / ratio 
                stretched_rb = pyrb.time_stretch(audio_data, sample_rate, stretch_rate)
                methods.append(("rubberband", stretched_rb))
            except Exception as e:
                logger.debug(f"rubberband недоступен: {e}")
            
            # 2. Пробуем librosa.effects.time_stretch
            try:
                stretched_librosa = librosa.effects.time_stretch(audio_data, rate=1.0/ratio)
                methods.append(("librosa", stretched_librosa))
            except Exception as e:
                logger.debug(f"librosa time_stretch недоступен: {e}")
            
            # 3. Пробуем наш алгоритм WSOLA
            try:
                stretched_wsola = wsola_timestretch(audio_data, sample_rate, ratio)
                methods.append(("wsola", stretched_wsola))
            except Exception as e:
                logger.debug(f"WSOLA недоступен: {e}")
            
            # 4. Простой ресемплинг (последний запасной вариант)
            try:
                stretched_simple = simple_resample_timestretch(audio_data, sample_rate, ratio)
                methods.append(("simple", stretched_simple))
            except Exception as e:
                logger.debug(f"simple resample недоступен: {e}")
            
            # Если ни один метод не сработал, возвращаем исходные данные
            if not methods:
                logger.warning("Ни один метод time stretch не сработал. Возвращаем оригинальный аудио.")
                return audio_data
                
            # Выбираем первый успешный метод (в порядке предпочтения)
            method_name, stretched_audio = methods[0]
            logger.info(f"Используем метод time stretch: {method_name}")
            
            return stretched_audio

        # Рассчитываем общую длину холста в сэмплах с небольшим запасом
        total_samples = math.ceil((total_duration_sec + 2.0) * sample_rate)
        logger.info(f"Создание холста для сборки: {total_samples} сэмплов ({total_samples/sample_rate:.2f} сек)")
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        # Фильтруем только существующие TTS файлы и сортируем по времени начала
        valid_segments = [(i, info) for i, info in enumerate(tts_files_info) if info.get('filepath') is not None]
        valid_segments.sort(key=lambda x: x[1]['start'])  # Сортируем по времени начала
        
        total_valid_segments = len(valid_segments)
        if total_valid_segments == 0:
            logger.warning("Нет действительных TTS сегментов для сборки!")
            return output_path
            
        logger.info(f"Найдено {total_valid_segments} действительных TTS сегментов для сборки")
        
        # Константы для умного стретча
        MIN_STRETCH_RATIO = 0.5  # Максимально до 50% ускорения (было 0.7 / 30%)
        MAX_STRETCH_RATIO = 1.7  # Максимально до 70% замедления (было 1.4 / 40%)
        CROSSFADE_DUR_SEC = 0.1  # Длительность кросс-фейда в секундах
        crossfade_samples = int(CROSSFADE_DUR_SEC * sample_rate)
        
        # Проходим по сортированным сегментам
        for idx, (segment_idx, segment_info) in enumerate(valid_segments):
            self.check_cancel()
            
            # Определяем временные границы сегмента
            start_time = segment_info['start']
            tts_filepath = segment_info['filepath']
            
            # Если это не последний сегмент, используем время начала следующего как конец текущего
            if idx < len(valid_segments) - 1:
                next_start_time = valid_segments[idx + 1][1]['start']
                end_time = next_start_time
            else:
                # Для последнего сегмента оценим длительность из аудиофайла
                tts_duration = self.get_audio_duration(tts_filepath)
                end_time = start_time + tts_duration
                
            target_duration = end_time - start_time
            
                # Загружаем TTS-фрагмент
            try:
                tts_data, tts_sr = librosa.load(tts_filepath, sr=sample_rate, res_type='kaiser_fast')
                original_duration = len(tts_data) / sample_rate
                
                # Вычисляем коэффициент растяжения/сжатия
                # target_duration / original_duration = ratio, если ratio < 1 - ускоряем, если > 1 - замедляем
                stretch_ratio = target_duration / original_duration
                
                # Ограничиваем коэффициент в значительно более узких пределах для сохранения качества
                stretch_ratio = max(MIN_STRETCH_RATIO, min(MAX_STRETCH_RATIO, stretch_ratio))
                
                # Применяем time stretch для изменения длительности без изменения высоты звука
                if abs(stretch_ratio - 1.0) > 0.03:  # Если разница больше 3% (было 5%)
                    logger.info(f"Сегмент {segment_idx}: Применение time stretch с коэффициентом {stretch_ratio:.2f} "
                               f"(с {original_duration:.2f}s до {target_duration:.2f}s)")
                    
                    # Обработчик time_stretch с использованием разных методов и обработкой ошибок
                    stretched_data = None
                    stretch_method_used = None
                    
                    # 1. Пробуем pyrubberband, если доступен
                    if rubberband_available is None:
                        try:
                            import pyrubberband as pyrb
                            # Тестируем работу с маленьким массивом
                            test_data = np.zeros(1000, dtype=np.float32)
                            _ = pyrb.time_stretch(test_data, sample_rate, 0.9)
                            rubberband_available = True
                            logger.info("Pyrubberband доступен и будет использован для более качественного time stretch")
                        except (ImportError, RuntimeError) as e:
                            rubberband_available = False
                            logger.warning(f"Pyrubberband недоступен: {e}")
                    
                    if rubberband_available:
                        try:
                            import pyrubberband as pyrb
                            # Инвертируем коэффициент - pyrubberband принимает скорость, а не коэффициент длительности
                            stretch_rate = 1.0 / stretch_ratio
                            logger.info(f"Сегмент {segment_idx}: Используем pyrubberband с rate = {stretch_rate:.2f} (ratio = {stretch_ratio:.2f})")
                            stretched_data = pyrb.time_stretch(tts_data, tts_sr, stretch_rate)
                            stretch_method_used = "pyrubberband"
                        except Exception as e:
                            logger.warning(f"Ошибка pyrubberband для сегмента {segment_idx}: {e}")
                            stretched_data = None
                    
                    # 2. Пробуем librosa.effects.time_stretch если pyrubberband не сработал
                    if stretched_data is None and librosa_timestretch_works is not False:
                        try:
                            stretched_data = librosa.effects.time_stretch(tts_data, rate=1.0/stretch_ratio)
                            stretch_method_used = "librosa time_stretch"
                            if librosa_timestretch_works is None:
                                librosa_timestretch_works = True
                        except Exception as e:
                            logger.warning(f"Ошибка librosa time_stretch для сегмента {segment_idx}: {e}")
                            librosa_timestretch_works = False
                            stretched_data = None
                    
                    # 3. Используем простой ресемплинг как запасной вариант
                    if stretched_data is None:
                        try:
                            # Используем улучшенный алгоритм, сохраняющий тональность
                            stretched_data = pitch_preserving_timestretch(tts_data, sample_rate, stretch_ratio)
                            stretch_method_used = "pitch-preserving stretch"
                        except Exception as e:
                            logger.warning(f"Ошибка pitch-preserving stretch для сегмента {segment_idx}: {e}")
                            # Используем исходные данные, если все методы стретчинга не сработали
                            stretched_data = tts_data
                            stretch_method_used = "none (using original)"
                    
                    # Используем результат стретчинга
                    if stretched_data is not None:
                        tts_data = stretched_data
                        logger.info(f"Для сегмента {segment_idx} использован метод: {stretch_method_used}")
                
                # Вычисляем позицию в сэмплах для вставки
                start_sample = int(start_time * sample_rate)
                end_sample = start_sample + len(tts_data)
                
                # Проверяем границы холста
                if start_sample < 0:
                    # Отрезаем начало, если выходит за левую границу
                    tts_data = tts_data[-start_sample:]
                    start_sample = 0
                
                if end_sample > total_samples:
                    # Отрезаем конец, если выходит за правую границу
                    tts_data = tts_data[:total_samples-start_sample]
                    end_sample = total_samples
                
                # Если после обрезки ничего не осталось
                if len(tts_data) == 0:
                    logger.warning(f"Сегмент {segment_idx}: После обрезки по границам холста не осталось данных")
                    continue
                
                # Обработка перекрытий - плавное смешивание (кросс-фейд)
                existing_audio = final_audio[start_sample:end_sample]
                
                # Проверяем, есть ли уже данные в этом диапазоне
                if np.max(np.abs(existing_audio)) > 0.001:  # Если уже есть данные
                    # Создаем маски для кросс-фейда
                    overlap_len = min(len(tts_data), len(existing_audio))
                    fade_in = np.linspace(0, 1, min(crossfade_samples, overlap_len))
                    fade_out = np.linspace(1, 0, min(crossfade_samples, overlap_len))
                    
                    # Применяем только в начале и конце перекрытия
                    if overlap_len > crossfade_samples * 2:
                        fade_mask = np.ones(overlap_len)
                        fade_mask[:crossfade_samples] = fade_in
                        fade_mask[-crossfade_samples:] = fade_out
                    else:
                        # Для коротких перекрытий - линейный фейд
                        fade_mask = np.linspace(0, 1, overlap_len)
                    
                    # Микшируем с учетом фейдов
                    mixed = existing_audio * (1 - fade_mask) + tts_data[:overlap_len] * fade_mask
                    final_audio[start_sample:start_sample+overlap_len] = mixed
                else:
                    # Если перекрытий нет, просто записываем данные
                    final_audio[start_sample:end_sample] = tts_data
                
                logger.info(f"Сегмент {segment_idx} [{idx+1}/{total_valid_segments}]: "
                           f"Время={start_time:.2f}s→{end_time:.2f}s, "
                           f"Длит.={len(tts_data)/sample_rate:.2f}s, "
                           f"Позиция={start_sample}→{end_sample}")

            except Exception as e:
                logger.exception(f"Ошибка при обработке сегмента {segment_idx} ({tts_filepath}):")
                continue

            self.progress.emit(80 + int(10 * (idx + 1) / total_valid_segments))  # Прогресс 80-90%

        # Обрезаем тишину в конце
        try:
            non_silent_indices = np.where(np.abs(final_audio) > 0.0001)[0]
            if len(non_silent_indices) > 0:
                last_non_silent_sample = non_silent_indices[-1]
                final_audio = final_audio[:last_non_silent_sample + int(0.5 * sample_rate)]  # Добавляем 0.5 сек тишины в конце
                logger.info(f"Обрезана тишина в конце. Итоговая длина {len(final_audio)} сэмплов ({len(final_audio)/sample_rate:.2f} сек)")
            else:
                logger.warning("Собранный аудиофайл полностью состоит из тишины!")
        except Exception as e:
            logger.error(f"Ошибка при обрезке тишины в конце: {e}")

        # Нормализация финального аудио
        try:
            max_amp = np.max(np.abs(final_audio))
            if max_amp > 0:
                final_audio = final_audio * 0.9 / max_amp  # 90% от максимальной амплитуды
                logger.info(f"Аудио нормализовано до уровня 90% (коэффициент {0.9 / max_amp:.2f})")
        except Exception as e:
            logger.error(f"Ошибка при нормализации аудио: {e}")

        # Сохраняем результат
        try:
            sf.write(output_path, final_audio, sample_rate, subtype='PCM_16')
            logger.info(f"Собранная аудиодорожка сохранена: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Не удалось сохранить собранную аудиодорожку {output_path}: {str(e)}")
            raise

    def mix_audio(self, assembled_speech_path, background_path, target_sr=44100):
        self.check_cancel()
        self.status.emit("Микширование аудио...")
        output_path = os.path.join(self.working_dir, "final_mixed_audio.wav")
        logger.info(f"Микширование '{assembled_speech_path}' и '{background_path}' -> '{output_path}'")

        try:
            speech, sr_s = librosa.load(assembled_speech_path, sr=target_sr)
            background, sr_b = librosa.load(background_path, sr=target_sr)

            # Определяем максимальную длину
            max_len = max(len(speech), len(background))

            # Паддинг до одинаковой длины
            speech = librosa.util.fix_length(speech, size=max_len)
            background = librosa.util.fix_length(background, size=max_len)

            # Используем предустановленные значения громкости без нормализации
            mixed = speech * self.speech_gain + background * self.background_gain

            # Предотвращение клиппинга
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
                logger.warning("Обнаружен клиппинг при микшировании, применена нормализация.")
            elif max_val == 0:
                 logger.warning("Результат микширования - тишина.")

            sf.write(output_path, mixed, target_sr, subtype='PCM_16')
            logger.info(f"Финальное смикшированное аудио сохранено: {output_path}")
            return output_path

        except Exception as e:
             logger.exception("Ошибка микширования аудио:")
             raise

    def merge_audio_video(self, audio_path):
        self.check_cancel()
        self.status.emit("Сборка финального видео...")
        # Определяем, какое видео использовать (оригинал или с синхронизированными губами)
        video_path_to_use = self.video_path
        if self.use_wav2lip and hasattr(self, 'lipsync_video_path') and self.lipsync_video_path and os.path.exists(self.lipsync_video_path):
            video_path_to_use = self.lipsync_video_path
            logger.info(f"Используем видео с синхронизированными губами: {video_path_to_use}")
        else:
            logger.info(f"Используем оригинальное видео: {video_path_to_use}")
            
        # Создаем имя выходного файла на основе входного
        input_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_video = f"{input_basename}_dubbed_{self.target_language}.mp4"
        if self.use_wav2lip and video_path_to_use != self.video_path:
            output_video = f"{input_basename}_dubbed_{self.target_language}_lipsync.mp4"
            
        logger.info(f"Сборка видео '{video_path_to_use}' с аудио '{audio_path}' -> '{output_video}'")
        logger.info("Используется опция '-shortest' для обрезки по длине видео.")

        try:
            input_video = ffmpeg.input(video_path_to_use)
            input_audio = ffmpeg.input(audio_path)

            # Определяем самый короткий поток (обычно видео) и используем его длительность
            stream = ffmpeg.output(
                input_video['v'],    # Явно указываем видеопоток
                input_audio['a'],    # Явно указываем аудиопоток
                output_video,
                acodec='aac',        # Стандартный кодек для mp4
                vcodec='copy',       # Не перекодируем видео
                shortest=None,       # <-- Обрезка по самому короткому входу
                strict='experimental', # Может быть нужно для aac
                progress='-',        # Вывод прогресса в stdout
                loglevel='error'     # Уровень логирования
            )

            # Запуск ffmpeg с захватом stderr
            stdout, stderr = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)

            # Логирование вывода ffmpeg
            if stderr:
                 # Логируем stderr как ошибку для лучшей видимости
                 logger.error("ffmpeg stderr:\n%s", stderr.decode(errors='ignore'))

            # Проверка, существует ли выходной файл
            if not os.path.exists(output_video):
                 logger.error(f"ffmpeg завершился, но выходной файл '{output_video}' не найден!")
                 raise FileNotFoundError(f"Выходной файл ffmpeg не создан: {output_video}")

            logger.info(f"Финальное видео успешно собрано: {output_video}")
            return output_video
        except ffmpeg.Error as e:
            # Логируем stderr из исключения ffmpeg.Error
            stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else "Нет stderr"
            logger.error(f"Ошибка ffmpeg при сборке видео: {stderr_decoded}")
            raise RuntimeError(f"Ошибка ffmpeg: {stderr_decoded}") # Преобразуем в RuntimeError
        except Exception as e:
             logger.exception("Неожиданная ошибка при сборке видео:")
             raise

    def run_wav2lip(self, video_source, audio_source):
        """
        Запускает Wav2Lip для синхронизации губ на видео с аудио.
        
        Args:
            video_source: Путь к исходному видео
            audio_source: Путь к аудио для синхронизации губ
            
        Returns:
            Путь к видео с синхронизированными губами
        """
        self.check_cancel()
        self.status.emit("Синхронизация губ (Wav2Lip)...")
        logger.info(f"Запуск Wav2Lip для видео '{video_source}' и аудио '{audio_source}'")
        
        # Создаем уникальный кеш-ключ на основе имени входного видео и параметров
        video_basename = os.path.splitext(os.path.basename(video_source))[0]
        cache_key = f"{video_basename}_lipsync_rf{self.resize_factor}"  # Включаем resize_factor в ключ кеша
        cache_dir = os.path.join("temp_processing", "wav2lip_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cached_output = os.path.join(cache_dir, f"{cache_key}.mp4")
        
        # Проверяем, существует ли кешированный результат
        if os.path.exists(cached_output):
            self.status.emit("Найден кешированный результат Wav2Lip, используем его...")
            logger.info(f"Используем кешированный результат Wav2Lip: {cached_output}")
            
            # Копируем кешированный файл в рабочую директорию
            output_lipsync_video = os.path.join(self.working_dir, "lipsync_result.mp4")
            try:
                shutil.copy2(cached_output, output_lipsync_video)
                logger.info(f"Кешированный файл успешно скопирован в: {output_lipsync_video}")
                return output_lipsync_video
            except Exception as e:
                logger.warning(f"Не удалось скопировать кешированный файл: {e}, запускаем Wav2Lip заново")
                # Если не удалось скопировать, продолжаем обычную обработку
        else:
            logger.info(f"Кешированный результат не найден, запускаем Wav2Lip")
            
        output_lipsync_video = os.path.join(self.working_dir, "lipsync_result.mp4")
        
        # Формируем команду для запуска inference.py
        command = [
            WAV2LIP_PYTHON_PATH,
            WAV2LIP_INFERENCE_SCRIPT,
            "--checkpoint_path", WAV2LIP_CHECKPOINT,
            "--face", video_source,  # Видео с лицом
            "--audio", audio_source, # Аудио для синхронизации
            "--outfile", output_lipsync_video,
            # Дополнительные параметры при необходимости
            "--pads", "0", "0", "0", "0",  # Отступы [top, bottom, left, right]
            "--face_det_batch_size", "16",
            "--wav2lip_batch_size", "64",
            "--resize_factor", str(self.resize_factor),  # Используем значение из параметра класса
        ]
        
        # Wav2Lip сам определит доступность GPU через PyTorch
        # Строка с добавлением --gpu удалена, так как этот аргумент не поддерживается
        
        logger.info(f"Команда Wav2Lip: {' '.join(command)}")
        
        try:
            # Запускаем Wav2Lip как внешний процесс
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            
            # Логируем вывод
            if process.stdout:
                logger.info(f"Wav2Lip stdout:\n{process.stdout}")
            if process.stderr:
                if process.returncode != 0:
                    logger.error(f"Wav2Lip stderr:\n{process.stderr}")
                else:
                    logger.info(f"Wav2Lip stderr (предупреждения):\n{process.stderr}")
            
            # Проверка кода возврата
            if process.returncode != 0:
                raise RuntimeError(f"Wav2Lip завершился с ошибкой (код {process.returncode})")
            
            # Проверка существования выходного файла
            if not os.path.exists(output_lipsync_video):
                raise FileNotFoundError(f"Wav2Lip не создал ожидаемый выходной файл: {output_lipsync_video}")
            
            logger.info(f"Wav2Lip успешно завершен. Результат: {output_lipsync_video}")
            
            # Сохраняем результат в кеш для будущего использования
            try:
                shutil.copy2(output_lipsync_video, cached_output)
                logger.info(f"Результат Wav2Lip сохранен в кеш: {cached_output}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить результат в кеш: {e}")
            
            return output_lipsync_video
            
        except FileNotFoundError as e:
            logger.error(f"Ошибка запуска Wav2Lip: {e}")
            raise RuntimeError(f"Не удалось запустить Wav2Lip: {e}")
        except Exception as e:
            logger.exception("Ошибка при выполнении Wav2Lip:")
            raise

    def check_cache(self, step_name):
        """Проверяет наличие кешированных результатов для определенного шага обработки"""
        cache_path = os.path.join(self.cache_dir, f"{self.cache_key}_{step_name}")
        return os.path.exists(cache_path), cache_path

    def save_to_cache(self, step_name, file_path):
        """Сохраняет результат в кеш"""
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Невозможно сохранить в кеш несуществующий файл: {file_path}")
            return False
            
        cache_path = os.path.join(self.cache_dir, f"{self.cache_key}_{step_name}")
        try:
            shutil.copy2(file_path, cache_path)
            logger.info(f"Результат шага '{step_name}' сохранен в кеш: {cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Не удалось сохранить в кеш результат шага '{step_name}': {e}")
            return False
            
    def load_from_cache(self, step_name, target_path=None):
        """Загружает результат из кеша"""
        cache_exists, cache_path = self.check_cache(step_name)
        if not cache_exists:
            logger.info(f"В кеше не найден результат для шага '{step_name}'")
            return None
            
        if not target_path:
            return cache_path
            
        try:
            shutil.copy2(cache_path, target_path)
            logger.info(f"Результат шага '{step_name}' загружен из кеша в {target_path}")
            return target_path
        except Exception as e:
            logger.warning(f"Не удалось загрузить из кеша результат шага '{step_name}': {e}")
            return None

    def run(self):
        self.status.emit("Начало обработки...")
        output_video_path = None
        video_source_for_final_merge = self.video_path  # По умолчанию используем оригинальное видео
        
        # Проверяем наличие финального результата в кеше
        cache_exists, cached_output = self.check_cache("final_output")
        if cache_exists:
            try:
                output_basename = os.path.basename(cached_output)
                output_video_path = os.path.join(os.path.dirname(self.video_path), output_basename)
                shutil.copy2(cached_output, output_video_path)
                logger.info(f"Найден кешированный финальный результат. Файл скопирован в: {output_video_path}")
                self.progress.emit(100)
                self.status.emit(f"Готово! Результат загружен из кеша: {output_video_path}")
                self.finished.emit(output_video_path)
                return
            except Exception as e:
                logger.warning(f"Не удалось использовать кешированный финальный результат: {e}")
                # Продолжаем обычную обработку
        
        try:
            # --- Шаг 1: Извлечение аудио ---
            self.progress.emit(5)
            cache_exists, cached_audio = self.check_cache("original_audio")
            if cache_exists:
                self.status.emit("Загрузка извлеченного аудио из кеша...")
                original_audio_path = os.path.join(self.working_dir, "original_audio.wav")
                self.load_from_cache("original_audio", original_audio_path)
                logger.info(f"Извлеченное аудио загружено из кеша: {original_audio_path}")
            else:
                self.status.emit("Извлечение аудио...")
                original_audio_path = self.extract_audio()
                self.save_to_cache("original_audio", original_audio_path)

            # --- Шаг 2: Разделение аудио ---
            self.progress.emit(15)
            vocals_cached, cached_vocals = self.check_cache("vocals")
            background_cached, cached_background = self.check_cache("background")
            
            if vocals_cached and background_cached:
                self.status.emit("Загрузка разделенного аудио из кеша...")
                vocals_path = os.path.join(self.working_dir, "vocals.wav")
                background_path = os.path.join(self.working_dir, "background.wav")
                self.load_from_cache("vocals", vocals_path)
                self.load_from_cache("background", background_path)
                logger.info(f"Разделенное аудио загружено из кеша: {vocals_path}, {background_path}")
            else:
                self.status.emit("Разделение аудио (Demucs)...")
                vocals_path, background_path = self.separate_audio(original_audio_path)
                self.save_to_cache("vocals", vocals_path)
                self.save_to_cache("background", background_path)

            # --- Шаг 3: Референс для TTS ---
            self.progress.emit(25)
            reference_cached, cached_reference = self.check_cache("reference_audio")
            if reference_cached:
                self.status.emit("Загрузка референсного аудио из кеша...")
                # Проверяем, что кеш содержит корректный референсный аудио из vocals.wav
                vocals_path_basename = os.path.basename(vocals_path)
                if cached_reference and os.path.exists(cached_reference):
                    logger.info(f"Референсное аудио загружено из кеша: {cached_reference}")
                    reference_audio = cached_reference
                else:
                    logger.warning("Кешированный референс не найден или недоступен. Создаем новый из vocals.wav")
                    reference_audio = self.get_reference_audio(vocals_path)
                    self.save_to_cache("reference_audio", reference_audio)
            else:
                self.status.emit("Создание референсного аудио...")
                # Обязательно используем демукс-голос (vocals.wav)
                reference_audio = self.get_reference_audio(vocals_path)
                self.save_to_cache("reference_audio", reference_audio)

            # --- Шаг 4: Распознавание речи ---
            self.progress.emit(35)
            segments_cached, cached_segments = self.check_cache("segments")
            if segments_cached:
                self.status.emit("Загрузка распознанной речи из кеша...")
                try:
                    with open(cached_segments, 'r', encoding='utf-8') as f:
                        original_segments = eval(f.read())
                    logger.info(f"Распознанные сегменты загружены из кеша: {len(original_segments)} сегментов")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить сегменты из кеша: {e}, выполняем распознавание")
                    original_segments = self.transcribe_audio(vocals_path)
                    with open(os.path.join(self.cache_dir, f"{self.cache_key}_segments"), 'w', encoding='utf-8') as f:
                        f.write(str(original_segments))
            else:
                self.status.emit("Распознавание речи...")
                original_segments = self.transcribe_audio(vocals_path)
                with open(os.path.join(self.cache_dir, f"{self.cache_key}_segments"), 'w', encoding='utf-8') as f:
                    f.write(str(original_segments))
                
            if not original_segments:
                raise ValueError("Не удалось распознать речь (Whisper вернул 0 сегментов).")

            # --- Шаг 5: Перевод текста ---
            self.progress.emit(50)
            translations_cached, cached_translations = self.check_cache("translations")
            if translations_cached:
                self.status.emit("Загрузка переведенного текста из кеша...")
                try:
                    with open(cached_translations, 'r', encoding='utf-8') as f:
                        translated_segments = eval(f.read())
                    logger.info(f"Переведенные сегменты загружены из кеша: {len(translated_segments)} сегментов")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить переводы из кеша: {e}, выполняем перевод")
                    translated_segments = self.translate_text(original_segments)
                    with open(os.path.join(self.cache_dir, f"{self.cache_key}_translations"), 'w', encoding='utf-8') as f:
                        f.write(str(translated_segments))
            else:
                self.status.emit("Перевод текста...")
                translated_segments = self.translate_text(original_segments)
                with open(os.path.join(self.cache_dir, f"{self.cache_key}_translations"), 'w', encoding='utf-8') as f:
                    f.write(str(translated_segments))

            # --- Шаг 6: Генерация речи (TTS) ---
            self.progress.emit(60)
            tts_cached, cached_tts = self.check_cache("tts_info")
            if tts_cached:
                self.status.emit("Загрузка сгенерированных аудио из кеша...")
                tts_files_info = self._load_tts_from_cache(cached_tts, translated_segments, reference_audio)
            else:
                self.status.emit("Генерация речи (TTS)...")
                tts_files_info = self.generate_speech(translated_segments, reference_audio)
                self._save_tts_to_cache(tts_files_info)
                
            # --- Шаг 7: Сборка аудиодорожки ---
            self.progress.emit(80)
            assembled_cached, cached_assembled = self.check_cache("assembled_audio")
            if assembled_cached:
                self.status.emit("Загрузка собранной аудиодорожки из кеша...")
                assembled_audio_path = os.path.join(self.working_dir, "assembled_speech.wav")
                self.load_from_cache("assembled_audio", assembled_audio_path)
                logger.info(f"Собранная аудиодорожка загружена из кеша: {assembled_audio_path}")
            else:
                self.status.emit("Сборка аудиодорожки...")
                original_vocals_duration = self.get_audio_duration(vocals_path)
                if original_vocals_duration == 0:
                    if original_segments: original_vocals_duration = original_segments[-1]['end']
                    else: original_vocals_duration = 1 # Fallback
                    logger.warning(f"Не удалось точно определить длительность вокала, используем {original_vocals_duration:.2f}s")
                logger.info(f"Ожидаемая максимальная длительность для сборки аудио: {original_vocals_duration:.2f} секунд.")
                assembled_audio_path = self.assemble_audio(tts_files_info, original_vocals_duration)
                self.save_to_cache("assembled_audio", assembled_audio_path)

            # --- Шаг 8: Микширование с фоном ---
            self.progress.emit(90)
            final_audio_cached, cached_final_audio = self.check_cache("final_audio")
            if final_audio_cached:
                self.status.emit("Загрузка готового аудио из кеша...")
                final_audio_path = os.path.join(self.working_dir, "final_mixed_audio.wav")
                self.load_from_cache("final_audio", final_audio_path)
                logger.info(f"Готовое аудио загружено из кеша: {final_audio_path}")
            else:
                self.status.emit("Микширование аудио...")
                final_audio_path = self.mix_audio(assembled_audio_path, background_path)
                self.save_to_cache("final_audio", final_audio_path)

            # --- Шаг 8.5: Синхронизация губ с Wav2Lip (опционально) ---
            if self.use_wav2lip:
                self.progress.emit(92)
                lipsync_cached, cached_lipsync = self.check_cache("lipsync_video")
                if lipsync_cached:
                    self.status.emit("Загрузка видео с синхронизацией губ из кеша...")
                    self.lipsync_video_path = os.path.join(self.working_dir, "lipsync_result.mp4")
                    self.load_from_cache("lipsync_video", self.lipsync_video_path)
                    logger.info(f"Видео с синхронизацией губ загружено из кеша: {self.lipsync_video_path}")
                else:
                    # Запускаем Wav2Lip на оригинальном видео и финальном аудио
                    try:
                        self.lipsync_video_path = self.run_wav2lip(self.video_path, final_audio_path)
                        self.save_to_cache("lipsync_video", self.lipsync_video_path)
                        self.progress.emit(95)
                    except Exception as e:
                        logger.error(f"Ошибка при синхронизации губ: {e}")
                        # Если Wav2Lip не удался, продолжаем с оригинальным видео
                        self.lipsync_video_path = None
                        self.status.emit(f"Ошибка синхронизации губ, продолжаем без неё: {str(e)}")
            else:
                self.progress.emit(95)

            # --- Шаг 9: Сборка видео ---
            self.status.emit("Сборка финального видео...")
            output_video_path = self.merge_audio_video(final_audio_path)
            self.save_to_cache("final_output", output_video_path)
            
            self.progress.emit(100)
            self.status.emit(f"Готово! Результат: {output_video_path}")
            logger.info("Обработка успешно завершена.")
            self.finished.emit(output_video_path) # Передаем путь к файлу

        except InterruptedError:
             self.status.emit("Обработка отменена.")
             logger.info("Обработка была отменена.")
             self.error.emit("Отменено") # Используем сигнал ошибки для отмены

        except Exception as e:
            logger.exception("Критическая ошибка в процессе обработки:") # Логируем полный traceback
            self.status.emit(f"Ошибка: {e}")
            self.error.emit(str(e)) # Передаем текст ошибки в GUI

        finally:
            # Очистка CUDA кэша в конце, если используется
            if hasattr(self, 'device') and self.device == "cuda":
                 logger.info("Очистка CUDA кэша...")
                 torch.cuda.empty_cache()
            # Можно добавить очистку временной папки, но делать это лучше в GUI после получения finished/error
            logger.info(f"Обработка в потоке завершена (возможно, с ошибкой). Временная папка: {self.working_dir}")

    def _save_tts_to_cache(self, tts_files_info):
        """Вспомогательный метод для сохранения TTS файлов в кеш"""
        try:
            # Выводим дополнительную информацию для диагностики
            logger.info(f"Сохранение TTS в кеш. Количество файлов: {len(tts_files_info) if isinstance(tts_files_info, list) else 'не список'}")
            
            # Проверяем корректность структуры данных
            if not isinstance(tts_files_info, list):
                logger.warning(f"Неверный формат TTS информации (не список): {type(tts_files_info)}")
                return
                
            # Сохраняем информацию в кеш
            with open(os.path.join(self.cache_dir, f"{self.cache_key}_tts_info"), 'w', encoding='utf-8') as f:
                f.write(str(tts_files_info))
            
            # Создаем директорию для tts файлов
            tts_cache_dir = os.path.join(self.cache_dir, f"{self.cache_key}_tts_files")
            os.makedirs(tts_cache_dir, exist_ok=True)
            logger.info(f"Директория кеша для TTS файлов: {tts_cache_dir}")
            
            # Сохраняем оригинальные имена файлов для восстановления
            file_mapping = {}
            valid_file_count = 0
            
            # Копируем tts файлы в директорию кеша
            for i, info in enumerate(tts_files_info):
                if 'path' not in info:
                    logger.warning(f"Пропуск файла TTS: ключ 'path' отсутствует в элементе {i}")
                    continue
                    
                src_path = info['path']
                logger.info(f"Проверка исходного TTS файла: {src_path}")
                
                if not os.path.exists(src_path):
                    logger.warning(f"Пропуск файла TTS: исходный файл не существует: {src_path}")
                    continue
                
                # Сохраняем оригинальное имя файла
                original_filename = os.path.basename(src_path)
                cache_filename = f"tts_{i}.wav"
                file_mapping[cache_filename] = original_filename
                
                dst_path = os.path.join(tts_cache_dir, cache_filename)
                shutil.copy2(src_path, dst_path)
                valid_file_count += 1
                logger.info(f"TTS файл {original_filename} скопирован в кеш как {cache_filename}")
            
            # Сохраняем маппинг имен файлов
            mapping_path = os.path.join(self.cache_dir, f"{self.cache_key}_tts_mapping")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                f.write(str(file_mapping))
                
            logger.info(f"TTS информация и файлы сохранены в кеш, всего файлов: {valid_file_count}")
        except Exception as e:
            logger.exception(f"Ошибка при сохранении TTS в кеш: {e}")
            # Продолжаем выполнение даже при ошибке кеширования
            
            
    def _load_tts_from_cache(self, cached_tts, translated_segments, reference_audio):
        """Вспомогательный метод для загрузки TTS файлов из кеша"""
        try:
            with open(cached_tts, 'r', encoding='utf-8') as f:
                tts_files_info_str = f.read()
            
            # Проверяем наличие маппинга имен файлов
            mapping_path = os.path.join(self.cache_dir, f"{self.cache_key}_tts_mapping")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    file_mapping = eval(f.read())
            else:
                file_mapping = {}
                logger.warning("Файл маппинга имен не найден, используем стандартные имена")
            
            # Создаем копию tts файлов из кеша в рабочую директорию
            tts_cache_dir = os.path.join(self.cache_dir, f"{self.cache_key}_tts_files")
            if os.path.exists(tts_cache_dir):
                tts_files_info = eval(tts_files_info_str)
                
                # Создаем директорию для tts сегментов если её нет
                tts_segments_dir = os.path.join(self.working_dir, "tts_segments")
                os.makedirs(tts_segments_dir, exist_ok=True)
                
                # Восстанавливаем файлы из кеша и обновляем пути в tts_files_info
                new_tts_files_info = []
                
                # Выводим дополнительный лог для диагностики
                logger.info(f"Путь к директории TTS сегментов: {tts_segments_dir}")
                logger.info(f"Путь к кешу TTS файлов: {tts_cache_dir}")
                logger.info(f"Количество кешированных TTS файлов: {len(os.listdir(tts_cache_dir))}")
                
                for i, info in enumerate(tts_files_info):
                    cache_filename = f"tts_{i}.wav"
                    cached_file = os.path.join(tts_cache_dir, cache_filename)
                    
                    if not os.path.exists(cached_file):
                        logger.warning(f"TTS файл {cached_file} не найден в кеше")
                        continue
                    
                    # Определяем оригинальное имя файла или используем стандартное
                    if cache_filename in file_mapping:
                        original_filename = file_mapping[cache_filename]
                    else:
                        original_filename = f"segment_{i:04d}.wav"
                    
                    # Создаем путь для восстановленного файла
                    target_file = os.path.join(tts_segments_dir, original_filename)
                    
                    # Копируем файл из кеша
                    shutil.copy2(cached_file, target_file)
                    
                    # Обновляем путь в информации
                    new_info = info.copy()
                    new_info['path'] = target_file
                    new_tts_files_info.append(new_info)
                    
                    logger.info(f"TTS файл {cache_filename} восстановлен как {original_filename}")
                
                if len(new_tts_files_info) > 0:
                    logger.info(f"TTS информация загружена из кеша: {len(new_tts_files_info)} файлов")
                    return new_tts_files_info
                else:
                    logger.warning("Не удалось восстановить ни один TTS файл из кеша")
                    raise ValueError("Не удалось восстановить файлы из кеша")
            else:
                raise FileNotFoundError(f"Директория с TTS файлами не найдена: {tts_cache_dir}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить TTS из кеша: {e}, генерируем заново")
            self.status.emit("Генерация речи (TTS)...")
            return self.generate_speech(translated_segments, reference_audio)


# --- Классы GUI (StyledFrame, AnimatedProgressBar, MainWindow) ---
# ... (Они остаются такими же, как ты предоставил в предыдущих сообщениях)
# ... ВАЖНО: Убедись, что класс MainWindow импортирует shutil, если он отвечает за очистку папки

class StyledFrame(QFrame):
    """Стилизованная карточка для группировки элементов"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("styledFrame")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        # Стиль будет применен в apply_theme MainWindow

    def update_style(self, is_dark):
        # Стили перенесены в apply_theme для централизации
        pass # Оставляем пустым, стиль задается глобально

class AnimatedProgressBar(QProgressBar):
    """Прогресс-бар с анимацией"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._animation = QPropertyAnimation(self, b"value")
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.setDuration(300) # Чуть быстрее анимация
        # Стиль будет применен в apply_theme MainWindow

    def update_style(self, is_dark):
        # Стили перенесены в apply_theme для централизации
        pass # Оставляем пустым, стиль задается глобально

    def setValue(self, value):
        # Предотвращаем анимацию при сбросе на 0 или установке 100
        if self.value() == 0 or value == 0 or value == 100:
             super().setValue(value)
        else:
            self._animation.stop()
            self._animation.setStartValue(self.value())
            self._animation.setEndValue(value)
            self._animation.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSynch")
        self.setGeometry(100, 100, 600, 680) # Больше места для комфортного использования

        try:
            app_icon = QIcon("icon.png")
            if not app_icon.isNull():
                 self.setWindowIcon(app_icon)
            else:
                 logger.warning("Файл иконки 'icon.png' не найден или пуст.")
        except Exception as e:
             logger.error(f"Ошибка загрузки иконки: {e}")

        # Обновленные цветовые схемы в стиле Windows 11
        self.color_schemes = {
            'dark': {
                'primary': '#60CDFF',       # Акцентный синий Windows 11
                'secondary': '#4CC2FF',     # Светло-синий
                'background': '#202020',    # Темный фон
                'surface': '#2C2C2C',       # Карточки/контейнеры 
                'surface_hover': '#333333', # При наведении
                'mica': 'rgba(44, 44, 44, 0.85)', # Эффект Mica
                'text': '#FFFFFF',          # Белый текст
                'text_secondary': 'rgba(255, 255, 255, 0.7)',
                'button': '#60CDFF',        # Кнопки в цвет акцента
                'button_hover': '#82D9FF',  # Светлее при наведении
                'button_pressed': '#3CB4F2', # Темнее при нажатии
                'disabled_bg': 'rgba(255, 255, 255, 0.1)',
                'disabled_fg': 'rgba(255, 255, 255, 0.4)',
                'error': '#FF7875',         # Красный для ошибок
                'error_hover': '#FF9692',   # Светлее красный
                'border': 'rgba(255, 255, 255, 0.15)', # Более заметная рамка
                'progress_bg': 'rgba(255, 255, 255, 0.1)',
                'progress_chunk': 'qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #60CDFF, stop:1 #4CC2FF)'
            },
            'light': {
                'primary': '#0067C0',       # Акцентный синий Windows 11
                'secondary': '#0078D4',     # Светло-синий
                'background': '#F9F9F9',    # Светло-серый фон
                'surface': '#FFFFFF',       # Белые карточки/контейнеры
                'surface_hover': '#F5F5F5', # При наведении
                'mica': 'rgba(249, 249, 249, 0.85)', # Эффект Mica 
                'text': '#202020',          # Темно-серый текст
                'text_secondary': 'rgba(0, 0, 0, 0.6)',
                'button': '#0067C0',        # Кнопки в цвет акцента
                'button_hover': '#0078D4',  # Светлее при наведении
                'button_pressed': '#005494', # Темнее при нажатии
                'disabled_bg': 'rgba(0, 0, 0, 0.05)',
                'disabled_fg': 'rgba(0, 0, 0, 0.4)',
                'error': '#D13438',         # Красный для ошибок
                'error_hover': '#E25A5E',   # Светлее красный
                'border': 'rgba(0, 0, 0, 0.1)', # Более заметная рамка
                'progress_bg': 'rgba(0, 103, 192, 0.1)',
                'progress_chunk': 'qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #0078D4, stop:1 #0067C0)'
            }
        }

        # Определяем тип ускорения для отображения
        if torch.cuda.is_available():
            self.acceleration_type = "CUDA"
        else:
            self.acceleration_type = "CPU"

        self.is_dark_theme = True # По умолчанию темная
        self.processor = None
        self.current_temp_dir = None # Храним путь к временной папке

        self.init_ui()
        self.apply_theme()

    def init_ui(self):
        # Устанавливаем шрифт в стиле Windows 11
        font = self.font()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.setFont(font)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)  # Больше пространства между элементами
        main_layout.setContentsMargins(22, 22, 22, 22)  # Больше отступов

        # --- Верхняя панель (Инфо + Управление) ---
        top_frame = StyledFrame()
        top_layout = top_frame.layout # Используем существующий layout фрейма
        top_layout.setSpacing(8)

        info_layout = QHBoxLayout()
        self.acceleration_label = QLabel(f"Ускорение: {self.acceleration_type}")
        self.acceleration_label.setObjectName("accelerationLabel") # Для стилизации
        info_layout.addStretch()
        info_layout.addWidget(self.acceleration_label)
        top_layout.addLayout(info_layout)

        buttons_layout = QHBoxLayout()
        self.theme_button = QPushButton()
        self.theme_button.setToolTip("Сменить тему оформления")
        self.theme_button.setFixedSize(140, 32) # Чуть шире
        self.theme_button.clicked.connect(self.toggle_theme)
        self.update_theme_button() # Устанавливаем иконку и текст

        buttons_layout.addStretch()
        buttons_layout.addWidget(self.theme_button)
        top_layout.addLayout(buttons_layout)

        main_layout.addWidget(top_frame)

        # --- Основная панель (Выбор файла + Язык) ---
        main_frame = StyledFrame()
        main_frame_layout = main_frame.layout
        main_frame_layout.setSpacing(10)

        language_layout = QHBoxLayout()
        self.language_label = QLabel("Язык перевода:")
        self.language_label.setObjectName("languageLabel")
        self.target_language_combo = QComboBox()
        self.target_language_combo.setObjectName("languageCombo")
        # Сортируем языки для удобства
        self.language_codes = {
            "Русский": "ru", "English": "en", "Deutsch": "de", "Español": "es",
            "Français": "fr", "Italiano": "it", "Português": "pt", "Polski": "pl",
            "Türkçe": "tr", "Magyar": "hu", "Čeština": "cs", "Nederlands": "nl",
            "日本語": "ja", "中文": "zh-cn", "한국어": "ko", "हिन्दी": "hi", "العربية": "ar"
        }
        sorted_languages = sorted(self.language_codes.keys())
        self.target_language_combo.addItems(sorted_languages)
        # Устанавливаем русский по умолчанию, если есть
        if "Русский" in sorted_languages:
             self.target_language_combo.setCurrentText("Русский")

        language_layout.addWidget(self.language_label)
        language_layout.addWidget(self.target_language_combo)
        main_frame_layout.addLayout(language_layout)
        
        # Добавляем опцию для Wav2Lip
        wav2lip_layout = QHBoxLayout()
        
        # Чекбокс Wav2Lip
        self.wav2lip_checkbox = QCheckBox("Включить синхронизацию губ с аудио (Wav2Lip)")
        self.wav2lip_checkbox.setChecked(False)  # По умолчанию выключено
        self.wav2lip_checkbox.setEnabled(WAV2LIP_AVAILABLE)  # Включить только если Wav2Lip доступен
        if not WAV2LIP_AVAILABLE:
            self.wav2lip_checkbox.setToolTip("Wav2Lip недоступен. Проверьте наличие необходимых файлов.")
        else:
            self.wav2lip_checkbox.setToolTip("Включает синхронизацию движения губ с произносимым текстом. Требует GPU и значительно увеличивает время обработки.")
        
        # Выпадающий список для resize_factor
        self.resize_factor_label = QLabel("Масштаб лица:")
        self.resize_factor_label.setObjectName("resizeFactorLabel")
        self.resize_factor_combo = QComboBox()
        self.resize_factor_combo.setObjectName("resizeFactorCombo")
        self.resize_factor_combo.addItems(["1", "2", "4", "8", "16"])
        self.resize_factor_combo.setCurrentText("4")  # По умолчанию 4
        self.resize_factor_combo.setToolTip("Влияет на скорость обработки и качество. Меньшие значения - лучше качество, но медленнее.")
        
        # Включаем/отключаем контролы для resize_factor в зависимости от статуса чекбокса Wav2Lip
        self.resize_factor_label.setEnabled(self.wav2lip_checkbox.isChecked() and WAV2LIP_AVAILABLE)
        self.resize_factor_combo.setEnabled(self.wav2lip_checkbox.isChecked() and WAV2LIP_AVAILABLE)
        
        # Подключаем обработчик события изменения чекбокса
        self.wav2lip_checkbox.stateChanged.connect(self.on_wav2lip_checkbox_changed)
        
        wav2lip_layout.addWidget(self.wav2lip_checkbox)
        wav2lip_layout.addStretch()
        wav2lip_layout.addWidget(self.resize_factor_label)
        wav2lip_layout.addWidget(self.resize_factor_combo)
        
        main_frame_layout.addLayout(wav2lip_layout)

        self.select_button = QPushButton("Выбрать видео для дубляжа")
        # Не используем стандартную иконку, используем Unicode символ
        # self.select_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.select_button.setText("\u0020\u2b07\u0020\u0020Выбрать видео для дубляжа") # Символ стрелки вниз с пробелами
        self.select_button.clicked.connect(self.select_file)
        main_frame_layout.addWidget(self.select_button)

        self.file_label = QLabel("Видеофайл не выбран")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setObjectName("fileLabel")
        main_frame_layout.addWidget(self.file_label)

        # --- Слайдеры регулировки громкости ---
        volume_layout = QHBoxLayout()
        
        # Слайдер громкости речи
        speech_volume_layout = QVBoxLayout()
        speech_volume_label = QLabel("Громкость речи:")
        speech_volume_label.setObjectName("volumeLabel")
        self.speech_volume_slider = QSlider(Qt.Horizontal)
        self.speech_volume_slider.setMinimum(0)
        self.speech_volume_slider.setMaximum(100)
        self.speech_volume_slider.setValue(80)  # По умолчанию 80%
        self.speech_volume_slider.setToolTip("Регулировка громкости речи")
        self.speech_volume_value = QLabel("80%")
        self.speech_volume_value.setAlignment(Qt.AlignCenter)
        self.speech_volume_slider.valueChanged.connect(self.on_speech_volume_changed)
        speech_volume_layout.addWidget(speech_volume_label)
        speech_volume_layout.addWidget(self.speech_volume_slider)
        speech_volume_layout.addWidget(self.speech_volume_value)
        
        # Слайдер громкости фона
        background_volume_layout = QVBoxLayout()
        background_volume_label = QLabel("Громкость фона:")
        background_volume_label.setObjectName("volumeLabel")
        self.background_volume_slider = QSlider(Qt.Horizontal)
        self.background_volume_slider.setMinimum(0)
        self.background_volume_slider.setMaximum(100)
        self.background_volume_slider.setValue(20)  # По умолчанию 20%
        self.background_volume_slider.setToolTip("Регулировка громкости фонового шума")
        self.background_volume_value = QLabel("20%")
        self.background_volume_value.setAlignment(Qt.AlignCenter)
        self.background_volume_slider.valueChanged.connect(self.on_background_volume_changed)
        background_volume_layout.addWidget(background_volume_label)
        background_volume_layout.addWidget(self.background_volume_slider)
        background_volume_layout.addWidget(self.background_volume_value)
        
        volume_layout.addLayout(speech_volume_layout)
        volume_layout.addLayout(background_volume_layout)
        
        main_frame_layout.addLayout(volume_layout)

        main_layout.addWidget(main_frame)

        # --- Панель прогресса и статуса ---
        progress_frame = StyledFrame()
        progress_layout = progress_frame.layout
        progress_layout.setSpacing(8)

        self.progress_bar = AnimatedProgressBar()
        progress_layout.addWidget(self.progress_bar)

        status_layout = QHBoxLayout()
        self.time_label = QLabel("Время: --:--")
        self.time_label.setObjectName("timeLabel")
        self.cancel_button = QPushButton("Отмена")
        # Не используем стандартную иконку, используем Unicode символ
        # self.cancel_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.cancel_button.setText("\u0020\u2715\u0020\u0020Отмена") # Символ крестика с пробелами
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        status_layout.addWidget(self.time_label)
        status_layout.addStretch()
        status_layout.addWidget(self.cancel_button)
        progress_layout.addLayout(status_layout)

        self.status_label = QLabel("Готов к работе")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True) # Перенос слов для длинных сообщений
        self.status_label.setObjectName("statusLabel")
        progress_layout.addWidget(self.status_label)

        main_layout.addWidget(progress_frame)

        # Таймер для оценки времени
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_estimate)
        self.start_time = None
        self.processing = False

    def apply_theme(self):
        """Применение выбранной темы и стилизация кастомных элементов в стиле Windows 11"""
        theme_name = 'dark_blue.xml' if self.is_dark_theme else 'light_blue.xml'
        try:
            apply_stylesheet(self, theme=theme_name, invert_secondary=True)
            logger.info(f"Применена тема: {theme_name}")
        except Exception as e:
            logger.error(f"Ошибка применения темы {theme_name}: {e}")
            # Применяем базовый стиль Qt, если тема не удалась
            QApplication.setStyle(QStyleFactory.create('Fusion'))

        # Получаем текущую цветовую схему
        colors = self.color_schemes['dark' if self.is_dark_theme else 'light']
        text_color = colors['text']
        text_secondary_color = colors['text_secondary']
        button_color = colors['button']
        button_hover_color = colors['button_hover']
        button_pressed_color = colors['button_pressed']
        disabled_bg = colors['disabled_bg']
        disabled_fg = colors['disabled_fg']
        surface_color = colors['surface']
        surface_hover_color = colors['surface_hover']
        mica_color = colors['mica']
        border_color = colors['border']
        progress_bg = colors['progress_bg']
        progress_chunk = colors['progress_chunk']
        
        # Windows 11 использует более закругленные углы и мягкие тени
        border_radius = "12px"
        
        # Глобальный стиль окна
        self.setStyleSheet(f"""
            QMainWindow {{ 
                background-color: {colors['background']}; 
                font-family: 'Segoe UI', sans-serif;
            }}
            
            QLabel {{ 
                font-family: 'Segoe UI', sans-serif;
                color: {text_color};
            }}
            
            QComboBox, QCheckBox {{ 
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }}
            
            QPushButton {{ 
                font-family: 'Segoe UI', sans-serif;
                font-weight: 400;
                font-size: 10pt;
            }}
        """)

        # Обновляем стили для кастомных элементов
        for frame in self.findChildren(StyledFrame):
             frame.setStyleSheet(f"""
                 QFrame {{
                     background-color: {mica_color};
                     border-radius: {border_radius};
                     border: 1px solid {border_color};
                 }}
             """)

        # Прогресс-бар в стиле Windows 11
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 8px;
                text-align: center;
                background-color: {progress_bg};
                height: 8px;
                max-height: 8px;
                color: transparent;
            }}
            QProgressBar::chunk {{
                border-radius: 4px;
                background-color: {progress_chunk};
            }}
        """)

        # Стиль для ComboBox (выпадающий список языков и resize_factor)
        for combo in [self.target_language_combo, self.resize_factor_combo]:
            combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {surface_color};
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: {border_radius};
                    padding: 6px 12px;
                    min-width: 150px;
                    font-size: 10pt;
                }}
                QComboBox:hover {{
                    background-color: {surface_hover_color};
                    border: 1px solid {colors['primary']};
                }}
                QComboBox::drop-down {{
                    border: none;
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 20px;
                    padding-right: 10px;
                }}
                QComboBox::down-arrow {{
                    color: {text_color};
                }}
                /* Явно указываем стиль для выпадающего списка */
                QComboBox QAbstractItemView {{
                    background-color: {surface_color};
                    color: {text_color};
                    selection-background-color: {colors['primary']};
                    selection-color: white;
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    padding: 4px;
                    outline: 0px;
                }}
                /* Дополнительный селектор для QListView, который фактически используется */
                QComboBox QListView {{
                    background-color: {surface_color};
                    color: {text_color};
                    border: 1px solid {border_color};
                }}
                QComboBox QAbstractItemView::item {{
                    min-height: 30px;
                    padding: 6px 10px;
                    border-radius: 4px;
                    color: {text_color}; /* Явно устанавливаем цвет текста элемента */
                }}
                QComboBox QAbstractItemView::item:selected {{
                    background-color: {colors['primary']};
                    color: white;
                }}
                QComboBox QAbstractItemView::item:hover {{
                    background-color: {surface_hover_color};
                    color: {text_color};
                }}
            """)
        
        # Глобальные стили для выпадающих списков
        QApplication.instance().setStyleSheet(f"""
            QComboBox QAbstractItemView {{
                background-color: {surface_color};
                color: {text_color};
                selection-background-color: {colors['primary']};
                selection-color: white;
            }}
        """)

        # Стиль для чекбокса в стиле Windows 11
        self.wav2lip_checkbox.setStyleSheet(f"""
            QCheckBox {{
                spacing: 8px;
                color: {text_color};
                font-size: 10pt;
            }}
            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid {border_color};
                background-color: {surface_color};
            }}
            QCheckBox::indicator:unchecked:hover {{
                background-color: {surface_hover_color};
                border: 1px solid {colors['primary']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['primary']};
                border: 1px solid {colors['primary']};
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>');
            }}
            QCheckBox::indicator:checked:hover {{
                background-color: {button_hover_color};
            }}
            QCheckBox:disabled {{
                color: {disabled_fg};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {disabled_bg};
                border: 1px solid {disabled_bg};
            }}
        """)

        # Общий стиль для кнопок в стиле Windows 11
        button_style = f"""
            QPushButton {{
                background-color: {button_color};
                color: white;
                border: none;
                border-radius: {border_radius};
                padding: 10px 20px;
                font-size: 10pt;
                font-weight: 400;
                min-height: 38px;
            }}
            QPushButton:hover {{
                background-color: {button_hover_color};
            }}
            QPushButton:pressed {{
                background-color: {button_pressed_color};
            }}
            QPushButton:disabled {{
                background-color: {disabled_bg};
                color: {disabled_fg};
            }}
        """
        
        self.select_button.setStyleSheet(button_style)
        
        # Кнопка отмены
        self.cancel_button.setStyleSheet(f"""
            {button_style}
            QPushButton {{ 
                background-color: {colors['error']};
                padding: 8px 16px;
                min-height: 34px;
            }}
            QPushButton:hover {{ 
                background-color: {colors['error_hover']};
            }}
        """)

        # Стиль для кнопки темы в стиле Windows 11
        theme_button_bg = 'rgba(255, 255, 255, 0.08)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.05)'
        theme_button_hover_bg = 'rgba(255, 255, 255, 0.12)' if self.is_dark_theme else 'rgba(0, 0, 0, 0.08)'
        self.theme_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {theme_button_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: {border_radius};
                padding: 8px 12px;
                font-weight: 400;
                text-align: center;
                font-size: 10pt;
                min-height: 34px;
            }}
            QPushButton:hover {{
                background-color: {theme_button_hover_bg};
                border: 1px solid {colors['primary']};
            }}
        """)

        # Стили текстовых меток
        label_style = f"color: {text_color}; font-size: 10pt; font-weight: 400;"
        secondary_label_style = f"color: {text_secondary_color}; font-size: 9pt; font-weight: 400;"

        self.language_label.setStyleSheet(label_style)
        self.resize_factor_label.setStyleSheet(label_style)
        self.file_label.setStyleSheet(secondary_label_style + "font-style: italic;")
        self.time_label.setStyleSheet(secondary_label_style)
        self.status_label.setStyleSheet(secondary_label_style + "font-style: italic;")
        self.acceleration_label.setStyleSheet(f"color: {colors['primary']}; font-size: 10pt; font-weight: 500;")


    def show_with_animation(self):
        """Анимированное появление окна"""
        self.opacity_effect = QGraphicsOpacityEffect(self) # Указываем родителя
        self.setGraphicsEffect(self.opacity_effect)

        self.fade_in = QPropertyAnimation(self.opacity_effect, b"opacity", self) # Указываем родителя
        self.fade_in.setStartValue(0.0)
        self.fade_in.setEndValue(1.0)
        self.fade_in.setDuration(400) # Чуть быстрее
        self.fade_in.setEasingCurve(QEasingCurve.InOutQuad) # Более плавно

        # Показываем окно перед анимацией, но делаем его прозрачным
        super().show() # Используем show() родительского класса
        self.opacity_effect.setOpacity(0.0) # Устанавливаем начальную прозрачность
        self.fade_in.start()


    def toggle_theme(self):
        """Переключение между темной и светлой темой"""
        self.is_dark_theme = not self.is_dark_theme
        logger.info(f"Переключение темы на {'темную' if self.is_dark_theme else 'светлую'}")
        self.update_theme_button()
        self.apply_theme()

    def update_theme_button(self):
        """Обновление внешнего вида кнопки темы"""
        if self.is_dark_theme:
            # Не используем стандартную иконку, используем Unicode символ солнца
            self.theme_button.setText("\u0020\u2600\u0020\u0020Светлая тема") # Символ солнца с пробелами
            # self.theme_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView)) # Иконка светлой темы
        else:
            # Не используем стандартную иконку, используем Unicode символ луны
            self.theme_button.setText("\u0020\u263D\u0020\u0020Тёмная тема") # Символ луны с пробелами
            # self.theme_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView)) # Иконка темной темы

    def update_time_estimate(self):
        """Обновление оценки оставшегося времени"""
        if not self.processing or self.start_time is None:
            self.time_label.setText("Время: --:--")
            return

        progress = self.progress_bar.value()
        if progress > 1 and progress < 100: # Не считаем для 0, 1 и 100
            try:
                # Используем QTime для более точного расчета времени
                elapsed_msecs = self.start_time.elapsed()
                elapsed_secs = elapsed_msecs / 1000.0
                total_estimated_secs = (elapsed_secs * 100.0) / progress
                remaining_secs = max(0, total_estimated_secs - elapsed_secs) # Убедимся, что не отрицательное

                minutes = int(remaining_secs // 60)
                seconds = int(remaining_secs % 60)
                self.time_label.setText(f"Время: ~{minutes:02d}:{seconds:02d}")
            except ZeroDivisionError:
                self.time_label.setText("Время: --:--")
            except Exception as e:
                 logger.error(f"Ошибка расчета времени: {e}")
                 self.time_label.setText("Время: Ошибка")
        elif progress == 100:
             self.time_label.setText("Время: 00:00")
        else:
             self.time_label.setText("Время: Расчет...")


    def cancel_processing(self):
        """Отмена текущей обработки"""
        if hasattr(self, 'processor') and self.processor and self.processor.isRunning():
            logger.info("Отмена обработки запрошена пользователем")
            self.status_label.setText("Отмена обработки...")
            self.processor.request_cancellation()
            self.cancel_button.setEnabled(False)
            
            # Поток должен обработать запрос на отмену и завершиться
            # Любая очистка будет выполнена в обработчике finished

    def reset_ui_state(self, processing_active=False):
        """Сброс состояния UI в зависимости от режима (обработка активна/неактивна)"""
        self.select_button.setEnabled(not processing_active)
        self.target_language_combo.setEnabled(not processing_active)
        self.theme_button.setEnabled(not processing_active) # Блокируем смену темы
        self.wav2lip_checkbox.setEnabled(not processing_active and WAV2LIP_AVAILABLE)
        self.resize_factor_label.setEnabled(not processing_active and self.wav2lip_checkbox.isChecked() and WAV2LIP_AVAILABLE)
        self.resize_factor_combo.setEnabled(not processing_active and self.wav2lip_checkbox.isChecked() and WAV2LIP_AVAILABLE)
        self.speech_volume_slider.setEnabled(not processing_active)
        self.background_volume_slider.setEnabled(not processing_active)
        self.cancel_button.setEnabled(processing_active)
        
        if not processing_active:
            self.progress_bar.setValue(0)
            if self.timer.isActive():
                self.timer.stop()
            self.time_label.setText("Время: --:--")
            self.processing = False

    def start_processing(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.status_label.setText("Ошибка: видеофайл не выбран или не существует")
            return

        # Получаем выбранный язык
        target_language = self.language_codes[self.target_language_combo.currentText()]
        logger.info(f"Выбран язык перевода: {target_language}")
        
        # Получаем состояние Wav2Lip
        use_wav2lip = self.wav2lip_checkbox.isChecked() and WAV2LIP_AVAILABLE
        resize_factor = int(self.resize_factor_combo.currentText())
        
        # Получаем значения громкости
        speech_gain = self.speech_volume_slider.value() / 100.0
        background_gain = self.background_volume_slider.value() / 100.0

        # Создаем и настраиваем видеопроцессор
        self.processor = VideoProcessor(
            self.video_path, 
            target_language=target_language,
            use_wav2lip=use_wav2lip,
            resize_factor=resize_factor
        )
        
        # Устанавливаем параметры громкости
        self.processor.speech_gain = speech_gain
        self.processor.background_gain = background_gain
        
        # Обновляем UI
        self.status_label.setText("Инициализация...")
        self.progress_bar.setValue(0)
        self.select_button.setEnabled(False)
        self.target_language_combo.setEnabled(False)
        self.wav2lip_checkbox.setEnabled(False)
        self.resize_factor_combo.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Запускаем таймер
        self.start_time = QTime.currentTime()
        self.timer.start(1000)  # Обновление каждую секунду
        self.processing = True
        
        # Подключаем сигналы
        self.processor.status.connect(self.status_label.setText)
        self.processor.progress.connect(self.progress_bar.setValue)
        self.processor.finished.connect(self.processing_finished)
        self.processor.error.connect(self.processing_error)
        self.processor.acceleration_changed.connect(self.update_acceleration_label)
        
        # Запускаем поток
        self.processor.start()


    def processing_finished(self, output_file_path):
        """Обработчик успешного завершения обработки видео"""
        logger.info(f"Сигнал finished получен. Результат: {output_file_path}")
        self.status_label.setText(f"Готово! Результат сохранен: {output_file_path}")
        self.progress_bar.setValue(100)
        self.reset_ui_state(processing_active=False) # Разблокируем UI
        QMessageBox.information(self, "Успех", f"Дубляж видео завершен!\nФайл сохранен как:\n{output_file_path}")
        self.cleanup_temp_dir() # Очищаем временную папку после успеха


    def processing_error(self, error_message):
        """Обработчик ошибки во время обработки видео"""
        logger.error(f"Сигнал error получен: {error_message}")
        self.reset_ui_state(processing_active=False) # Разблокируем UI

        if error_message == "Отменено":
             self.status_label.setText("Обработка отменена пользователем.")
             # QMessageBox.warning(self, "Отмена", "Обработка была отменена.")
        else:
             self.status_label.setText(f"Ошибка обработки: {error_message}")
             QMessageBox.critical(self, "Ошибка обработки", f"Во время обработки произошла ошибка:\n\n{error_message}\n\nПодробности смотрите в логах.")

        self.cleanup_temp_dir() # Очищаем временную папку и после ошибки/отмены


    def cleanup_temp_dir(self):
        """Удаление временной папки"""
        if self.current_temp_dir and os.path.exists(self.current_temp_dir):
            reply = QMessageBox.question(self, "Очистка",
                                         f"Удалить временную папку с промежуточными файлами?\n({self.current_temp_dir})",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                try:
                    shutil.rmtree(self.current_temp_dir)
                    logger.info(f"Временная папка {self.current_temp_dir} удалена.")
                    self.current_temp_dir = None # Сбрасываем путь
                except Exception as e:
                    logger.error(f"Не удалось удалить временную папку {self.current_temp_dir}: {e}")
                    QMessageBox.warning(self, "Ошибка очистки", f"Не удалось удалить временную папку:\n{e}")
            else:
                 logger.info(f"Очистка временной папки {self.current_temp_dir} пропущена пользователем.")
        else:
             logger.info("Нет временной папки для очистки.")


    def update_acceleration_label(self, acceleration_type):
        """Обновление метки с информацией об ускорении"""
        self.acceleration_type = acceleration_type
        self.acceleration_label.setText(f"Ускорение: {acceleration_type}")
        logger.info(f"Тип ускорения обновлен на: {acceleration_type}")

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.processing:
            reply = QMessageBox.question(self, 'Подтверждение выхода',
                                         "Идет процесс обработки видео. Вы уверены, что хотите выйти?\nОбработка будет прервана.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                logger.warning("Приложение закрывается во время обработки. Запрос отмены.")
                if self.processor:
                    self.processor.request_cancellation()
                    # Даем потоку немного времени на завершение перед выходом
                    # self.processor.wait(1000) # Может заблокировать GUI, лучше не ждать
                event.accept() # Разрешаем закрытие
            else:
                event.ignore() # Отменяем закрытие
        else:
             # Спросим про очистку если папка осталась
             if self.current_temp_dir and os.path.exists(self.current_temp_dir):
                 reply = QMessageBox.question(self, "Очистка перед выходом",
                                             f"Осталась временная папка обработки. Удалить ее перед выходом?\n({self.current_temp_dir})",
                                             QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
                 if reply == QMessageBox.Yes:
                      try:
                           shutil.rmtree(self.current_temp_dir)
                           logger.info(f"Временная папка {self.current_temp_dir} удалена перед выходом.")
                      except Exception as e:
                           logger.error(f"Не удалось удалить временную папку {self.current_temp_dir} перед выходом: {e}")
                           QMessageBox.warning(self, "Ошибка очистки", f"Не удалось удалить временную папку:\n{e}")
                           # Все равно выходим
                      event.accept()
                 elif reply == QMessageBox.No:
                      logger.info("Очистка временной папки пропущена при выходе.")
                      event.accept()
                 else: # Cancel
                      event.ignore() # Отменяем выход
             else:
                  event.accept() # Закрываем без вопросов

    def select_file(self):
        if self.processing: return # Нельзя во время обработки

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл для дубляжа",
            "", # Начальная директория (пусто - последняя использованная)
            "Видео файлы (*.mp4 *.avi *.mkv *.mov *.wmv);;Все файлы (*.*)" # Фильтры файлов
        )
        if file_path:
            self.video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            logger.info(f"Выбран видеофайл: {file_path}")
            # Автоматически запускаем обработку после выбора файла
            self.start_processing()
        else:
            logger.info("Выбор файла отменен.")

    def on_wav2lip_checkbox_changed(self, state):
        """Обработчик изменения состояния чекбокса Wav2Lip"""
        enabled = (state == Qt.Checked and WAV2LIP_AVAILABLE)
        self.resize_factor_label.setEnabled(enabled)
        self.resize_factor_combo.setEnabled(enabled)

    def on_speech_volume_changed(self, value):
        """Обработчик изменения громкости речи"""
        self.speech_volume_value.setText(f"{value}%")
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.speech_gain = value / 100.0
    
    def on_background_volume_changed(self, value):
        """Обработчик изменения громкости фона"""
        self.background_volume_value.setText(f"{value}%")
        if hasattr(self, 'processor') and self.processor is not None:
            self.processor.background_gain = value / 100.0


# --- Точка входа ---
if __name__ == "__main__":
    # Установка переменной окружения для корректной работы matplotlib/librosa в PyInstaller
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

    # Создание QApplication
    app = QApplication(sys.argv)

    # Проверка доступности тем qt-material
    available_themes = list_themes()
    logger.info(f"Доступные темы qt-material: {available_themes}")
    if 'dark_blue.xml' not in available_themes or 'light_blue.xml' not in available_themes:
        logger.warning("Стандартные темы ('dark_blue.xml', 'light_blue.xml') не найдены! Стили могут отображаться некорректно.")
        # Можно выбрать другую доступную тему как fallback
        # fallback_theme = available_themes[0] if available_themes else None

    # Создание и показ главного окна
    window = MainWindow()
    # window.show() # Обычный показ
    window.show_with_animation() # Показ с анимацией

    # Запуск главного цикла приложения
    sys.exit(app.exec_())
