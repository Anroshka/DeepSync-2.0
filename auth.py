import os
import json
import hashlib
import requests
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QMessageBox, QTabWidget)
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView

# Путь к файлу с данными пользователей
USERS_FILE = "users.json"

class AuthManager:
    """Класс для управления авторизацией пользователей"""
    
    def __init__(self):
        self.current_user = None
        self._load_users()
    
    def _load_users(self):
        """Загрузка данных пользователей из файла"""
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                self.users = json.load(f)
        else:
            # Создаем файл с тестовым пользователем, если его нет
            self.users = {
                "admin": {
                    "password_hash": self._hash_password("admin123"),
                    "name": "Администратор"
                }
            }
            self._save_users()
    
    def _save_users(self):
        """Сохранение данных пользователей в файл"""
        with open(USERS_FILE, 'w') as f:
            json.dump(self.users, f, indent=4)
    
    def _hash_password(self, password):
        """Хеширование пароля"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login(self, username, password):
        """Авторизация пользователя по логину и паролю"""
        # Проверка на пустой логин
        if not username:
            return False, "Логин не может быть пустым"
        
        # Проверка на пустой пароль
        if not password:
            return False, "Пароль не может быть пустым"
        
        # Проверка существования пользователя
        if username not in self.users:
            return False, "Пользователь не найден"
        
        # Проверка пароля
        password_hash = self._hash_password(password)
        if password_hash != self.users[username]["password_hash"]:
            return False, "Неверный пароль"
        
        # Успешная авторизация
        self.current_user = username
        return True, "Авторизация успешна"
    
    def register(self, username, password, name):
        """Регистрация нового пользователя"""
        if username in self.users:
            return False, "Пользователь с таким логином уже существует"
        
        if not username or not password:
            return False, "Логин и пароль не могут быть пустыми"
        
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "name": name
        }
        self._save_users()
        return True, "Регистрация успешна"
    
    def logout(self):
        """Выход из системы"""
        self.current_user = None
        return True, "Выход выполнен"
    
    def get_current_user_info(self):
        """Получение информации о текущем пользователе"""
        if not self.current_user:
            return None
        return {
            "username": self.current_user,
            "name": self.users[self.current_user]["name"]
        }
    
    def login_vk(self, token, user_id):
        """Авторизация через VK ID"""
        # В реальном приложении здесь был бы код для проверки токена через API VK
        # Для демонстрации просто создаем/обновляем пользователя
        username = f"vk_{user_id}"
        
        if username not in self.users:
            self.users[username] = {
                "password_hash": "",  # Пустой хеш, так как авторизация через VK
                "name": f"VK User {user_id}",
                "vk_id": user_id
            }
            self._save_users()
        
        self.current_user = username
        return True, "Авторизация через VK успешна"
    
    def login_yandex(self, token, user_info):
        """Авторизация через Яндекс ID"""
        # В реальном приложении здесь был бы код для проверки токена через API Яндекса
        # Для демонстрации просто создаем/обновляем пользователя
        username = f"yandex_{user_info['id']}"
        
        if username not in self.users:
            self.users[username] = {
                "password_hash": "",  # Пустой хеш, так как авторизация через Яндекс
                "name": user_info.get('display_name', f"Yandex User {user_info['id']}"),
                "yandex_id": user_info['id']
            }
            self._save_users()
        
        self.current_user = username
        return True, "Авторизация через Яндекс успешна"


class LoginDialog(QDialog):
    """Диалог авторизации"""
    
    def __init__(self, auth_manager, parent=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.setWindowTitle("Авторизация - DeepSynch")
        self.setMinimumWidth(400)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка интерфейса"""
        layout = QVBoxLayout(self)
        
        # Создаем вкладки для разных способов авторизации
        self.tab_widget = QTabWidget()
        
        # Вкладка для обычной авторизации
        self.login_tab = QWidget()
        self.setup_login_tab()
        self.tab_widget.addTab(self.login_tab, "Логин/Пароль")
        
        # Вкладка для авторизации через VK
        self.vk_tab = QWidget()
        self.setup_vk_tab()
        self.tab_widget.addTab(self.vk_tab, "VK ID")
        
        # Вкладка для авторизации через Яндекс
        self.yandex_tab = QWidget()
        self.setup_yandex_tab()
        self.tab_widget.addTab(self.yandex_tab, "Яндекс ID")
        
        layout.addWidget(self.tab_widget)
    
    def setup_login_tab(self):
        """Настройка вкладки обычной авторизации"""
        layout = QVBoxLayout(self.login_tab)
        
        # Поле для ввода логина
        login_layout = QHBoxLayout()
        login_label = QLabel("Логин:")
        self.login_edit = QLineEdit()
        login_layout.addWidget(login_label)
        login_layout.addWidget(self.login_edit)
        layout.addLayout(login_layout)
        
        # Поле для ввода пароля
        password_layout = QHBoxLayout()
        password_label = QLabel("Пароль:")
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_edit)
        layout.addLayout(password_layout)
        
        # Кнопки
        button_layout = QHBoxLayout()
        self.login_button = QPushButton("Войти")
        self.login_button.clicked.connect(self.login)
        self.register_button = QPushButton("Регистрация")
        self.register_button.clicked.connect(self.register)
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.register_button)
        layout.addLayout(button_layout)
    
    def setup_vk_tab(self):
        """Настройка вкладки авторизации через VK"""
        layout = QVBoxLayout(self.vk_tab)
        
        # В реальном приложении здесь был бы WebView для авторизации через VK OAuth
        self.vk_web_view = QWebEngineView()
        layout.addWidget(self.vk_web_view)
        
        # Для демонстрации добавим кнопку эмуляции авторизации
        self.vk_login_button = QPushButton("Войти через VK")
        self.vk_login_button.clicked.connect(self.login_vk)
        layout.addWidget(self.vk_login_button)
    
    def setup_yandex_tab(self):
        """Настройка вкладки авторизации через Яндекс"""
        layout = QVBoxLayout(self.yandex_tab)
        
        # В реальном приложении здесь был бы WebView для авторизации через Яндекс OAuth
        self.yandex_web_view = QWebEngineView()
        layout.addWidget(self.yandex_web_view)
        
        # Для демонстрации добавим кнопку эмуляции авторизации
        self.yandex_login_button = QPushButton("Войти через Яндекс")
        self.yandex_login_button.clicked.connect(self.login_yandex)
        layout.addWidget(self.yandex_login_button)
    
    def login(self):
        """Обработка авторизации по логину и паролю"""
        username = self.login_edit.text()
        password = self.password_edit.text()
        
        success, message = self.auth_manager.login(username, password)
        
        if success:
            self.accept()  # Закрываем диалог с кодом успеха
        else:
            QMessageBox.warning(self, "Ошибка авторизации", message)
    
    def register(self):
        """Обработка регистрации нового пользователя"""
        username = self.login_edit.text()
        password = self.password_edit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Ошибка", "Логин и пароль не могут быть пустыми")
            return
        
        # В реальном приложении здесь был бы дополнительный диалог для ввода имени
        name = username  # Для простоты используем логин как имя
        
        success, message = self.auth_manager.register(username, password, name)
        
        if success:
            QMessageBox.information(self, "Регистрация", message)
            # Автоматически авторизуем пользователя после регистрации
            self.auth_manager.login(username, password)
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка регистрации", message)
    
    def login_vk(self):
        """Эмуляция авторизации через VK"""
        # В реальном приложении здесь был бы код для получения токена через OAuth
        # Для демонстрации просто эмулируем успешную авторизацию
        token = "demo_vk_token"
        user_id = "12345"
        
        success, message = self.auth_manager.login_vk(token, user_id)
        
        if success:
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка авторизации", message)
    
    def login_yandex(self):
        """Эмуляция авторизации через Яндекс"""
        # В реальном приложении здесь был бы код для получения токена через OAuth
        # Для демонстрации просто эмулируем успешную авторизацию
        token = "demo_yandex_token"
        user_info = {
            "id": "67890",
            "display_name": "Яндекс Пользователь"
        }
        
        success, message = self.auth_manager.login_yandex(token, user_info)
        
        if success:
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка авторизации", message) 