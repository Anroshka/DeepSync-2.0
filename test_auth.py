import unittest
import os
import json
import shutil
import auth
from auth import AuthManager

class TestAuthManager(unittest.TestCase):
    """Тесты для класса AuthManager"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        # Создаем временный файл для тестов
        self.test_users_file = "test_users.json"
        
        # Сохраняем оригинальное значение USERS_FILE
        self.original_users_file = auth.USERS_FILE
        auth.USERS_FILE = self.test_users_file
        
        # Создаем тестовых пользователей
        test_users = {
            "testuser": {
                "password_hash": AuthManager()._hash_password("password123"),
                "name": "Test User"
            }
        }
        
        with open(self.test_users_file, 'w') as f:
            json.dump(test_users, f)
        
        # Создаем экземпляр AuthManager для тестов
        self.auth_manager = AuthManager()
    
    def tearDown(self):
        """Очистка после каждого теста"""
        # Восстанавливаем оригинальное значение USERS_FILE
        auth.USERS_FILE = self.original_users_file
        
        # Удаляем временный файл
        if os.path.exists(self.test_users_file):
            os.remove(self.test_users_file)
    
    def test_login_success(self):
        """Тест 1: Успешная авторизация с корректными логином и паролем"""
        success, message = self.auth_manager.login("testuser", "password123")
        self.assertTrue(success)
        self.assertEqual(self.auth_manager.current_user, "testuser")
    
    def test_login_nonexistent_user(self):
        """Тест 2: Попытка авторизации с несуществующим логином"""
        success, message = self.auth_manager.login("nonexistent", "password123")
        self.assertFalse(success)
        self.assertEqual(message, "Пользователь не найден")
        self.assertIsNone(self.auth_manager.current_user)
    
    def test_login_wrong_password(self):
        """Тест 3: Попытка авторизации с неправильным паролем"""
        success, message = self.auth_manager.login("testuser", "wrongpassword")
        self.assertFalse(success)
        self.assertEqual(message, "Неверный пароль")
        self.assertIsNone(self.auth_manager.current_user)
    
    def test_login_empty_password(self):
        """Тест 4: Попытка авторизации с пустым паролем"""
        success, message = self.auth_manager.login("testuser", "")
        self.assertFalse(success)
        self.assertEqual(message, "Пароль не может быть пустым")
        self.assertIsNone(self.auth_manager.current_user)
    
    def test_login_empty_username(self):
        """Тест 5: Попытка авторизации с пустым логином"""
        success, message = self.auth_manager.login("", "password123")
        self.assertFalse(success)
        self.assertEqual(message, "Логин не может быть пустым")
        self.assertIsNone(self.auth_manager.current_user)
    
    def test_login_vk(self):
        """Тест 6: Авторизация через VK ID"""
        token = "test_token"
        user_id = "12345"
        
        success, message = self.auth_manager.login_vk(token, user_id)
        self.assertTrue(success)
        self.assertEqual(self.auth_manager.current_user, f"vk_{user_id}")
        
        # Проверяем, что пользователь был добавлен в файл
        with open(self.test_users_file, 'r') as f:
            users = json.load(f)
        
        self.assertIn(f"vk_{user_id}", users)
        self.assertEqual(users[f"vk_{user_id}"]["vk_id"], user_id)
    
    def test_login_yandex(self):
        """Тест 7: Авторизация через Яндекс ID"""
        token = "test_token"
        user_info = {
            "id": "67890",
            "display_name": "Яндекс Пользователь"
        }
        
        success, message = self.auth_manager.login_yandex(token, user_info)
        self.assertTrue(success)
        self.assertEqual(self.auth_manager.current_user, f"yandex_{user_info['id']}")
        
        # Проверяем, что пользователь был добавлен в файл
        with open(self.test_users_file, 'r') as f:
            users = json.load(f)
        
        self.assertIn(f"yandex_{user_info['id']}", users)
        self.assertEqual(users[f"yandex_{user_info['id']}"]["yandex_id"], user_info['id'])
        self.assertEqual(users[f"yandex_{user_info['id']}"]["name"], user_info['display_name'])

if __name__ == "__main__":
    unittest.main() 