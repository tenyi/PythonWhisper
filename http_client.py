"""
簡單的 HTTP 客戶端模組
用於與 Ollama API 進行通訊
"""

import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HTTPClient:
    """簡單的 HTTP 客戶端"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化 HTTP 客戶端

        Args:
            base_url: API 基礎 URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5 分鐘超時

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        發送 POST 請求

        Args:
            endpoint: API 端點
            data: 請求資料

        Returns:
            API 回應資料

        Raises:
            requests.RequestException: 當請求失敗時
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            logger.debug(f"發送 POST 請求到: {url}")
            response = self.session.post(url, json=data)

            # 檢查 HTTP 狀態碼
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"HTTP 請求失敗: {url} - {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"JSON 解析失敗: {str(e)}")
            raise

    def get(self, endpoint: str) -> Dict[str, Any]:
        """
        發送 GET 請求

        Args:
            endpoint: API 端點

        Returns:
            API 回應資料
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            logger.debug(f"發送 GET 請求到: {url}")
            response = self.session.get(url)

            # 檢查 HTTP 狀態碼
            response.raise_for_status()

            result = response.json()
            return result

        except requests.RequestException as e:
            logger.error(f"HTTP 請求失敗: {url} - {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"JSON 解析失敗: {str(e)}")
            raise

    def close(self):
        """關閉 HTTP 會話"""
        if self.session:
            self.session.close()