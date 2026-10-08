#!/usr/bin/env python3
"""
測試新的隊列式 Whisper API
"""

import requests
import time
import json

# API 基礎 URL
BASE_URL = "http://localhost:22434"

def test_queue_api():
    """測試隊列 API 功能"""
    
    print("🧪 測試隊列式 Whisper API")
    print("=" * 50)
    
    # 1. 檢查健康狀態
    print("1. 檢查服務健康狀態...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ 服務正常運行")
            print(f"   隊列狀態: {health_data['queue_info']}")
        else:
            print(f"❌ 健康檢查失敗: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ 無法連接到服務: {e}")
        return
    
    # 2. 檢查隊列狀態
    print("\n2. 檢查隊列狀態...")
    try:
        response = requests.get(f"{BASE_URL}/queue/status")
        if response.status_code == 200:
            queue_data = response.json()
            print(f"✅ 隊列狀態: {queue_data}")
        else:
            print(f"❌ 獲取隊列狀態失敗: {response.status_code}")
    except Exception as e:
        print(f"❌ 獲取隊列狀態錯誤: {e}")
    
    # 3. 模擬上傳音頻文件（這裡只是演示 API 調用）
    print("\n3. 模擬提交 transcribe 任務...")
    print("   注意: 這個測試需要實際的音頻文件")
    print("   可以使用以下 curl 命令測試:")
    print(f"   curl -X POST -F 'file=@your_audio.mp3' {BASE_URL}/transcribe")
    
    # 4. 測試任務狀態查詢（使用假的 task_id）
    print("\n4. 測試任務狀態查詢...")
    fake_task_id = "00000000-0000-0000-0000-000000000000"
    try:
        response = requests.get(f"{BASE_URL}/task/{fake_task_id}")
        if response.status_code == 404:
            print(f"✅ 正確處理不存在的任務 ID")
        else:
            print(f"⚠️  意外的響應: {response.status_code}")
    except Exception as e:
        print(f"❌ 任務狀態查詢錯誤: {e}")

def test_concurrent_requests():
    """測試並發請求處理"""
    print("\n🔄 測試並發請求處理")
    print("=" * 50)
    print("   這個測試需要實際的音頻文件來驗證隊列功能")
    print("   建議使用多個終端同時發送請求:")
    print(f"   curl -X POST -F 'file=@audio1.mp3' {BASE_URL}/transcribe")
    print(f"   curl -X POST -F 'file=@audio2.mp3' {BASE_URL}/transcribe")
    print("   然後觀察返回的 task_id 和隊列狀態")

if __name__ == "__main__":
    test_queue_api()
    test_concurrent_requests()
    
    print("\n📝 使用說明:")
    print("1. 啟動服務: python -m flask --app flask-whisper:create_app run --host=0.0.0.0 --port=22434")
    print("2. 提交任務: curl -X POST -F 'file=@audio.mp3' http://localhost:22434/transcribe")
    print("3. 查詢狀態: curl http://localhost:22434/task/{task_id}")
    print("4. 查詢隊列: curl http://localhost:22434/queue/status")