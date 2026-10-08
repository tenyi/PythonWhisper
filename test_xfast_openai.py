#!/usr/bin/env python3
"""
測試 xfast.py 使用 openai-whisper 的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_initialization():
    """測試模型初始化功能"""
    print("測試模型初始化...")
    from xfast import initialize_model, release_model, get_best_device
    
    # 檢查裝置
    device = get_best_device()
    print(f"偵測到的最佳裝置: {device}")
    
    # 初始化模型
    try:
        initialize_model()
        print("✅ 模型初始化成功")
        
        # 釋放模型
        release_model()
        print("✅ 模型釋放成功")
        
    except Exception as e:
        print(f"❌ 模型操作失敗: {e}")
        return False
        
    return True

def test_transcribe_functionality():
    """測試轉錄功能（需要音檔）"""
    print("\n測試轉錄功能...")
    
    # 檢查是否有可用的測試音檔
    test_files = [
        "test.wav",
        "test.mp3", 
        "test.m4a",
        "audio.wav",
        "audio.mp3"
    ]
    
    test_file = None
    for filename in test_files:
        if os.path.exists(filename):
            test_file = filename
            break
    
    if not test_file:
        print("⚠️  找不到測試音檔，跳過轉錄測試")
        print("   可用的測試檔案名稱：", ", ".join(test_files))
        return True
    
    try:
        from xfast import transcribe_openai_format, transcribe_full
        
        # 測試 OpenAI 格式輸出
        print(f"使用檔案 {test_file} 進行測試...")
        result = transcribe_openai_format(test_file)
        print("✅ OpenAI 格式轉錄成功")
        print(f"結果長度: {len(result)} 字元")
        
        # 測試完整轉錄功能
        full_result = transcribe_full(test_file)
        print("✅ 完整轉錄成功")
        print(f"完整結果長度: {len(full_result)} 字元")
        
    except Exception as e:
        print(f"❌ 轉錄功能測試失敗: {e}")
        return False
        
    return True

def test_data_structures():
    """測試自定義的資料結構"""
    print("\n測試資料結構...")
    
    try:
        from xfast import Segment, TranscriptionInfo
        
        # 測試 Segment
        segment = Segment(start=0.0, end=5.0, text="測試文字")
        print(f"✅ Segment 建立成功: {segment.start}-{segment.end}: {segment.text}")
        
        # 測試 TranscriptionInfo
        info = TranscriptionInfo(language="zh", language_probability=0.95, duration=10.0)
        print(f"✅ TranscriptionInfo 建立成功: {info.language} ({info.language_probability}) {info.duration}s")
        
    except Exception as e:
        print(f"❌ 資料結構測試失敗: {e}")
        return False
        
    return True

def main():
    """主測試函式"""
    print("🎵 openai-whisper 整合測試開始")
    print("=" * 50)
    
    tests = [
        ("資料結構", test_data_structures),
        ("模型初始化", test_model_initialization),
        ("轉錄功能", test_transcribe_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 測試 {test_name} 發生異常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 測試結果摘要:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n總計: {passed}/{len(results)} 個測試通過")
    
    if passed == len(results):
        print("🎉 所有測試都通過！openai-whisper 整合成功")
    else:
        print("⚠️  部分測試未通過，請檢查配置")

if __name__ == "__main__":
    main()
