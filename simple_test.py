#!/usr/bin/env python3
"""
簡單的 xfast.py 測試
"""

def test_basic_imports():
    """測試基本匯入"""
    print("📋 測試基本匯入...")
    try:
        from xfast import Segment, TranscriptionInfo, get_best_device
        print("✅ 基本匯入成功")
        return True
    except Exception as e:
        print(f"❌ 基本匯入失敗: {e}")
        return False

def test_data_structures():
    """測試資料結構"""
    print("📋 測試資料結構...")
    try:
        from xfast import Segment, TranscriptionInfo
        
        # 測試 Segment
        segment = Segment(0.0, 5.0, "測試文字")
        print(f"✅ Segment: {segment.start}-{segment.end}: {segment.text}")
        
        # 測試 TranscriptionInfo
        info = TranscriptionInfo("zh", 0.95, 10.0)
        print(f"✅ TranscriptionInfo: {info.language} ({info.language_probability}) {info.duration}s")
        
        return True
    except Exception as e:
        print(f"❌ 資料結構測試失敗: {e}")
        return False

def test_device_detection():
    """測試裝置偵測"""
    print("📋 測試裝置偵測...")
    try:
        from xfast import get_best_device
        device = get_best_device()
        print(f"✅ 偵測到裝置: {device}")
        return True
    except Exception as e:
        print(f"❌ 裝置偵測失敗: {e}")
        return False

def test_model_initialization():
    """測試模型初始化（可能需要較長時間）"""
    print("📋 測試模型初始化...")
    try:
        from xfast import initialize_model, release_model
        
        print("正在初始化模型（首次下載可能需要幾分鐘）...")
        initialize_model()
        print("✅ 模型初始化成功")
        
        release_model()
        print("✅ 模型釋放成功")
        return True
    except Exception as e:
        print(f"❌ 模型初始化失敗: {e}")
        return False

def main():
    """主測試函式"""
    print("🎵 xfast.py 簡單測試")
    print("=" * 40)
    
    tests = [
        ("基本匯入", test_basic_imports),
        ("資料結構", test_data_structures),
        ("裝置偵測", test_device_detection),
        ("模型初始化", test_model_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 發生異常: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("📊 測試結果:")
    
    passed = sum(1 for _, result in results if result)
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n總計: {passed}/{len(results)} 個測試通過")
    
    if passed == len(results):
        print("🎉 所有測試都通過！openai-whisper 整合成功")
        return True
    else:
        print("⚠️  部分測試未通過")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
