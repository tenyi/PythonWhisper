#!/usr/bin/env python3
"""
xfast.py openai-whisper 整合驗證腳本
"""

def main():
    print("🎵 xfast.py openai-whisper 整合驗證")
    print("=" * 50)
    
    # 測試 1: 基本資料結構
    print("\n📋 測試 1: 基本資料結構")
    try:
        from xfast import Segment, TranscriptionInfo
        
        seg = Segment(0.0, 5.0, "測試語音片段")
        info = TranscriptionInfo("zh", 0.95, 10.0)
        
        print(f"✅ Segment: {seg.start}s-{seg.end}s: '{seg.text}'")
        print(f"✅ TranscriptionInfo: {info.language} ({info.language_probability:.2f}) {info.duration}s")
        
    except Exception as e:
        print(f"❌ 資料結構測試失敗: {e}")
        return False
    
    # 測試 2: JSON 序列化
    print("\n📋 測試 2: JSON 序列化")
    try:
        from xfast import OpenAI_Transcribe
        
        segments = [
            Segment(0.0, 2.5, "第一段語音"),
            Segment(2.5, 5.0, "第二段語音")
        ]
        
        transcribe = OpenAI_Transcribe(
            text="第一段語音 第二段語音",
            language="zh",
            duration=5.0,
            segments=segments
        )
        
        json_output = transcribe.model_dump_json()
        print(f"✅ JSON 序列化成功，長度: {len(json_output)} 字元")
        print(f"   預覽: {json_output[:100]}...")
        
    except Exception as e:
        print(f"❌ JSON 序列化測試失敗: {e}")
        return False
    
    # 測試 3: 裝置偵測
    print("\n📋 測試 3: 裝置偵測")
    try:
        from xfast import get_best_device
        
        device = get_best_device()
        print(f"✅ 偵測完成")
        
    except Exception as e:
        print(f"❌ 裝置偵測失敗: {e}")
        return False
    
    # 測試 4: 模型載入（可選）
    print("\n📋 測試 4: 模型載入測試")
    print("⚠️  注意：首次執行會下載模型，可能需要幾分鐘時間")
    
    try_model = input("是否要測試模型載入？(y/N): ").strip().lower()
    
    if try_model == 'y':
        try:
            from xfast import initialize_model, release_model
            
            print("正在載入 Whisper 模型...")
            initialize_model()
            print("✅ 模型載入成功")
            
            release_model()
            print("✅ 模型釋放成功")
            
        except Exception as e:
            print(f"❌ 模型載入測試失敗: {e}")
            return False
    else:
        print("⏭️  跳過模型載入測試")
    
    # 總結
    print("\n" + "=" * 50)
    print("🎉 所有測試完成！")
    print("✅ xfast.py 已成功整合 openai-whisper")
    print("\n📝 接下來可以：")
    print("   1. 使用 transcribe_openai_format(audio_file) 進行語音轉錄")
    print("   2. 使用 transcribe_full(audio_file) 進行完整轉錄並生成所有格式")
    print("   3. 直接執行: python xfast.py your_audio_file.mp3")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 整合成功！可以開始使用 openai-whisper 了。")
    else:
        print("\n⚠️  整合過程中發現問題，請檢查錯誤訊息。")
    
    exit(0 if success else 1)
