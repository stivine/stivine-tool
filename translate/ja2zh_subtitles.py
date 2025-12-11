import os
import json
import tempfile
import argparse
import ffmpeg
import whisper
import pysrt
from tqdm import tqdm
from pathlib import Path

# ========== CONFIG ==========
MODEL_NAME = "large"           # Whisper 模型
GPT_MODEL = "deepseek/deepseek-chat"      # 支持 litellm 的任意模型
LANG = "ja"                    # 原始语音语言
OUTPUT_LANG = "zh"             # 目标语言
CACHE_DIR = ".cache"           # 缓存目录名

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {
    '.mp4', '.mkv', '.mov', '.avi', '.wmv', '.flv', '.webm',  # 视频
    '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'           # 音频
}
# ============================

import litellm
from litellm import completion


class VideoSubtitleGenerator:
    def __init__(self, api_key=None, api_base=None):
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_base = api_base or os.getenv("API_BASE")
        if not self.api_key and GPT_MODEL.startswith("gpt"):
            raise ValueError("❌ 未提供 API Key。请设置环境变量 API_KEY 或通过 --api_key 参数传入。")

        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.whisper_model = None  # 延迟加载

    def extract_audio(self, video_path, wav_path):
        """提取音频为单声道 16kHz WAV"""
        # 如果输入本身就是音频文件，直接转换格式，不做提取流操作
        input_ext = Path(video_path).suffix.lower()
        is_audio_file = input_ext in {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}
        
        print(f"🎧 正在处理音频: {Path(video_path).name}...")
        try:
            stream = ffmpeg.input(video_path)
            stream.output(wav_path, ac=1, ar=16000).overwrite_output().run(
                quiet=True, capture_stdout=False, capture_stderr=False
            )
            # print(f"✅ 音频准备就绪: {wav_path}")
        except ffmpeg.Error as e:
            raise RuntimeError(f"音频提取/转换失败: {e.stderr.decode()}") from e

    def _seconds_to_srttime(self, seconds):
        """将浮点秒数转换为 pysrt.SubRipTime 对象"""
        ms = int((seconds - int(seconds)) * 1000)
        total_seconds = int(seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return pysrt.SubRipTime(hours, minutes, seconds, ms)

    def transcribe_with_cache(self, audio_path, cache_key):
        """使用缓存进行语音识别"""
        cache_file = self.cache_dir / f"{cache_key}_transcribe.json"
        
        if cache_file.exists():
            print("🔍 [1/2] 检测到转录缓存，正在加载...")
            with open(cache_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            print("🧠 [1/2] 使用 Whisper 进行语音识别...")
            if self.whisper_model is None:
                print(f"   (正在加载 Whisper {MODEL_NAME} 模型，请稍候...)")
                self.whisper_model = whisper.load_model(MODEL_NAME)

            result = self.whisper_model.transcribe(audio_path, language=LANG, verbose=False)
            segments = result["segments"]

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

        subs = pysrt.SubRipFile()
        for i, seg in enumerate(segments):
            start = self._seconds_to_srttime(seg["start"])
            end = self._seconds_to_srttime(seg["end"])
            text = seg["text"].strip()
            subs.append(pysrt.SubRipItem(index=i+1, start=start, end=end, text=text))
        return subs

    def translate_with_cache(self, subs, cache_key):
        """使用 litellm 进行翻译"""
        cache_file = self.cache_dir / f"{cache_key}_translate.json"
        translated_subs = pysrt.SubRipFile()

        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                print(f"🔍 [2/2] 检测到翻译缓存，已恢复 {len(cache_data)} 条记录")
            except Exception:
                pass

        litellm.drop_params = True
        litellm.telemetry = False

        print("🌐 [2/2] 开始 AI 翻译...")
        # 使用 tqdm 显示进度
        pbar = tqdm(subs, desc="   翻译进度", unit="句")
        for sub in pbar:
            key = str(sub.index)
            
            # 如果缓存中有且不为空，直接使用
            if key in cache_data and cache_data[key].strip():
                zh_text = cache_data[key]
            else:
                prompt = f"请将以下{LANG}字幕自然流畅地翻译成简体中文字幕，不要解释，不要保留原文，不要添加额外内容：\n\n{sub.text}"
                try:
                    response = completion(
                        model=GPT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        api_key=self.api_key,
                        api_base=self.api_base
                    )
                    zh_text = response.choices[0].message['content'].strip()
                except Exception as e:
                    # 简单展示错误信息，不打断整个流程
                    pbar.write(f"⚠️ 第 {sub.index} 句翻译失败: {e}")
                    zh_text = f"[翻译失败]"
                
                # 写入缓存
                cache_data[key] = zh_text
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)

            translated_subs.append(
                pysrt.SubRipItem(index=sub.index, start=sub.start, end=sub.end, text=zh_text)
            )

        return translated_subs

    def generate_subtitles(self, input_path, output_srt):
        input_path = Path(input_path).resolve()
        
        # 使用文件名作为缓存Key，防止路径变化导致缓存失效
        # 注意：如果有不同文件夹下的同名文件，可能会冲突，建议加上文件大小或Hash更严谨，此处简化处理
        cache_key = f"{input_path.stem}_{input_path.stat().st_size}"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            wav_path = tmp_wav.name
            
            # Windows下 tempfile 有时会因为权限问题无法被ffmpeg写入，如果报错可改为当前目录临时文件
            try:
                self.extract_audio(str(input_path), wav_path)
            except PermissionError:
                wav_path = "temp_audio_extract.wav"
                self.extract_audio(str(input_path), wav_path)

            subs_ja = self.transcribe_with_cache(wav_path, cache_key)
            subs_zh = self.translate_with_cache(subs_ja, cache_key)

            subs_zh.save(output_srt, encoding="utf-8")
            print(f"🎉 字幕已保存: {output_srt}\n" + "-"*40)
            
            if os.path.exists("temp_audio_extract.wav"):
                os.remove("temp_audio_extract.wav")


def find_media_files(path):
    """递归查找目录下所有支持的媒体文件"""
    path = Path(path)
    files = []
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    elif path.is_dir():
        for item in path.rglob('*'):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(item)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser(description="🎬 视频/音频字幕自动生成（支持 批量目录处理 + 断点续传）")
    parser.add_argument("input", help="输入文件路径 或 包含媒体文件的目录路径")
    parser.add_argument("--api_key", help="API Key")
    parser.add_argument("--api_base", help="API Base URL (例如 Ollama: http://localhost:11434)")
    parser.add_argument("--output", help="输出路径 (如果是单文件则为文件名，如果是目录则为输出文件夹，默认保存在同级目录)")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        exit(1)

    # 1. 扫描文件
    media_files = find_media_files(input_path)
    
    if not media_files:
        print(f"❌ 在 '{input_path}' 中未找到支持的媒体文件。")
        print(f"支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")
        exit(1)

    print(f"📂 找到 {len(media_files)} 个待处理文件。")
    print("-" * 40)

    # 2. 初始化生成器
    try:
        generator = VideoSubtitleGenerator(api_key=args.api_key, api_base=args.api_base)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        exit(1)

    # 3. 批量处理循环
    success_count = 0
    fail_count = 0

    for idx, file_path in enumerate(media_files, 1):
        print(f"🎬 [{idx}/{len(media_files)}] 正在处理: {file_path.name}")
        
        # 确定输出文件路径
        if input_path.is_file() and args.output and not Path(args.output).suffix == '':
            # 单文件输入，且指定了具体的输出文件名
            output_srt = Path(args.output)
        else:
            # 目录输入，或者未指定具体文件名 -> 自动生成同名 .zh.srt
            # 如果 args.output 是一个目录，则保存到该目录；否则保存在原视频同级目录
            target_dir = file_path.parent
            if args.output and Path(args.output).is_dir():
                target_dir = Path(args.output)
            
            output_srt = target_dir / f"{file_path.stem}.zh.srt"

        # 如果输出文件已存在，可以选择跳过（此处未实现，可根据需求添加检查）
        
        try:
            generator.generate_subtitles(file_path, output_srt)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理失败 '{file_path.name}': {e}")
            fail_count += 1
            # 继续处理下一个，不退出
            print("-" * 40)

    # 4. 总结
    print("\n" + "=" * 40)
    print(f"🏁 任务完成！成功: {success_count}，失败: {fail_count}")

if __name__ == "__main__":
    main()