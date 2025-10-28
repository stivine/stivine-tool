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
MODEL_NAME = "small"           # Whisper 模型 (tiny, base, small, medium, large)
GPT_MODEL = "deepseek/deepseek-chat"      # 支持 litellm 的任意模型：gpt-4o-mini, claude-3-haiku, gemini/gemini-pro, ollama/llama3, etc.
LANG = "ja"                    # 原始语音语言
OUTPUT_LANG = "zh"             # 目标语言
CACHE_DIR = ".cache"           # 缓存目录名
# ============================

# 使用 litellm 代替 openai
import litellm
from litellm import completion


class VideoSubtitleGenerator:
    def __init__(self, api_key=None, api_base=None):
        self.api_key = api_key or os.getenv("API_KEY")  # 可适配不同平台的 KEY
        self.api_base = api_base or os.getenv("API_BASE")  # 如 Ollama 自定义地址
        if not self.api_key and GPT_MODEL.startswith("gpt"):
            raise ValueError("❌ 未提供 API Key。请设置环境变量 API_KEY 或通过 --api_key 参数传入。")

        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def extract_audio(self, video_path, wav_path):
        """提取音频为单声道 16kHz WAV"""
        print("🎧 正在提取音频...")
        try:
            ffmpeg.input(video_path).output(
                wav_path, ac=1, ar=16000
            ).overwrite_output().run(quiet=True, capture_stdout=False, capture_stderr=False)
            print(f"✅ 音频已提取至: {wav_path}")
        except ffmpeg.Error as e:
            raise RuntimeError(f"音频提取失败: {e.stderr.decode()}") from e

    def _seconds_to_srttime(self, seconds):
        """
        将浮点秒数转换为 pysrt.SubRipTime 对象
        兼容不支持 from_seconds() 的 pysrt 版本
        """
        ms = int((seconds - int(seconds)) * 1000)
        total_seconds = int(seconds)
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return pysrt.SubRipTime(hours, minutes, seconds, ms)

    def transcribe_with_cache(self, audio_path, cache_key):
        """使用缓存进行语音识别"""
        cache_file = self.cache_dir / f"{cache_key}_transcribe.json"
        
        # 检查缓存是否存在
        if cache_file.exists():
            print("🔍 检测到转录缓存，正在加载...")
            with open(cache_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
            print("✅ 转录缓存加载完成")
        else:
            print("🧠 使用 Whisper 进行语音识别...")
            if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                self.whisper_model = whisper.load_model(MODEL_NAME)

            result = self.whisper_model.transcribe(audio_path, language=LANG, verbose=False)
            segments = result["segments"]

            # 保存缓存
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            print("✅ 转录完成并已缓存")

        # 转换为 pysrt 字幕对象
        subs = pysrt.SubRipFile()
        for i, seg in enumerate(segments):
            start = self._seconds_to_srttime(seg["start"])
            end = self._seconds_to_srttime(seg["end"])
            text = seg["text"].strip()
            subs.append(pysrt.SubRipItem(index=i+1, start=start, end=end, text=text))
        return subs

    def translate_with_cache(self, subs, cache_key):
        """使用 litellm 进行翻译，支持多种模型 + 断点续传"""
        cache_file = self.cache_dir / f"{cache_key}_translate.json"
        translated_subs = pysrt.SubRipFile()

        # 尝试加载已有翻译缓存
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                print(f"🔍 检测到翻译缓存，已恢复 {len(cache_data)} 条翻译记录")
            except Exception as e:
                print(f"⚠️ 无法读取翻译缓存: {e}")

        # 设置 litellm 全局参数（可选）
        litellm.drop_params = True  # 允许传入但模型不支持的参数自动忽略
        litellm.telemetry = False  # 关闭遥测

        # 逐条翻译
        for sub in tqdm(subs, desc="🌐 翻译进度"):
            key = str(sub.index)
            if key in cache_data and cache_data[key].strip():
                zh_text = cache_data[key]
                translated_subs.append(
                    pysrt.SubRipItem(index=sub.index, start=sub.start, end=sub.end, text=zh_text)
                )
                continue

            # 构建提示词
            prompt = f"请将以下日语字幕自然流畅地翻译成简体中文字幕，不要解释，不要保留日语，不要添加额外内容：\n\n{sub.text}"

            try:
                response = completion(
                    model=GPT_MODEL,  # 如 "gpt-4o-mini", "claude-3-haiku", "gemini/gemini-pro", "ollama/llama3"
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    api_key=self.api_key,
                    api_base=self.api_base  # 支持自定义端点（如 Ollama）
                )
                zh_text = response.choices[0].message['content'].strip()
            except Exception as e:
                print(f"\n⚠️ 第 {sub.index} 句翻译失败: {e}")
                zh_text = f"[翻译失败: {str(e)}]"

            # 更新缓存和字幕
            cache_data[key] = zh_text
            translated_subs.append(
                pysrt.SubRipItem(index=sub.index, start=sub.start, end=sub.end, text=zh_text)
            )

            # 实时保存缓存，防止中断
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print("✅ 翻译完成")
        return translated_subs

    def generate_subtitles(self, video_path, output_srt):
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cache_key = video_path.stem

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            wav_path = tmp_wav.name

            # 1. 提取音频
            self.extract_audio(str(video_path), wav_path)

            # 2. 转录
            subs_ja = self.transcribe_with_cache(wav_path, cache_key)

            # 3. 翻译（使用 litellm）
            subs_zh = self.translate_with_cache(subs_ja, cache_key)

            # 4. 保存字幕
            subs_zh.save(output_srt, encoding="utf-8")
            print(f"✅ 中文字幕已保存: {output_srt}")


def main():
    parser = argparse.ArgumentParser(description="🎬 日语视频 → 中文字幕自动生成（支持断点续传，使用 litellm 多模型）")
    parser.add_argument("video", help="输入视频文件路径（如: video.mp4）")
    parser.add_argument("--api_key", help="API Key（如 OpenAI、Anthropic、Gemini 等）")
    parser.add_argument("--api_base", help="API Base URL（如 Ollama 地址: http://localhost:11434）")
    parser.add_argument("--output", default="subs_zh.srt", help="输出字幕文件路径（默认: subs_zh.srt）")

    args = parser.parse_args()

    try:
        generator = VideoSubtitleGenerator(api_key=args.api_key, api_base=args.api_base)
        generator.generate_subtitles(args.video, args.output)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        exit(1)


if __name__ == "__main__":
    main()