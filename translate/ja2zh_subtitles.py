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
# Whisper 模型大小 (tiny, base, small, medium, large)
MODEL_NAME = "large"

# 支持 litellm 的任意模型：gpt-4o, deepseek-chat, claude-3-haiku, ollama/llama3 等
GPT_MODEL = "deepseek/deepseek-chat"

# 语言设置
LANG = "ja"                    # 原始语音语言
OUTPUT_LANG = "zh"             # 目标语言

# 缓存目录
CACHE_DIR = ".cache"

# 【重要】批量翻译大小：一次发送给 AI 多少句字幕
# 建议 10-20。太小没上下文，太大容易导致 AI 漏翻或 JSON 格式错误
BATCH_SIZE = 15

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
        
        # 简单检查 Key (本地模型如 Ollama 可忽略)
        if not self.api_key and "gpt" in GPT_MODEL:
            print("⚠️ 警告: 未检测到 API Key，如果使用 OpenAI/DeepSeek 等在线模型可能会失败。")

        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.whisper_model = None  # 延迟加载，只有在没有缓存时才加载

    def extract_audio(self, video_path, wav_path):
        """提取音频为单声道 16kHz WAV"""
        input_ext = Path(video_path).suffix.lower()
        # 如果本身就是音频，直接转码；如果是视频，则提取
        print(f"🎧 正在处理音频: {Path(video_path).name}...")
        try:
            stream = ffmpeg.input(video_path)
            stream.output(wav_path, ac=1, ar=16000).overwrite_output().run(
                quiet=True, capture_stdout=False, capture_stderr=False
            )
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
        """Step 1: 语音识别 (Whisper) + 缓存"""
        cache_file = self.cache_dir / f"{cache_key}_transcribe.json"
        
        if cache_file.exists():
            print("🔍 [1/2] 检测到转录缓存，直接加载...")
            with open(cache_file, "r", encoding="utf-8") as f:
                segments = json.load(f)
        else:
            print("🧠 [1/2] 正在进行 Whisper 语音识别...")
            if self.whisper_model is None:
                print(f"   (正在加载 Whisper {MODEL_NAME} 模型，首次运行可能需要下载...)")
                self.whisper_model = whisper.load_model(MODEL_NAME)

            result = self.whisper_model.transcribe(audio_path, language=LANG, verbose=False)
            segments = result["segments"]

            # 写入缓存
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)

        # 转换为 pysrt 对象
        subs = pysrt.SubRipFile()
        for i, seg in enumerate(segments):
            start = self._seconds_to_srttime(seg["start"])
            end = self._seconds_to_srttime(seg["end"])
            text = seg["text"].strip()
            subs.append(pysrt.SubRipItem(index=i+1, start=start, end=end, text=text))
        return subs

    def translate_with_cache(self, subs, cache_key):
        """Step 2: 批量上下文翻译 (LLM) + 缓存"""
        cache_file = self.cache_dir / f"{cache_key}_translate.json"
        
        # 1. 加载已有翻译缓存
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                print(f"🔍 [2/2] 检测到翻译缓存，已恢复 {len(cache_data)} 条记录")
            except Exception:
                pass

        # 配置 litellm
        litellm.drop_params = True
        litellm.telemetry = False

        # 2. 准备数据
        all_items = []
        for sub in subs:
            all_items.append({
                "id": sub.index,
                "text": sub.text,
                "start": sub.start,
                "end": sub.end
            })

        total_batches = (len(all_items) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"🌐 [2/2] 开始 AI 批量翻译 (共 {len(all_items)} 句，分 {total_batches} 批)...")

        translated_subs = pysrt.SubRipFile()

        # 3. 逐批次处理
        # 使用 tqdm 进度条，步长为 BATCH_SIZE
        pbar = tqdm(range(0, len(all_items), BATCH_SIZE), desc="   翻译进度", unit="批")
        for i in pbar:
            batch_items = all_items[i : i + BATCH_SIZE]
            
            # 检查该批次是否全部在缓存中
            batch_needs_translation = False
            for item in batch_items:
                if str(item["id"]) not in cache_data:
                    batch_needs_translation = True
                    break
            
            # 如果缓存命中，直接使用
            if not batch_needs_translation:
                for item in batch_items:
                    zh_text = cache_data[str(item["id"])]
                    translated_subs.append(pysrt.SubRipItem(
                        index=item["id"], start=item["start"], end=item["end"], text=zh_text
                    ))
                continue

            # 构建 Prompt
            # 将多句话打包成 JSON 格式: {"1": "原文1", "2": "原文2"}
            source_dict = {str(item["id"]): item["text"] for item in batch_items}
            prompt_content = json.dumps(source_dict, ensure_ascii=False, indent=1)
            
            system_prompt = (
                f"你是一个专业的字幕翻译专家。请将以下{LANG}字幕翻译成简体中文。\n"
                "要求：\n"
                "1. 结合上下文翻译，确保通顺、自然、符合逻辑。\n"
                "2. 严格返回 JSON 格式，Key是字幕ID，Value是翻译结果。\n"
                "3. 绝对不要合并句子，不要漏掉任何ID，必须一一对应。\n"
                "4. 只输出 JSON，不要包含 Markdown 代码块或解释。"
            )

            try:
                response = completion(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=0.3,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    response_format={"type": "json_object"} # 提示模型返回 JSON
                )
                
                content = response.choices[0].message['content'].strip()
                
                # 清洗 Markdown 标记 (以防模型不听话)
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "")
                
                translated_batch = json.loads(content)
                
            except Exception as e:
                pbar.write(f"⚠️ 批次翻译异常: {e}")
                translated_batch = {} # 标记为空，后续逻辑会回退到原文

            # 处理返回结果并更新缓存
            for item in batch_items:
                sid = str(item["id"])
                # 如果翻译成功取翻译，否则取原文并标记
                if sid in translated_batch and isinstance(translated_batch[sid], str):
                    final_text = translated_batch[sid]
                else:
                    final_text = f"[翻译失败] {item['text']}"

                cache_data[sid] = final_text
                
                translated_subs.append(pysrt.SubRipItem(
                    index=item["id"], start=item["start"], end=item["end"], text=final_text
                ))

            # 实时保存缓存 (每批保存一次，防止崩溃丢失)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        return translated_subs

    def generate_subtitles(self, input_path, output_srt):
        input_path = Path(input_path).resolve()
        
        # 使用文件名+大小作为缓存Key
        cache_key = f"{input_path.stem}_{input_path.stat().st_size}"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
            wav_path = tmp_wav.name
            
            # Windows 权限兼容处理
            temp_local_wav = None
            try:
                self.extract_audio(str(input_path), wav_path)
            except PermissionError:
                # 如果系统临时目录无法写入，使用当前目录
                temp_local_wav = "temp_audio_extract.wav"
                self.extract_audio(str(input_path), temp_local_wav)
                wav_path = temp_local_wav

            # 1. 语音转文字
            subs_ja = self.transcribe_with_cache(wav_path, cache_key)
            
            # 2. 文字翻译
            subs_zh = self.translate_with_cache(subs_ja, cache_key)

            # 3. 保存
            subs_zh.save(output_srt, encoding="utf-8")
            print(f"🎉 字幕已保存: {output_srt}\n" + "-"*40)
            
            # 清理本地临时文件
            if temp_local_wav and os.path.exists(temp_local_wav):
                os.remove(temp_local_wav)

def find_media_files(path):
    """递归查找所有支持的媒体文件"""
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
    parser = argparse.ArgumentParser(description="🎬 视频/音频字幕自动生成 (Whisper + LLM 批量上下文翻译)")
    parser.add_argument("input", help="输入文件路径 或 包含媒体文件的目录路径")
    parser.add_argument("--api_key", help="API Key (OpenAI/DeepSeek等)")
    parser.add_argument("--api_base", help="API Base URL (例如 Ollama: http://localhost:11434)")
    parser.add_argument("--output", help="指定输出目录 (默认保存在视频同级目录)")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        exit(1)

    # 1. 扫描文件
    media_files = find_media_files(input_path)
    if not media_files:
        print(f"❌ 未找到支持的媒体文件。\n支持格式: {', '.join(SUPPORTED_EXTENSIONS)}")
        exit(1)

    print(f"📂 待处理文件数: {len(media_files)}")
    print("-" * 40)

    # 2. 初始化
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
        
        # 确定输出路径
        if input_path.is_file() and args.output and not Path(args.output).suffix == '':
            # 单文件模式且指定了文件名
            output_srt = Path(args.output)
        else:
            # 批量模式或未指定文件名，自动生成 *.zh.srt
            target_dir = file_path.parent
            if args.output and Path(args.output).is_dir():
                target_dir = Path(args.output)
                target_dir.mkdir(parents=True, exist_ok=True)
            
            output_srt = target_dir / f"{file_path.stem}.zh.srt"

        try:
            generator.generate_subtitles(file_path, output_srt)
            success_count += 1
        except Exception as e:
            print(f"❌ 处理失败 '{file_path.name}': {e}")
            fail_count += 1
            print("-" * 40)

    print(f"\n🏁 任务全部完成！成功: {success_count}，失败: {fail_count}")

if __name__ == "__main__":
    main()