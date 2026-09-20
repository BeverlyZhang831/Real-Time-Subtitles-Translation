# 实时字幕与翻译系统

一个面向英语大学课堂场景的实时语音识别、中文翻译与课堂总结工具。

本项目基于开源项目进行二次开发，使用 Whisper 进行英语语音识别，并结合机器翻译与 Qwen LLM，实现实时字幕、中文翻译、课堂记录和自动课堂总结。

## ✨ 主要功能

- 🎙️ 英语实时语音识别
- 📝 实时显示英文字幕
- 🇨🇳 英文 → 中文实时翻译
- 🤖 支持机器翻译与 LLM 翻译
- 📚 针对计算机科学课堂优化的术语识别
- 🧠 自动生成课堂总结
- 💾 自动保存课堂记录与总结
- 🖥️ PyQt5 图形界面
- 🍎 支持 Apple Silicon Mac 本地运行

## 🛠️ 技术栈

- Python
- PyQt5
- OpenAI Whisper
- MarianMT
- Ollama
- Qwen2.5 7B
- PyAudio
- PyTorch

## 🎯 项目背景

这个项目主要用于英语授课的大学课堂。

在计算机科学课程中，课程中会出现大量专业术语，例如：

- Programming
- Algorithms
- Data Structures
- Computer Organisation
- Operating Systems
- Compiler
- Pointer
- Recursion
- Database
- Computer Networks
- Machine Learning

因此，我在 Whisper 的识别提示中加入了大量计算机科学和大学课堂相关术语，以改善特定课堂场景下的识别效果。

## 📖 课堂总结

系统可以根据课堂中已经识别出的内容，使用 Qwen LLM 自动生成结构化课堂总结，包括：

- 课堂主题
- 核心知识点
- 重要术语
- 老师讲解的例子
- 需要复习的重点

总结同时会保存到本地文件。

## 🔄 工作流程

语音输入  
↓  
Whisper 英语语音识别  
↓  
实时英文字幕  
↓  
中文翻译  
↓  
课堂内容记录  
↓  
Qwen LLM 自动总结

## 📌 项目说明

本项目是在开源项目基础上进行的个人二次开发，保留并遵循原项目的相关许可证及版权要求。

## 👩‍💻 Author

Beverly Zhang

Computer Science Student
