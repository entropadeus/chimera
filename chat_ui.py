"""
Chat UI for Chimera LLM - Simple conversational interface.
"""

import json
import argparse
import socket
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from model import Chimera, chimera_small, chimera_medium, chimera_base
from tokenizer import ChimeraTokenizer

MODEL = None
TOKENIZER = None
DEVICE = None

HTML = b'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wyrd</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
    <style>
        :root {
            --cream: #f8f6f1;
            --cream-dark: #efe9df;
            --paper: #fffef9;
            --ink: #2d2a24;
            --ink-soft: #5c584f;
            --ink-muted: #9a958a;
            --sage: #7d9a7a;
            --sage-light: #b8ccb5;
            --sage-pale: #e8efe7;
            --sage-deep: #4a6348;
            --rust: #c4693d;
            --rust-light: #e8a67c;
            --rust-pale: #faf0ea;
            --rust-deep: #9a4a25;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        html { font-size: 16px; }

        body {
            font-family: 'DM Sans', Georgia, serif;
            background: var(--cream);
            color: var(--ink);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            -webkit-font-smoothing: antialiased;
            overflow: hidden;
        }

        /* Animated background texture */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background:
                radial-gradient(ellipse 80% 50% at 20% 20%, var(--sage-pale) 0%, transparent 50%),
                radial-gradient(ellipse 60% 40% at 80% 80%, var(--rust-pale) 0%, transparent 50%);
            opacity: 0.7;
            pointer-events: none;
            z-index: 0;
        }

        /* Paper grain texture overlay */
        body::after {
            content: '';
            position: fixed;
            inset: 0;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)'/%3E%3C/svg%3E");
            opacity: 0.03;
            pointer-events: none;
            z-index: 1;
        }

        /* Floating organic shapes */
        .blob {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            pointer-events: none;
            z-index: 0;
            animation: float 20s ease-in-out infinite;
        }

        .blob-1 {
            width: 500px;
            height: 500px;
            background: var(--sage-light);
            top: -200px;
            left: -100px;
            animation-delay: 0s;
        }

        .blob-2 {
            width: 400px;
            height: 400px;
            background: var(--rust-light);
            bottom: -150px;
            right: -100px;
            animation-delay: -7s;
        }

        .blob-3 {
            width: 300px;
            height: 300px;
            background: var(--sage);
            top: 50%;
            right: 10%;
            opacity: 0.2;
            animation-delay: -14s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(30px, -30px) scale(1.05); }
            50% { transform: translate(-20px, 20px) scale(0.95); }
            75% { transform: translate(20px, 30px) scale(1.02); }
        }

        /* Main container */
        .app {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 1000px;
            margin: 0 auto;
            width: 100%;
        }

        /* Header */
        .header {
            padding: 28px 40px;
            display: flex;
            align-items: center;
            gap: 20px;
            animation: slideDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .logo {
            width: 56px;
            height: 56px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .logo-ring {
            position: absolute;
            inset: 0;
            border: 2px solid var(--sage);
            border-radius: 50%;
            animation: spin 20s linear infinite;
        }

        .logo-ring::before {
            content: '';
            position: absolute;
            width: 8px;
            height: 8px;
            background: var(--rust);
            border-radius: 50%;
            top: -4px;
            left: 50%;
            transform: translateX(-50%);
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .logo-inner {
            font-family: 'Cormorant Garamond', serif;
            font-size: 28px;
            font-weight: 600;
            color: var(--ink);
            font-style: italic;
        }

        .brand {
            flex: 1;
        }

        .title {
            font-family: 'Cormorant Garamond', serif;
            font-size: 32px;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--ink);
            line-height: 1;
        }

        .subtitle {
            font-size: 13px;
            color: var(--ink-muted);
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-top: 4px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 18px;
            background: var(--paper);
            border: 1px solid var(--cream-dark);
            border-radius: 100px;
            font-size: 12px;
            font-weight: 500;
            color: var(--ink-soft);
            box-shadow: 0 2px 8px rgba(45, 42, 36, 0.04);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--sage);
            border-radius: 50%;
            position: relative;
        }

        .status-dot::after {
            content: '';
            position: absolute;
            inset: -3px;
            border: 1px solid var(--sage);
            border-radius: 50%;
            animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
        }

        @keyframes ping {
            0% { transform: scale(1); opacity: 0.8; }
            100% { transform: scale(1.8); opacity: 0; }
        }

        /* Chat Area */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 0 40px 40px;
            display: flex;
            flex-direction: column;
            gap: 32px;
            scroll-behavior: smooth;
        }

        .chat-area::-webkit-scrollbar { width: 4px; }
        .chat-area::-webkit-scrollbar-track { background: transparent; }
        .chat-area::-webkit-scrollbar-thumb {
            background: var(--cream-dark);
            border-radius: 2px;
        }

        /* Welcome */
        .welcome {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 60px 20px;
            animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .welcome-mark {
            width: 120px;
            height: 120px;
            margin-bottom: 40px;
            position: relative;
        }

        .welcome-mark svg {
            width: 100%;
            height: 100%;
        }

        .mark-path {
            fill: none;
            stroke: var(--sage);
            stroke-width: 1;
            stroke-dasharray: 400;
            stroke-dashoffset: 400;
            animation: draw 2s ease forwards 0.5s;
        }

        .mark-fill {
            fill: var(--sage-pale);
            opacity: 0;
            animation: fillIn 0.8s ease forwards 2s;
        }

        @keyframes draw {
            to { stroke-dashoffset: 0; }
        }

        @keyframes fillIn {
            to { opacity: 1; }
        }

        .welcome h1 {
            font-family: 'Cormorant Garamond', serif;
            font-size: 48px;
            font-weight: 400;
            color: var(--ink);
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }

        .welcome h1 em {
            color: var(--rust);
            font-style: italic;
        }

        .welcome p {
            font-size: 17px;
            color: var(--ink-soft);
            max-width: 400px;
            line-height: 1.7;
        }

        .prompts {
            margin-top: 48px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            width: 100%;
            max-width: 400px;
        }

        .prompt-btn {
            padding: 18px 24px;
            background: var(--paper);
            border: 1px solid var(--cream-dark);
            border-radius: 16px;
            font-family: 'DM Sans', sans-serif;
            font-size: 15px;
            color: var(--ink-soft);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            text-align: left;
            position: relative;
            overflow: hidden;
        }

        .prompt-btn::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, var(--sage-pale), var(--rust-pale));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .prompt-btn span {
            position: relative;
            z-index: 1;
        }

        .prompt-btn:hover {
            border-color: var(--sage);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(45, 42, 36, 0.08);
        }

        .prompt-btn:hover::before {
            opacity: 1;
        }

        .prompt-btn:hover span {
            color: var(--ink);
        }

        /* Messages */
        .message-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            animation: messageSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .message-group.user { align-items: flex-end; }
        .message-group.assistant { align-items: flex-start; }

        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.98);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .message-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0 8px;
        }

        .message-author {
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .message-group.user .message-author { color: var(--rust); }
        .message-group.assistant .message-author { color: var(--sage-deep); }

        .message-time {
            font-size: 11px;
            color: var(--ink-muted);
        }

        .message {
            max-width: 75%;
            padding: 20px 24px;
            font-size: 16px;
            line-height: 1.7;
            white-space: pre-wrap;
        }

        .message.user {
            background: linear-gradient(135deg, var(--rust) 0%, var(--rust-deep) 100%);
            color: var(--paper);
            border-radius: 24px 24px 8px 24px;
            box-shadow:
                0 4px 16px rgba(196, 105, 61, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        .message.assistant {
            background: var(--paper);
            color: var(--ink);
            border-radius: 24px 24px 24px 8px;
            border: 1px solid var(--cream-dark);
            box-shadow: 0 4px 20px rgba(45, 42, 36, 0.06);
            font-family: 'Cormorant Garamond', serif;
            font-size: 19px;
            line-height: 1.8;
        }

        /* Word reveal animation */
        .word {
            display: inline-block;
            opacity: 0;
            transform: translateY(10px) rotateX(-20deg);
            animation: wordReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes wordReveal {
            to {
                opacity: 1;
                transform: translateY(0) rotateX(0);
            }
        }

        /* Thinking indicator */
        .thinking {
            display: flex;
            flex-direction: column;
            gap: 8px;
            align-items: flex-start;
            animation: messageSlide 0.4s ease;
        }

        .thinking-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0 8px;
        }

        .thinking-label {
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--sage-deep);
        }

        .thinking-box {
            background: var(--paper);
            border: 1px solid var(--cream-dark);
            border-radius: 24px 24px 24px 8px;
            padding: 20px 28px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 4px 20px rgba(45, 42, 36, 0.06);
        }

        .thinking-orb {
            width: 24px;
            height: 24px;
            position: relative;
        }

        .thinking-orb span {
            position: absolute;
            width: 100%;
            height: 100%;
            border: 2px solid var(--sage);
            border-radius: 50%;
            animation: orbPulse 1.5s ease-in-out infinite;
        }

        .thinking-orb span:nth-child(2) { animation-delay: 0.3s; }
        .thinking-orb span:nth-child(3) { animation-delay: 0.6s; }

        @keyframes orbPulse {
            0%, 100% { transform: scale(0.5); opacity: 0; }
            50% { transform: scale(1); opacity: 1; }
        }

        .thinking-text {
            font-family: 'Cormorant Garamond', serif;
            font-size: 17px;
            font-style: italic;
            color: var(--ink-soft);
        }

        /* Input Area */
        .input-area {
            padding: 24px 40px 32px;
            animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .input-wrapper {
            display: flex;
            gap: 16px;
            align-items: flex-end;
            background: var(--paper);
            border: 1px solid var(--cream-dark);
            border-radius: 24px;
            padding: 8px 8px 8px 24px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(45, 42, 36, 0.04);
        }

        .input-wrapper:focus-within {
            border-color: var(--sage);
            box-shadow:
                0 4px 20px rgba(45, 42, 36, 0.04),
                0 0 0 4px var(--sage-pale);
        }

        .input-field {
            flex: 1;
            padding: 16px 0;
            background: transparent;
            border: none;
            font-family: 'DM Sans', sans-serif;
            font-size: 16px;
            color: var(--ink);
            resize: none;
            outline: none;
            min-height: 56px;
            max-height: 160px;
            line-height: 1.5;
        }

        .input-field::placeholder {
            color: var(--ink-muted);
        }

        .send-btn {
            width: 56px;
            height: 56px;
            background: var(--rust);
            border: none;
            border-radius: 18px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            position: relative;
            overflow: hidden;
        }

        .send-btn::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, transparent, rgba(255,255,255,0.2));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .send-btn:hover::before { opacity: 1; }

        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 24px rgba(196, 105, 61, 0.3);
        }

        .send-btn:active {
            transform: scale(0.98);
        }

        .send-btn:disabled {
            background: var(--cream-dark);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .send-btn svg {
            width: 22px;
            height: 22px;
            stroke: var(--paper);
            stroke-width: 2.5;
            fill: none;
            transition: transform 0.3s ease;
        }

        .send-btn:hover:not(:disabled) svg {
            transform: translateX(2px);
        }

        .send-btn:disabled svg {
            stroke: var(--ink-muted);
        }

        /* Speaker button for TTS */
        .speak-btn {
            margin-top: 8px;
            width: 28px;
            height: 28px;
            background: var(--sage-pale);
            border: 1px solid var(--sage-light);
            border-radius: 8px;
            cursor: pointer;
            opacity: 0.6;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .speak-btn:hover {
            opacity: 1;
            background: var(--sage);
            border-color: var(--sage);
        }

        .speak-btn:hover svg {
            stroke: var(--paper);
        }

        .speak-btn.speaking {
            opacity: 1;
            background: var(--rust);
            border-color: var(--rust);
            animation: pulse-speaker 1s ease-in-out infinite;
        }

        .speak-btn.speaking svg {
            stroke: var(--paper);
        }

        @keyframes pulse-speaker {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .speak-btn svg {
            width: 14px;
            height: 14px;
            stroke: var(--sage-deep);
            stroke-width: 2;
            fill: none;
        }

        .speak-btn.loading svg {
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 680px) {
            .header { padding: 20px 24px; }
            .chat-area { padding: 0 24px 24px; }
            .input-area { padding: 16px 24px 24px; }
            .title { font-size: 26px; }
            .welcome h1 { font-size: 36px; }
            .message { max-width: 88%; }
            .status { display: none; }
        }
    </style>
</head>
<body>
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>

    <div class="app">
        <header class="header">
            <div class="logo">
                <div class="logo-ring"></div>
                <span class="logo-inner">W</span>
            </div>
            <div class="brand">
                <h1 class="title">Wyrd</h1>
                <p class="subtitle">Fate Woven in Words</p>
            </div>
            <div class="status">
                <span class="status-dot"></span>
                <span>106M Parameters</span>
            </div>
        </header>

        <main class="chat-area" id="chat">
            <div class="welcome" id="welcome">
                <div class="welcome-mark">
                    <svg viewBox="0 0 100 100">
                        <path class="mark-fill" d="M50 10 C20 10 10 40 10 50 C10 80 40 90 50 90 C80 90 90 60 90 50 C90 20 60 10 50 10"/>
                        <path class="mark-path" d="M50 10 C20 10 10 40 10 50 C10 80 40 90 50 90 C80 90 90 60 90 50 C90 20 60 10 50 10"/>
                        <path class="mark-path" d="M30 50 Q50 30 70 50 Q50 70 30 50" style="animation-delay: 1s"/>
                    </svg>
                </div>
                <h1>Speak, and <em>fate</em> answers</h1>
                <p>Wyrd weaves destiny from whispered words. Offer a thread, and watch the tapestry unfold.</p>
                <div class="prompts">
                    <button class="prompt-btn" onclick="usePrompt(this)">
                        <span>Once upon a time, in a forest deep...</span>
                    </button>
                    <button class="prompt-btn" onclick="usePrompt(this)">
                        <span>There was a little rabbit who dreamed of...</span>
                    </button>
                    <button class="prompt-btn" onclick="usePrompt(this)">
                        <span>On a morning painted gold, a child found...</span>
                    </button>
                </div>
            </div>
        </main>

        <div class="input-area">
            <div class="input-wrapper">
                <textarea
                    class="input-field"
                    id="input"
                    placeholder="Whisper to the loom..."
                    rows="1"
                ></textarea>
                <button class="send-btn" id="send">
                    <svg viewBox="0 0 24 24">
                        <path d="M5 12h14M13 5l7 7-7 7"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        const welcome = document.getElementById('welcome');

        let isGenerating = false;
        let currentUtterance = null;
        window.ttsEnabled = 'speechSynthesis' in window;  // Browser TTS - always available!

        function speakText(text, btn) {
            // Stop current speech if playing
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
                document.querySelectorAll('.speak-btn.speaking').forEach(b => b.classList.remove('speaking'));
            }

            const utterance = new SpeechSynthesisUtterance(text);
            currentUtterance = utterance;

            // Get available voices and pick a fun one
            const voices = speechSynthesis.getVoices();
            // Try to find a dramatic/funny voice, fallback to default
            const preferredVoice = voices.find(v =>
                v.name.includes('Google') ||
                v.name.includes('Microsoft') ||
                v.name.includes('Zira') ||
                v.name.includes('David')
            ) || voices[0];

            if (preferredVoice) utterance.voice = preferredVoice;
            utterance.rate = 0.9;   // Slightly slow for dramatic effect
            utterance.pitch = 1.0;

            btn.classList.add('speaking');

            utterance.onend = () => {
                btn.classList.remove('speaking');
                currentUtterance = null;
            };

            utterance.onerror = () => {
                btn.classList.remove('speaking');
                currentUtterance = null;
            };

            speechSynthesis.speak(utterance);
        }

        // Load voices (they load async on some browsers)
        speechSynthesis.getVoices();

        function usePrompt(btn) {
            input.value = btn.querySelector('span').textContent;
            input.focus();
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 160) + 'px';
        }

        function getTime() {
            return new Date().toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true
            }).toLowerCase();
        }

        function addMessage(text, role, animate = false) {
            if (welcome) {
                welcome.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                welcome.style.opacity = '0';
                welcome.style.transform = 'translateY(-20px)';
                setTimeout(() => welcome.remove(), 400);
            }

            const group = document.createElement('div');
            group.className = 'message-group ' + role;

            const meta = document.createElement('div');
            meta.className = 'message-meta';

            const author = document.createElement('span');
            author.className = 'message-author';
            author.textContent = role === 'user' ? 'You' : 'Wyrd';

            const time = document.createElement('span');
            time.className = 'message-time';
            time.textContent = getTime();

            meta.appendChild(author);
            meta.appendChild(time);

            const msg = document.createElement('div');
            msg.className = 'message ' + role;

            group.appendChild(meta);
            group.appendChild(msg);

            // Add speaker button for assistant messages if TTS enabled
            if (role === 'assistant' && window.ttsEnabled) {
                const speakBtn = document.createElement('button');
                speakBtn.className = 'speak-btn';
                speakBtn.title = 'Listen';
                speakBtn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';
                speakBtn.onclick = () => speakText(text, speakBtn);
                group.appendChild(speakBtn);  // Appears after message in flow
            }

            chat.appendChild(group);

            if (animate && role === 'assistant') {
                revealWords(msg, text);
            } else {
                msg.textContent = text;
            }

            chat.scrollTop = chat.scrollHeight;
            return msg;
        }

        function revealWords(element, text) {
            element.innerHTML = '';
            const words = text.split(' ');

            words.forEach((word, i) => {
                const span = document.createElement('span');
                span.className = 'word';
                span.textContent = word + ' ';
                span.style.animationDelay = (i * 60) + 'ms';
                element.appendChild(span);
            });

            // Scroll as words appear
            const scrollInterval = setInterval(() => {
                chat.scrollTop = chat.scrollHeight;
            }, 50);

            setTimeout(() => clearInterval(scrollInterval), words.length * 60 + 500);
        }

        function showThinking() {
            const thinking = document.createElement('div');
            thinking.className = 'thinking';
            thinking.id = 'thinking';

            thinking.innerHTML = `
                <div class="thinking-meta">
                    <span class="thinking-label">Wyrd</span>
                </div>
                <div class="thinking-box">
                    <div class="thinking-orb">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <span class="thinking-text">The threads stir...</span>
                </div>
            `;

            chat.appendChild(thinking);
            chat.scrollTop = chat.scrollHeight;
        }

        function hideThinking() {
            const thinking = document.getElementById('thinking');
            if (thinking) {
                thinking.style.transition = 'opacity 0.3s ease';
                thinking.style.opacity = '0';
                setTimeout(() => thinking.remove(), 300);
            }
        }

        async function sendMessage() {
            const text = input.value.trim();
            if (!text || isGenerating) return;

            isGenerating = true;
            sendBtn.disabled = true;
            input.value = '';
            input.style.height = 'auto';

            addMessage(text, 'user');

            await new Promise(r => setTimeout(r, 300));
            showThinking();

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });

                const data = await res.json();
                hideThinking();
                await new Promise(r => setTimeout(r, 200));
                addMessage(data.response, 'assistant', true);
            } catch (e) {
                hideThinking();
                addMessage('The connection was lost. Please try again.', 'assistant');
            } finally {
                isGenerating = false;
                sendBtn.disabled = false;
                input.focus();
            }
        }

        sendBtn.addEventListener('click', sendMessage);

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 160) + 'px';
        });

        // Initial focus with delay for animation
        setTimeout(() => input.focus(), 1000);
    </script>
</body>
</html>'''


def load_model(checkpoint_path, device):
    global MODEL, TOKENIZER, DEVICE
    print(f"Loading model from {checkpoint_path}...")

    DEVICE = torch.device(device)
    TOKENIZER = ChimeraTokenizer()

    # Load weights first to detect config
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Strip _orig_mod. prefix from compiled model checkpoints
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned_state_dict[k[len("_orig_mod."):]] = v
        else:
            cleaned_state_dict[k] = v

    # Auto-detect config from state dict
    # Count layers and check d_model
    layer_nums = [int(k.split(".")[1]) for k in cleaned_state_dict.keys() if k.startswith("layers.")]
    n_layers = max(layer_nums) + 1 if layer_nums else 12
    d_model = cleaned_state_dict.get("embed_tokens.weight", torch.zeros(1, 768)).shape[1]

    print(f"  Detected: {n_layers} layers, d_model={d_model}")

    # Pick config based on detected architecture
    if n_layers == 12 and d_model == 768:
        config = chimera_small()
    elif n_layers == 16 and d_model == 1024:
        config = chimera_medium()
    elif n_layers == 24 and d_model == 2048:
        config = chimera_base()
    else:
        print(f"  Warning: Unknown config, defaulting to small")
        config = chimera_small()

    config.vocab_size = TOKENIZER.vocab_size
    MODEL = Chimera(config)
    MODEL.load_state_dict(cleaned_state_dict)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Loaded: {MODEL.get_num_params():,} params on {DEVICE}")


# ChatML-style tokens (must match train_instruct.py)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are Wyrd, a helpful AI assistant."

# Conversation history for multi-turn
conversation_history = []


def format_prompt(message, system=None):
    """Format a conversation using ChatML template."""
    parts = []

    # System prompt
    sys_content = system or DEFAULT_SYSTEM
    parts.append(f"{IM_START}system\n{sys_content}{IM_END}\n")

    # Add conversation history
    for turn in conversation_history:
        if turn["role"] == "user":
            parts.append(f"{IM_START}user\n{turn['content']}{IM_END}\n")
        else:
            parts.append(f"{IM_START}assistant\n{turn['content']}{IM_END}\n")

    # Add current user message
    parts.append(f"{IM_START}user\n{message}{IM_END}\n")

    # Add generation prompt
    parts.append(f"{IM_START}assistant\n")

    return "".join(parts)


def clean_response(text):
    """Strip chat tokens and clean up the response."""
    # Remove ChatML tokens
    for token in [IM_START, IM_END, "system", "user", "assistant"]:
        text = text.replace(token, "")

    # Clean up whitespace
    text = " ".join(text.split())

    # Remove any trailing incomplete sentences (no ending punctuation)
    lines = text.split(". ")
    if lines and not lines[-1].rstrip().endswith((".", "!", "?", '"')):
        lines = lines[:-1]
    text = ". ".join(lines)
    if text and not text.endswith((".", "!", "?", '"')):
        text += "."

    return text.strip()


@torch.no_grad()
def generate_response(message, max_tokens=200, temperature=0.75, rep_penalty=1.15):
    """Generate a response to the user message using ChatML format."""
    global conversation_history

    # Format with ChatML (how the model was fine-tuned)
    prompt = format_prompt(message)

    input_ids = torch.tensor([TOKENIZER.encode(prompt)], dtype=torch.long, device=DEVICE)

    # Truncate if context too long (keep system prompt + recent turns)
    max_context = 400
    if input_ids.shape[1] > max_context:
        # Trim old history to fit
        while input_ids.shape[1] > max_context and len(conversation_history) > 0:
            conversation_history.pop(0)  # Remove oldest turn
            prompt = format_prompt(message)
            input_ids = torch.tensor([TOKENIZER.encode(prompt)], dtype=torch.long, device=DEVICE)

    generated = []

    for _ in range(max_tokens):
        logits, _ = MODEL(input_ids)
        next_logits = logits[:, -1, :] / max(temperature, 0.1)

        # Repetition penalty on recently generated tokens
        if generated:
            for token_id in set(generated[-50:]):
                next_logits[0, token_id] /= rep_penalty

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        top_p = 0.92
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_logits[indices_to_remove] = float('-inf')

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        token_id = next_token.item()

        # Stop on EOS
        if token_id == TOKENIZER.eos_token_id:
            break

        generated.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Check for end token in decoded text
        decoded = TOKENIZER.decode(generated)
        if IM_END in decoded:
            decoded = decoded.split(IM_END)[0]
            break

        # Stop at natural sentence endings (after minimum length)
        if len(generated) > 30:
            if decoded.rstrip().endswith((".", "!", "?")):
                if len(generated) > 80 or decoded.count(".") >= 3:
                    break

    if generated:
        response = TOKENIZER.decode(generated)
        response = clean_response(response)

        # Add to conversation history (keep last 6 turns max)
        conversation_history.append({"role": "user", "content": message})
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > 12:  # 6 turns = 12 messages
            conversation_history = conversation_history[-12:]

        return response
    return "The threads are tangled... try again."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(HTML)

    def do_POST(self):
        if self.path == '/chat':
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))

            message = data.get('message', '')
            response = generate_response(message)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'response': response}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Chimera Chat UI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/instruct_step_400.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (0.0.0.0 for network access)")
    args = parser.parse_args()

    load_model(args.checkpoint, args.device)

    # Get local IP for display
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"

    server = HTTPServer((args.host, args.port), Handler)
    local_ip = get_local_ip()
    print(f"\n>>> Wyrd is listening <<<")
    print(f"    Local:   http://127.0.0.1:{args.port}")
    print(f"    Network: http://{local_ip}:{args.port}")
    print(f"    Voice:   Browser TTS (it's hilariously generic)\n")
    server.serve_forever()


if __name__ == '__main__':
    main()
