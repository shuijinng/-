// frontend/script.js

// 获取 DOM 元素
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-btn');

// 发送消息
function sendMessage() {
  const question = userInput.value.trim();
  if (!question) return;

  // 添加用户消息（靠右）
  addMessage(question, 'user');

  // 清空输入框
  userInput.value = '';

  // 显示机器人“正在思考”（靠左）
  const botTyping = addTypingIndicator();

  // 调用后端 API
  fetch('http://localhost:8001/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  })
  .then(res => {
    if (!res.ok) throw new Error('网络响应失败');
    return res.json();
  })
  .then(data => {
    removeTypingIndicator(botTyping);
    typeWriterEffect(data.answer || "抱歉，我没有收到回答。");
  })
  .catch(err => {
    removeTypingIndicator(botTyping);
    addMessage("网络错误，请稍后再试。", 'bot');
  });
}

// 添加消息（支持 user 和 bot）
function addMessage(text, sender) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${sender}`;
  const avatar = sender === 'bot' ? '🤖' : '👤';
  msgDiv.innerHTML = `
    <span class="avatar">${avatar}</span>
    <div class="bubble">${text}</div>
  `;
  chatBox.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv;
}

// 添加“正在输入”动画
function addTypingIndicator() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'message bot';
  typingDiv.innerHTML = `
    <span class="avatar">🤖</span>
    <div class="bubble typing">正在思考...</div>
  `;
  chatBox.appendChild(typingDiv);
  scrollToBottom();
  return typingDiv;
}

// 移除“正在输入”动画
function removeTypingIndicator(el) {
  if (el && el.parentNode) {
    el.remove();
  }
}

// 打字机效果：逐字显示机器人回复（修复版）
function typeWriterEffect(text) {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message bot';
  msgDiv.innerHTML = `
    <span class="avatar">🤖</span>
    <div class="bubble typing"></div>
  `;
  chatBox.appendChild(msgDiv);
  const bubble = msgDiv.querySelector('.bubble');

  let currentText = '';
  let i = 0;
  const isLongText = text.length > 120;
  const typingSpeed = isLongText ? 35 : 20; // 长文本慢一点

  const timer = setInterval(() => {
    if (i < text.length) {
      currentText += text.charAt(i);
      bubble.textContent = currentText;
      scrollToBottom();
      i++;
    } else {
      clearInterval(timer);
      bubble.classList.remove('typing');
    }
  }, typingSpeed);
}

// 平滑滚动到底部
function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

// 事件监听
sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
});