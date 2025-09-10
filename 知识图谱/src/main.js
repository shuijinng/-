// src/main.js
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Text } from 'troika-three-text';

// 🔽 动态导入 vis-network（因为它是 UMD 模块，不支持 ESM 直接 import）
let vis;
async function loadVis() {
  if (!vis) {
    const module = await import('https://unpkg.com/vis-network/standalone/umd/vis-network.min.js');
    vis = module;
  }
  return vis;
}

// 全局变量
let network = null;
let scene, camera, renderer, controls;
const tagObjects = [];

// ✅ 新增：预加载字体
async function loadFont(fontUrl) {
  return new Promise((resolve, reject) => {
    const font = new FontFace('NotoSansSC', `url(${fontUrl})`);
    font.load().then(() => {
      document.fonts.add(font);
      resolve(font);
    }).catch(err => reject(err));
  });
}

// ✅ 改为使用 /api/entities 获取实体（更准确）
async function loadEntities() {
  try {
    const res = await fetch("/api/entities", {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.success && Array.isArray(data.names)) {
      console.log(`✅ 加载 ${data.count} 个实体`);
      return data.names;
    } else {
      console.warn("⚠️ 后端返回格式异常", data);
      return [];
    }
  } catch (e) {
    console.error("❌ 获取实体失败:", e);
    return [];
  }
}

// ✅ 修复：确保字体加载后再创建标签
function createTag(text, x, y, z, onClick) {
  const tag = new Text();
  tag.text = text;
  tag.font = 'https://fonts.gstatic.com/s/notosanssc/v20/5hjMip6lL1f3pYVv5KQ6OcJG9KoK4n3qjW21p2H4.woff2';
  tag.fontSize = 1.6;

  // 随机颜色
  const hue = Math.random();
  const saturation = 0.8;
  const lightness = 0.7;
  const color = new THREE.Color().setHSL(hue, saturation, lightness);

  tag.material = new THREE.MeshStandardMaterial({
    color: color,
    metalness: 0.3,
    roughness: 0.4
  });

  tag.anchorX = 'center';
  tag.anchorY = 'middle';
  tag.position.set(x, y, z);
  tag.userData = { text, onClick };

  // 🔽 sync() 会在字体加载后由外部触发
  scene.add(tag);
  tagObjects.push(tag);

  // 返回 tag 对象，便于后续 sync
  return tag;
}

// 初始化 3D 标签云
async function init3DWordCloud() {
  const container = document.getElementById("wordcloud-container");
  while (container.firstChild) container.removeChild(container.firstChild);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    75,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  camera.position.z = 25;

  // 灯光
  const ambientLight = new THREE.AmbientLight(0x404040, 1);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
  directionalLight.position.set(10, 10, 10).normalize();
  scene.add(directionalLight);

  // 控制器
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.rotateSpeed = 0.5;

  try {
    // 🔽 1. 预加载字体
    await loadFont('https://fonts.gstatic.com/s/notosanssc/v20/5hjMip6lL1f3pYVv5KQ6OcJG9KoK4n3qjW21p2H4.woff2');
    console.log('✅ 字体加载完成');

    // 🔽 2. 加载实体
    const names = await loadEntities();
    const radius = 14;

    // 🔽 3. 创建标签并触发 sync
    names.forEach((name, i) => {
      const phi = Math.acos(-1 + (2 * i) / names.length);
      const theta = Math.sqrt(names.length * Math.PI) * phi;
      const x = radius * Math.cos(theta) * Math.sin(phi);
      const y = radius * Math.sin(theta) * Math.sin(phi);
      const z = radius * Math.cos(phi);

      const tag = createTag(name, x, y, z, () => askQuestion(name));
      tag.sync(); // ✅ 在字体加载完成后 sync
    });
  } catch (err) {
    console.error('❌ 初始化失败:', err);
    document.getElementById('answer').innerHTML = `字体加载失败: ${err.message}`;
  }

  // 动画循环
  function animate() {
    requestAnimationFrame(animate);

    // 自动缓慢旋转
    scene.rotation.y += 0.001;

    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // 窗口自适应
  function onWindowResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
  window.addEventListener('resize', () => {
    setTimeout(onWindowResize, 100);
  }, { passive: true });
}

// 向后端提问
async function askQuestion(entity) {
  const answerDiv = document.getElementById("answer");
  const graphDiv = document.getElementById("graph");
  answerDiv.style.display = "block";
  answerDiv.innerHTML = `🔍 正在查询 <strong>${entity}</strong> 的相关信息...`;
  graphDiv.style.display = "none";
  if (network) {
    network.destroy();
    network = null;
  }

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: entity }),
    });
    const data = await res.json();

    answerDiv.innerHTML = data.answer || "⚠️ 未获取到有效回答。";

    if (data.graph && Object.keys(data.graph).length > 0) {
      graphDiv.style.display = "block";
      await drawGraph(data.graph); // 等待 vis 加载
    }
  } catch (error) {
    answerDiv.innerHTML = `❌ 请求失败：<br><small>${error.message}</small>`;
    console.error("Fetch error:", error);
  }
}

// 绘制知识图谱
async function drawGraph(graphData) {
  await loadVis(); // 确保 vis 已加载

  const container = document.getElementById("graph");
  const data = {
    nodes: new vis.DataSet(graphData.nodes),
    edges: new vis.DataSet(graphData.edges)
  };

  const options = {
    nodes: {
      shape: 'dot',
      size: 18,
      font: { size: 14, color: '#ffffff', face: 'Microsoft YaHei' },
      borderWidth: 2,
      shadow: true
    },
    edges: {
      color: { color: '#aaaaaa', highlight: '#ffffff' },
      font: { size: 11, color: '#ffffff' },
      arrows: { to: { enabled: true, scaleFactor: 0.8 } },
      smooth: true,
      shadow: true
    },
    groups: {
      movie: { color: { background: '#e91e63', border: '#c2185b' }, shape: 'ellipse' },
      person: { color: { background: '#4caf50', border: '#388e3c' }, shape: 'icon', icon: { face: 'FontAwesome', code: '\uf007' } },
      company: { color: { background: '#2196f3', border: '#1976d2' }, shape: 'box' }
    },
    physics: { repulsion: { nodeDistance: 140, strength: 1000 }, stabilization: { iterations: 180 } },
    interaction: { hover: true }
  };

  network = new vis.Network(container, data, options);
}

// 启动
window.addEventListener('load', () => {
  setTimeout(init3DWordCloud, 100);
});