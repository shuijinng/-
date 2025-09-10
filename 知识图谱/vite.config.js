// vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    open: true,
    proxy: {
      // 🚀 保留原来的 /ask 代理
      '/ask': {
        target: 'http://127.0.0.1:8848',
        changeOrigin: true,
        secure: false,
      },
      // ✅ 新增：代理所有 /api 开头的请求
      '/api': {
        target: 'http://127.0.0.1:8848',  // 指向 Flask 后端
        changeOrigin: true,               // 支持跨域
        secure: false,                    // 不验证 HTTPS 证书
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});